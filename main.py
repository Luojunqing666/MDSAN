"""
MDSAN: Modified Deep Subdomain Adaptation Network for fault diagnosis.

Main training script for unsupervised cross-domain fault diagnosis
of bearings under speed fluctuation.

Usage:
    python main.py --data_dir /path/to/CWRU \
        --source_condition 3 --target_condition 2 \
        --num_classes 10 --epochs 200 --batch_size 256

    # Custom transfer task (e.g., 1797 RPM -> 1750 RPM)
    python main.py --data_dir /path/to/CWRU \
        --source_condition 0 --target_condition 2 \
        --num_classes 10 --epochs 200
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from models import MDSAN
from datasets import create_dataloaders


def train_epoch(epoch, model, dataloaders, optimizer, args):
    """
    Train for one epoch with domain adaptation.

    Args:
        epoch: Current epoch number.
        model: MDSAN model.
        dataloaders: Tuple of (source_loader, target_train_loader, target_test_loader).
        optimizer: Optimizer.
        args: Command-line arguments.
    """
    model.train()
    source_loader, target_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)

    for i in range(1, num_iter):
        # Load source batch
        data_source, label_source = next(iter_source)
        data_source = data_source.float().cuda()
        label_source = label_source.long().cuda()

        # Load target batch (cycle if shorter)
        try:
            data_target, _ = next(iter_target)
        except StopIteration:
            iter_target = iter(target_loader)
            data_target, _ = next(iter_target)
        data_target = data_target.float().cuda()

        # Forward pass
        optimizer.zero_grad()
        s_pred, loss_lmmd = model(data_source, data_target, label_source)

        # Classification loss
        loss_cls = F.nll_loss(F.log_softmax(s_pred, dim=1), label_source)

        # Adaptive weight schedule (modified from paper)
        lambd = (-4) / (1 + math.sqrt(epoch / (args.epochs + 1 - epoch))) + 4

        # Total loss
        loss = loss_cls + args.weight * lambd * loss_lmmd
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(f"  Epoch [{epoch:3d}] Iter [{i:4d}/{num_iter}] | "
                  f"Loss: {loss.item():.4f} (cls: {loss_cls.item():.4f}, "
                  f"lmmd: {loss_lmmd.item():.4f})")


def evaluate(model, dataloader):
    """
    Evaluate model on target domain.

    Args:
        model: MDSAN model.
        dataloader: Target test DataLoader.

    Returns:
        Number of correct predictions.
    """
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.float().cuda()
            target = target.long().cuda()

            pred = model.predict(data)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            correct += pred.argmax(dim=1).eq(target).sum().item()

    test_loss /= len(dataloader)
    total = len(dataloader.dataset)
    accuracy = 100. * correct / total
    print(f"  [Test] Loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return correct


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDSAN: Modified Deep Subdomain Adaptation Network for fault diagnosis"
    )

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of CWRU dataset.")
    parser.add_argument("--source_condition", type=int, nargs='+', default=[3],
                        help="Source domain load condition(s). 0=1797rpm, 1=1772rpm, 2=1750rpm, 3=1730rpm.")
    parser.add_argument("--target_condition", type=int, nargs='+', default=[2],
                        help="Target domain load condition(s).")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of fault classes.")
    parser.add_argument("--normalize", type=str, default="-1-1",
                        choices=["-1-1", "0-1"], help="Signal normalization method.")

    # Training
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay.")
    parser.add_argument("--weight", type=float, default=0.5,
                        help="Weight for LMMD adaptation loss.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval (iterations).")

    # System
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID.")
    parser.add_argument("--seed", type=int, default=2, help="Random seed.")
    parser.add_argument("--save_dir", type=str, default="./results", help="Save directory.")

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("MDSAN: Modified Deep Subdomain Adaptation Network")
    print("=" * 60)
    print(f"Config: {vars(args)}\n")

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print(f"Transfer task: condition {args.source_condition} -> {args.target_condition}")
    dataloaders = create_dataloaders(
        root_path=args.data_dir,
        source_conditions=args.source_condition,
        target_conditions=args.target_condition,
        batch_size=args.batch_size,
        normalize_type=args.normalize,
    )
    print(f"Source samples: {len(dataloaders[0].dataset)}, "
          f"Target samples: {len(dataloaders[2].dataset)}\n")

    # Build model
    model = MDSAN(num_classes=args.num_classes).cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer with layer-wise learning rates
    optimizer = torch.optim.Adam([
        {'params': model.feature_layers.parameters(), 'lr': args.lr},
        {'params': model.cls_fc.parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    # Training loop
    best_correct = 0
    for epoch in range(1, args.epochs + 1):
        # Learning rate decay
        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr / math.pow(1 + 10 * (epoch - 1) / args.epochs, 0.75)

        train_epoch(epoch, model, dataloaders, optimizer, args)
        correct = evaluate(model, dataloaders[-1])

        if correct > best_correct:
            best_correct = correct
            save_path = os.path.join(args.save_dir, "mdsan_best.pt")
            torch.save(model.state_dict(), save_path)

        total = len(dataloaders[-1].dataset)
        print(f"  [Best] {best_correct}/{total} ({100. * best_correct / total:.2f}%)\n")

    # Final results
    total = len(dataloaders[-1].dataset)
    print("=" * 60)
    print(f"Final best accuracy: {100. * best_correct / total:.2f}%")
    print(f"Model saved to: {os.path.join(args.save_dir, 'mdsan_best.pt')}")


if __name__ == "__main__":
    main()
