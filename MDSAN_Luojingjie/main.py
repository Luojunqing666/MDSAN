import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
from DSAN import DSAN
import data_loader



def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer):
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_source = data_source.type(torch.float32)
        label_source = [int(x) for x in label_source]
        label_source = torch.from_numpy(np.array(label_source)).type(torch.int64)
        data_target, _ = iter_target.next()
        data_target = data_target.type(torch.float)
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        # print(i,type(data_source),type(data_target))
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)
        # print(F.log_softmax(label_source_pred, dim=1))

        # label_source = torch.from_numpy(np.eye(10)[label_source.cpu()]).cuda()

        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        # lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        lambd = (-4) / (1 + math.sqrt(epoch /(args.nepoch+1-epoch))) + 4
        loss = loss_cls  + args.weight * lambd * loss_lmmd
        print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')

def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            target = [int(x) for x in target]
            target = torch.from_numpy(np.array(target)).type(torch.int64)
            data = data.type(torch.float32)
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset', default='这里输入你的数据地址')
    parser.add_argument('--src', type=str, help='Source domain', default='Source domain')
    parser.add_argument('--tar', type=str, help='Target domain', default='Target domain')
    parser.add_argument('--nclass', type=int, help='Number of classes', default=10) # Please make changes based on health status here
    parser.add_argument('--batch_size', type=float, help='batch size', default=256)###32
    parser.add_argument('--nepoch', type=int, help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.001, 0.001])
    parser.add_argument('--early_stop', type=int, help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int, help='Seed', default=2)
    parser.add_argument('--weight', type=float, help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float, help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument('--log_interval', type=int, help='Log interval', default=10)
    parser.add_argument('--gpu', type=str, help='GPU ID', default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    # SEED = args.seed
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataloaders = load_data(args.root_path, args.src,
                            args.tar, args.batch_size)
    model = DSAN(num_classes=args.nclass).cuda()
    
    correct = 0
    stop = 0

    if args.bottleneck:
        optimizer = torch.optim.Adam([
            {'params': model.feature_layers.parameters(),},
            # {'params': model.bottle.parameters(), 'lr': args.lr[1]},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0],  weight_decay=args.decay)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.feature_layers.parameters(), },
            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0],  weight_decay=args.decay)
    # momentum = args.momentum,
    for epoch in range(1, args.nepoch + 1):
        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

        train_epoch(epoch, model, dataloaders, optimizer)
        t_correct = test(model, dataloaders[-1])
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'model.pkl')
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        # if stop >= args.early_stop:
        #     print(
        #         f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
        #     break
