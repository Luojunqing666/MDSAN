"""
LMMD (Local Maximum Mean Discrepancy) loss for subdomain adaptation.

Implements class-conditional MMD that aligns subdomain distributions
between source and target domains, enabling fine-grained domain adaptation.
"""

import torch
import torch.nn as nn
import numpy as np


class LMMDLoss(nn.Module):
    """
    Local Maximum Mean Discrepancy (LMMD) loss.

    Unlike global MMD which aligns marginal distributions, LMMD aligns
    class-conditional (subdomain) distributions by weighting kernel values
    with class membership probabilities.

    Args:
        num_classes: Number of fault classes.
        kernel_type: Kernel type ('rbf' for Gaussian).
        kernel_mul: Bandwidth multiplier base.
        kernel_num: Number of Gaussian kernels.
        fix_sigma: Fixed bandwidth (None for adaptive).
    """

    def __init__(self, num_classes=10, kernel_type='rbf', kernel_mul=2.0,
                 kernel_num=5, fix_sigma=None):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def gaussian_kernel(self, source, target):
        """
        Compute multi-kernel Gaussian kernel matrix.

        Args:
            source: Source features [batch_size, feature_dim].
            target: Target features [batch_size, feature_dim].

        Returns:
            Sum of Gaussian kernel matrices.
        """
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def compute_class_weights(self, s_label, t_label, batch_size):
        """
        Compute class-conditional weight matrices for LMMD.

        Args:
            s_label: Source labels (integer) [batch_size].
            t_label: Target soft labels (probabilities) [batch_size, num_classes].
            batch_size: Batch size.

        Returns:
            Tuple of (weight_ss, weight_tt, weight_st) as float32 arrays.
        """
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = np.eye(self.num_classes)[s_sca_label]
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, self.num_classes)
        s_sum[s_sum == 0] = 100  # Avoid division by zero
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, self.num_classes)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        # Only align classes present in both domains
        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, self.num_classes))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss /= length
            weight_tt /= length
            weight_st /= length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])

        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

    def forward(self, source, target, s_label, t_label):
        """
        Compute LMMD loss between source and target features.

        Args:
            source: Source domain features [batch_size, feature_dim].
            target: Target domain features [batch_size, feature_dim].
            s_label: Source labels (integer) [batch_size].
            t_label: Target soft predictions (after softmax) [batch_size, num_classes].

        Returns:
            Scalar LMMD loss.
        """
        batch_size = source.size(0)
        device = source.device

        weight_ss, weight_tt, weight_st = self.compute_class_weights(
            s_label, t_label, batch_size
        )
        weight_ss = torch.from_numpy(weight_ss).to(device)
        weight_tt = torch.from_numpy(weight_tt).to(device)
        weight_st = torch.from_numpy(weight_st).to(device)

        kernels = self.gaussian_kernel(source, target)

        loss = torch.tensor([0.0], device=device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss

        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss
