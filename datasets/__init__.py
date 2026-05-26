"""
CWRU Bearing Dataset loader for domain adaptation.

Loads vibration signals from .mat files and creates PyTorch DataLoaders
for source and target domains with different operating conditions.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


# ============================================================
# CWRU Dataset Configuration
# ============================================================

SIGNAL_SIZE = 1024

# File names indexed by load condition (0-3 correspond to 1797/1772/1750/1730 RPM)
CWRU_FILES = {
    0: ["97.mat", "105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat", "234.mat"],
    1: ["98.mat", "106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat", "235.mat"],
    2: ["99.mat", "107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat", "236.mat"],
    3: ["100.mat", "108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat", "237.mat"],
}

DATASET_DIRS = [
    "12k Drive End Bearing Fault Data",
    "12k Fan End Bearing Fault Data",
    "48k Drive End Bearing Fault Data",
    "Normal Baseline Data",
]

AXIS_SUFFIX = "_DE_time"
LABELS = list(range(10))


# ============================================================
# Data Loading Functions
# ============================================================

def load_mat_signal(filepath, filename, normalize_type="-1-1"):
    """
    Load and normalize vibration signal from a .mat file.

    Args:
        filepath: Full path to the .mat file.
        filename: Filename (used to determine the data key).
        normalize_type: Normalization method ("-1-1" or "0-1").

    Returns:
        Normalized signal as 1D numpy array.
    """
    datanumber = filename.split(".")[0]
    if int(datanumber) < 100:
        key = "X0" + datanumber + AXIS_SUFFIX
    else:
        key = "X" + datanumber + AXIS_SUFFIX

    signal = loadmat(filepath)[key].flatten()

    # Normalize
    if normalize_type == "-1-1":
        signal = 2 * (signal - signal.min()) / (signal.max() - signal.min() + 1e-8) - 1
    elif normalize_type == "0-1":
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)

    return signal


def segment_signal(signal, segment_length=SIGNAL_SIZE, stride=None):
    """
    Segment a long signal into fixed-length samples.

    Args:
        signal: 1D numpy array.
        segment_length: Length of each segment.
        stride: Stride between segments (default: same as segment_length).

    Returns:
        List of signal segments.
    """
    if stride is None:
        stride = segment_length

    segments = []
    start = 0
    while start + segment_length <= len(signal):
        segments.append(signal[start:start + segment_length])
        start += stride
    return segments


def load_domain_data(root_path, load_conditions, signal_size=SIGNAL_SIZE,
                     normalize_type="-1-1"):
    """
    Load all data for specified load conditions (domain).

    Args:
        root_path: Root directory of CWRU dataset.
        load_conditions: List of load condition indices (e.g., [0] for 1797 RPM).
        signal_size: Length of each signal segment.
        normalize_type: Normalization method.

    Returns:
        Tuple of (data_list, label_list).
    """
    data, labels = [], []

    for condition in load_conditions:
        files = CWRU_FILES[condition]
        for idx, filename in enumerate(files):
            # First file (index 0) is normal data
            if idx == 0:
                filepath = os.path.join(root_path, DATASET_DIRS[3], filename)
            else:
                filepath = os.path.join(root_path, DATASET_DIRS[0], filename)

            signal = load_mat_signal(filepath, filename, normalize_type)
            segments = segment_signal(signal, signal_size)

            data.extend(segments)
            labels.extend([LABELS[idx]] * len(segments))

    return data, labels


# ============================================================
# PyTorch Dataset
# ============================================================

class VibrationDataset(Dataset):
    """
    PyTorch Dataset for vibration signal segments.

    Args:
        data_list: List of signal segments (numpy arrays).
        label_list: List of integer labels.
    """

    def __init__(self, data_list, label_list):
        self.data = data_list
        self.labels = label_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        # Reshape to [1, signal_length] for Conv1d
        signal = np.array(signal, dtype=np.float32).reshape(1, -1)
        return torch.from_numpy(signal), label


# ============================================================
# DataLoader Factory
# ============================================================

def create_dataloaders(root_path, source_conditions, target_conditions,
                       batch_size=256, signal_size=SIGNAL_SIZE,
                       normalize_type="-1-1", num_workers=1):
    """
    Create DataLoaders for source and target domains.

    Args:
        root_path: Root directory of CWRU dataset.
        source_conditions: List of load condition indices for source domain.
        target_conditions: List of load condition indices for target domain.
        batch_size: Batch size.
        signal_size: Signal segment length.
        normalize_type: Normalization method.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (source_loader, target_train_loader, target_test_loader).
    """
    kwargs = {'num_workers': num_workers, 'pin_memory': True}

    # Source domain
    src_data, src_labels = load_domain_data(
        root_path, source_conditions, signal_size, normalize_type
    )
    src_dataset = VibrationDataset(src_data, src_labels)
    source_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True,
                               drop_last=True, **kwargs)

    # Target domain (train: shuffled, test: same data for evaluation)
    tar_data, tar_labels = load_domain_data(
        root_path, target_conditions, signal_size, normalize_type
    )
    tar_dataset = VibrationDataset(tar_data, tar_labels)
    target_train_loader = DataLoader(tar_dataset, batch_size=batch_size, shuffle=True,
                                     drop_last=True, **kwargs)
    target_test_loader = DataLoader(tar_dataset, batch_size=batch_size, shuffle=False,
                                    **kwargs)

    return source_loader, target_train_loader, target_test_loader
