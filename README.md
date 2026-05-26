# MDSAN

**Modified Deep Subdomain Adaptation Network for Unsupervised Cross-Domain Fault Diagnosis of Bearings under Speed Fluctuation**

[![Paper](https://img.shields.io/badge/Paper-Journal%20of%20Manufacturing%20Systems-blue)](https://doi.org/10.1016/j.jmsy.2022.09.004)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Jingjie Luo, Haidong Shao*, Hongru Cao, Xingkai Chen, Baoping Cai, Bin Liu
>
> *Journal of Manufacturing Systems*, Volume 65, 2022, Pages 180-191

---

## Overview

MDSAN is a domain adaptation framework for unsupervised cross-domain fault diagnosis of bearings under speed fluctuation conditions. It combines a **1D CNN with multi-head self-attention** for feature extraction and **Local Maximum Mean Discrepancy (LMMD)** for subdomain-level distribution alignment, enabling effective knowledge transfer between different operating speeds without target domain labels.

<div align="center">
<img src="figures/framework.png" width="700" />
<p><em>Framework of the proposed MDSAN method</em></p>
</div>

## Key Features

- **Subdomain Adaptation**: LMMD aligns class-conditional distributions (subdomains) rather than marginal distributions, preserving discriminative structure during adaptation
- **Self-Attention Enhancement**: Multi-head self-attention captures long-range dependencies in vibration signals for more robust feature extraction
- **Adaptive Loss Weighting**: Dynamic trade-off schedule between classification and adaptation losses during training
- **Speed Fluctuation Robustness**: Designed for cross-speed transfer tasks where source and target domains operate at different rotational speeds

## Project Structure

```
MDSAN/
├── main.py                  # Training entry point
├── requirements.txt         # Dependencies
├── models/
│   ├── __init__.py
│   ├── mdsan.py             # MDSAN model (feature extractor + classifier + LMMD)
│   ├── backbone.py          # 1D CNN + Multi-head Self-Attention
│   └── lmmd.py              # Local MMD loss for subdomain alignment
├── datasets/
│   └── __init__.py          # CWRU dataset loader
├── results/                 # Saved checkpoints (gitignored)
└── figures/                 # Paper figures
```

## Installation

```bash
git clone https://github.com/Luojunqing666/MDSAN.git
cd MDSAN
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- NumPy >= 1.22
- SciPy >= 1.7
- pandas >= 1.4

## Usage

### Basic Training

```bash
# Transfer: Load 3 (1730 RPM) -> Load 2 (1750 RPM)
python main.py --data_dir /path/to/CWRU \
    --source_condition 3 --target_condition 2 \
    --num_classes 10 --epochs 200 --batch_size 256

# Transfer: Load 0 (1797 RPM) -> Load 3 (1730 RPM)
python main.py --data_dir /path/to/CWRU \
    --source_condition 0 --target_condition 3 \
    --num_classes 10 --epochs 200
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | - | Root directory of CWRU dataset |
| `--source_condition` | `3` | Source domain load condition (0=1797rpm, 1=1772rpm, 2=1750rpm, 3=1730rpm) |
| `--target_condition` | `2` | Target domain load condition |
| `--num_classes` | `10` | Number of fault classes |
| `--batch_size` | `256` | Batch size |
| `--epochs` | `200` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--weight` | `0.5` | Weight for LMMD adaptation loss |
| `--weight_decay` | `5e-4` | L2 regularization |
| `--normalize` | `-1-1` | Signal normalization: `-1-1` or `0-1` |
| `--gpu` | `0` | GPU device ID |
| `--seed` | `2` | Random seed |

### Dataset Preparation

Download the [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter) and organize as:

```
CWRU/
├── 12k Drive End Bearing Fault Data/
│   ├── 105.mat
│   ├── 118.mat
│   └── ...
└── Normal Baseline Data/
    ├── 97.mat
    ├── 98.mat
    └── ...
```

## Method

The MDSAN framework consists of three components:

1. **Feature Extractor**: A 4-layer 1D CNN followed by multi-head self-attention captures both local patterns and global dependencies in vibration signals.

2. **Classifier**: A fully-connected layer maps features to fault class predictions.

3. **LMMD Loss**: Unlike global MMD, LMMD computes class-conditional kernel distances weighted by predicted class probabilities, aligning subdomain distributions for fine-grained adaptation.

The total loss is:

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda \cdot w \cdot \mathcal{L}_{LMMD}$$

where $\lambda$ follows an adaptive schedule that increases during training.

## Citation

If you find this work useful, please cite:

```bibtex
@article{luo2022mdsan,
  title     = {Modified DSAN for unsupervised cross-domain fault diagnosis of bearing under speed fluctuation},
  author    = {Luo, Jingjie and Shao, Haidong and Cao, Hongru and Chen, Xingkai and Cai, Baoping and Liu, Bin},
  journal   = {Journal of Manufacturing Systems},
  volume    = {65},
  pages     = {180--191},
  year      = {2022},
  doi       = {10.1016/j.jmsy.2022.09.004},
  publisher = {Elsevier}
}
```

## Contact

- luojingjie@hnu.edu.cn
- luojingjie@sjtu.edu.cn

---

# MDSAN

**改进的深度子域自适应网络用于转速波动下轴承的无监督跨域故障诊断**

[![论文](https://img.shields.io/badge/论文-Journal%20of%20Manufacturing%20Systems-blue)](https://doi.org/10.1016/j.jmsy.2022.09.004)
[![许可证](https://img.shields.io/badge/许可证-MIT-green.svg)](LICENSE)

> 罗景杰, 邵海东*, 曹宏瑞, 陈兴凯, 蔡宝平, 刘斌
>
> *Journal of Manufacturing Systems*, 第65卷, 2022, 第180-191页

---

## 概述

MDSAN 是一个用于转速波动条件下轴承无监督跨域故障诊断的域自适应框架。它结合了**一维CNN + 多头自注意力**进行特征提取，以及**局部最大均值差异（LMMD）**进行子域级分布对齐，能够在无目标域标签的情况下实现不同运行转速之间的有效知识迁移。

<div align="center">
<img src="figures/framework.png" width="700" />
<p><em>所提 MDSAN 方法的框架图</em></p>
</div>

## 主要特点

- **子域自适应**：LMMD 对齐类条件分布（子域）而非边缘分布，在自适应过程中保持判别结构
- **自注意力增强**：多头自注意力捕获振动信号中的长程依赖关系，实现更鲁棒的特征提取
- **自适应损失加权**：训练过程中分类损失和自适应损失之间的动态权衡调度
- **转速波动鲁棒性**：专为源域和目标域在不同转速下运行的跨速度迁移任务设计

## 项目结构

```
MDSAN/
├── main.py                  # 训练入口
├── requirements.txt         # 依赖包
├── models/
│   ├── __init__.py
│   ├── mdsan.py             # MDSAN 模型（特征提取器 + 分类器 + LMMD）
│   ├── backbone.py          # 一维 CNN + 多头自注意力
│   └── lmmd.py              # 局部 MMD 损失（子域对齐）
├── datasets/
│   └── __init__.py          # CWRU 数据集加载器
├── results/                 # 模型检查点（已忽略）
└── figures/                 # 论文图片
```

## 安装

```bash
git clone https://github.com/Luojunqing666/MDSAN.git
cd MDSAN
pip install -r requirements.txt
```

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- NumPy >= 1.22
- SciPy >= 1.7
- pandas >= 1.4

## 使用方法

### 基本训练

```bash
# 迁移任务：载荷 3（1730 RPM）-> 载荷 2（1750 RPM）
python main.py --data_dir /path/to/CWRU \
    --source_condition 3 --target_condition 2 \
    --num_classes 10 --epochs 200 --batch_size 256

# 迁移任务：载荷 0（1797 RPM）-> 载荷 3（1730 RPM）
python main.py --data_dir /path/to/CWRU \
    --source_condition 0 --target_condition 3 \
    --num_classes 10 --epochs 200
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | - | CWRU 数据集根目录 |
| `--source_condition` | `3` | 源域载荷条件（0=1797rpm, 1=1772rpm, 2=1750rpm, 3=1730rpm） |
| `--target_condition` | `2` | 目标域载荷条件 |
| `--num_classes` | `10` | 故障类别数 |
| `--batch_size` | `256` | 批大小 |
| `--epochs` | `200` | 训练轮数 |
| `--lr` | `0.001` | 学习率 |
| `--weight` | `0.5` | LMMD 自适应损失权重 |
| `--weight_decay` | `5e-4` | L2 正则化 |
| `--normalize` | `-1-1` | 信号归一化方式：`-1-1` 或 `0-1` |
| `--gpu` | `0` | GPU 设备 ID |
| `--seed` | `2` | 随机种子 |

### 数据准备

下载 [CWRU 轴承数据集](https://engineering.case.edu/bearingdatacenter) 并按如下方式组织：

```
CWRU/
├── 12k Drive End Bearing Fault Data/
│   ├── 105.mat
│   ├── 118.mat
│   └── ...
└── Normal Baseline Data/
    ├── 97.mat
    ├── 98.mat
    └── ...
```

## 方法

MDSAN 框架由三个组件组成：

1. **特征提取器**：4层一维CNN后接多头自注意力，同时捕获振动信号中的局部模式和全局依赖关系。

2. **分类器**：全连接层将特征映射到故障类别预测。

3. **LMMD 损失**：与全局 MMD 不同，LMMD 计算由预测类别概率加权的类条件核距离，对齐子域分布以实现细粒度自适应。

总损失为：

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda \cdot w \cdot \mathcal{L}_{LMMD}$$

其中 $\lambda$ 遵循在训练过程中逐渐增大的自适应调度。

## 引用

如果本工作对您有帮助，请引用以下论文：

```bibtex
@article{luo2022mdsan,
  title     = {Modified DSAN for unsupervised cross-domain fault diagnosis of bearing under speed fluctuation},
  author    = {Luo, Jingjie and Shao, Haidong and Cao, Hongru and Chen, Xingkai and Cai, Baoping and Liu, Bin},
  journal   = {Journal of Manufacturing Systems},
  volume    = {65},
  pages     = {180--191},
  year      = {2022},
  doi       = {10.1016/j.jmsy.2022.09.004},
  publisher = {Elsevier}
}
```

## 联系方式

- luojingjie@hnu.edu.cn
- luojingjie@sjtu.edu.cn
