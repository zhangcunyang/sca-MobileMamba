# SCA-MobileMamba

基于Mamba架构的侧信道分析（Side-Channel Analysis）项目，专门用于ASCAD数据集的密钥恢复攻击。

## 项目概述

本项目实现了一个轻量级的MobileMamba模型，用于执行侧信道攻击中的深度学习密钥恢复。该模型结合了Mamba的选择性状态空间模型与移动端优化设计，在保持高效性能的同时降低了计算复杂度。

## 主要特性

- **MobileMambaSCA模型**: 基于Mamba架构的侧信道分析模型
- **多尺度卷积特征提取**: 使用33和51内核的多尺度卷积进行特征融合
- **多通道支持**: 支持单通道和多通道侧信道数据
- **高阶特征**: 支持一阶和二阶中心化特征提取
- **增强评估**: 包含猜测熵分析和t-SNE可视化
- **ASCAD数据集支持**: 专门针对ASCAD Fixed-Key数据集优化

## 文件结构

```
├── mobile_mamba.py              # MobileMamba模型架构
├── ascad_loader_fixed.py        # ASCAD数据集加载器
├── trace_dataset_highorder.py   # 高阶特征数据集类
├── train.py                     # 训练函数
├── evaluate.py                  # 评估函数
├── enhanced_evaluate.py         # 增强评估功能
├── run_ascad_mamba.py           # 主训练脚本
└── run_analysis_single_channel.py  # 单通道分析脚本
```

## 模型架构

MobileMambaSCA模型包含：

1. **多尺度卷积干**(Multi-scale Conv Stem): 使用不同内核大小提取多尺度特征
2. **Mamba块堆叠**: 多个RealMambaBlock进行序列建模
3. **全局平均池化**: 降维处理
4. **分类头**: 最终的256类分类器

## 使用方法

### 训练模型

```bash
python run_ascad_mamba.py
```

### 运行增强分析

```bash
python run_analysis_single_channel.py
```

## 依赖项

- PyTorch
- NumPy  
- scikit-learn
- matplotlib
- seaborn
- h5py
- pandas
- tqdm

## 评估指标

- **准确率**: 模型分类准确性
- **猜测熵**: 密钥恢复效果评估
- **Per-class分析**: 每个S-box输出的分类性能
- **t-SNE可视化**: 特征空间可视化

## 数据集

本项目使用ASCAD Fixed-Key数据集，需要将数据文件放置在 `datasets/raw/` 目录下。

## 许可证

本项目仅用于学术研究目的。
