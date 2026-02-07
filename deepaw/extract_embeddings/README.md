# Atomic Embedding Extraction

这个模块用于从晶体结构中提取原子嵌入（atomic embeddings）。这些嵌入是992维的特征向量，编码了电子云的形状信息（通过球谐函数L=0,1,2,3,4，对应s,p,d,f,g轨道）。

## 功能特点

- ✅ **输入简单**：只需要晶体结构（ASE Atoms对象、CIF、POSCAR等），不需要电荷密度网格
- ✅ **物理意义明确**：提取的嵌入编码了环境自适应的电子云形状信息
- ✅ **高维特征**：992维嵌入包含丰富的化学环境信息
- ✅ **可迁移**：可用于下游任务（能量预测、性质预测、力场等）
- ✅ **批量处理**：支持批量提取多个结构
- ✅ **GPU加速**：自动使用GPU加速（如果可用）

## 物理意义

提取的原子嵌入具有明确的物理意义：

| 特征维度 | 物理意义 | 对应轨道 |
|---------|---------|---------|
| L=0分量 | 球对称电荷分布 | s轨道 |
| L=1分量 | 偶极矩、极化方向 | p轨道 |
| L=2分量 | 四极矩、键的方向性 | d轨道 |
| L=3分量 | 八极矩 | f轨道 |
| L=4分量 | 十六极矩 | g轨道 |

这些嵌入是**环境自适应的**：
- 通过3层消息传递（message passing），每个原子的特征会受到其邻近原子（4.0Å cutoff内）的影响
- 编码了原子在特定化学环境下的"有效电子结构"
- 类似于PAW方法中的赝势描述

## 快速开始

### 1. Python API 使用

```python
from deepaw.extract_embeddings import AtomicEmbeddingExtractor
from ase.io import read

# 初始化提取器
extractor = AtomicEmbeddingExtractor()

# 从文件读取结构
atoms = read('structure.cif')

# 提取嵌入
embeddings = extractor.extract(atoms)
print(f"Embeddings shape: {embeddings.shape}")  # (n_atoms, 992)

# 查看统计信息
stats = extractor.get_embedding_statistics(embeddings)
print(f"Statistics: {stats}")

# 保存到文件
extractor.save_embeddings(embeddings, atoms, 'embeddings.npz')
```

### 2. 命令行使用

```bash
# 单个文件
python deepaw/extract_embeddings/extract_script.py structure.cif --output embeddings.npz

# 批量处理
python deepaw/extract_embeddings/extract_script.py *.cif --output_dir ./embeddings/

# 指定输出格式
python deepaw/extract_embeddings/extract_script.py structure.cif --output embeddings.npy --format npy

# 详细输出
python deepaw/extract_embeddings/extract_script.py structure.cif --output embeddings.npz --verbose
```

## 详细使用说明

### 初始化提取器

```python
from deepaw.extract_embeddings import AtomicEmbeddingExtractor

# 使用默认配置
extractor = AtomicEmbeddingExtractor()

# 指定checkpoint路径
extractor = AtomicEmbeddingExtractor(
    checkpoint_path='/path/to/checkpoint.pth'
)

# 指定设备
extractor = AtomicEmbeddingExtractor(device='cuda')

# 自定义模型参数
extractor = AtomicEmbeddingExtractor(
    cutoff=5.0,  # 增大cutoff半径
    lmax=3       # 减少到L=3
)
```

### 提取嵌入的不同方式

#### 从ASE Atoms对象提取

```python
from ase.build import bulk

# 创建晶体结构
atoms = bulk('Si', 'diamond', a=5.43)

# 提取嵌入
embeddings = extractor.extract(atoms)
print(f"Si晶体: {len(atoms)}个原子, 嵌入形状: {embeddings.shape}")
```

#### 从文件提取

```python
# 支持多种格式：CIF, POSCAR, xyz, pdb等
embeddings = extractor.extract_from_file('structure.cif')
embeddings = extractor.extract_from_file('POSCAR')
embeddings = extractor.extract_from_file('structure.xyz')
```

#### 批量提取

```python
from ase.io import read
from glob import glob

# 读取多个结构
structure_files = glob('structures/*.cif')
atoms_list = [read(f) for f in structure_files]

# 批量提取（带进度条）
embeddings_list = extractor.extract_batch(atoms_list, show_progress=True)

# 处理结果
for i, embeddings in enumerate(embeddings_list):
    print(f"结构 {i}: {embeddings.shape}")
```

### 保存和加载嵌入

#### NPZ格式（推荐）

```python
# 保存（包含元数据）
extractor.save_embeddings(embeddings, atoms, 'embeddings.npz', format='npz')

# 加载
import numpy as np
data = np.load('embeddings.npz')
embeddings = data['embeddings']
atomic_numbers = data['atomic_numbers']
positions = data['positions']
cell = data['cell']
```

#### NPY格式（仅嵌入）

```python
# 保存
extractor.save_embeddings(embeddings, atoms, 'embeddings.npy', format='npy')

# 加载
embeddings = np.load('embeddings.npy')
```

#### JSON格式（跨平台）

```python
# 保存
extractor.save_embeddings(embeddings, atoms, 'embeddings.json', format='json')

# 加载
import json
with open('embeddings.json', 'r') as f:
    data = json.load(f)
    embeddings = np.array(data['embeddings'])
```

## 应用场景

### 1. 迁移学习 - 能量预测

```python
import torch
import torch.nn as nn

# 提取嵌入作为特征
embeddings = extractor.extract(atoms)  # (n_atoms, 992)

# 构建下游模型
class EnergyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(992, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, embeddings):
        # 对所有原子的嵌入求和（或平均）
        structure_embedding = embeddings.mean(dim=0)
        energy = self.mlp(structure_embedding)
        return energy

# 训练能量预测模型
model = EnergyPredictor()
# ... 训练代码 ...
```

### 2. 相似性分析

```python
from sklearn.metrics.pairwise import cosine_similarity

# 提取多个结构的嵌入
embeddings1 = extractor.extract(atoms1)
embeddings2 = extractor.extract(atoms2)

# 计算原子级别的相似性
# 比较第一个原子
similarity = cosine_similarity(
    embeddings1[0:1],  # 结构1的第一个原子
    embeddings2[0:1]   # 结构2的第一个原子
)
print(f"原子相似度: {similarity[0, 0]:.4f}")

# 计算结构级别的相似性
struct_emb1 = embeddings1.mean(axis=0, keepdims=True)
struct_emb2 = embeddings2.mean(axis=0, keepdims=True)
struct_similarity = cosine_similarity(struct_emb1, struct_emb2)
print(f"结构相似度: {struct_similarity[0, 0]:.4f}")
```

### 3. 降维可视化

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 提取多个结构的嵌入
all_embeddings = []
labels = []
for i, atoms in enumerate(atoms_list):
    emb = extractor.extract(atoms)
    all_embeddings.append(emb)
    labels.extend([f'Structure_{i}'] * len(atoms))

# 合并所有嵌入
all_embeddings = np.vstack(all_embeddings)

# PCA降维到2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(all_embeddings)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
           c=range(len(embeddings_2d)), cmap='viridis', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Atomic Embeddings in 2D')
plt.colorbar(label='Atom Index')
plt.savefig('embeddings_pca.png', dpi=300)
```

### 4. 聚类分析

```python
from sklearn.cluster import KMeans

# 提取嵌入
embeddings = extractor.extract(atoms)

# K-means聚类
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# 分析每个簇
for i in range(n_clusters):
    cluster_atoms = np.where(clusters == i)[0]
    cluster_elements = atoms.get_chemical_symbols()[cluster_atoms]
    print(f"簇 {i}: {len(cluster_atoms)} 个原子")
    print(f"  元素: {set(cluster_elements)}")
```

## 嵌入分析

### 查看统计信息

```python
# 获取嵌入统计
stats = extractor.get_embedding_statistics(embeddings)
print(f"均值: {stats['mean']:.4f}")
print(f"标准差: {stats['std']:.4f}")
print(f"范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
print(f"向量范数均值: {stats['norm_mean']:.4f}")
```

### 分析不同轨道分量

```python
# 992维嵌入包含不同L值的分量
# 可以分析特定轨道的贡献

# 假设前面的维度对应低L值（这需要根据实际模型结构确定）
# 这里只是示例
s_orbital_features = embeddings[:, :167]  # L=0 (s轨道)
p_orbital_features = embeddings[:, 167:335]  # L=1 (p轨道)

print(f"s轨道特征范数: {np.linalg.norm(s_orbital_features, axis=1).mean():.4f}")
print(f"p轨道特征范数: {np.linalg.norm(p_orbital_features, axis=1).mean():.4f}")
```

## 注意事项

### 1. 内存使用

- 每个原子的嵌入是992维浮点数（~4KB）
- 对于大型结构（>10000原子），注意内存使用
- 建议批量处理时使用较小的batch size

### 2. 计算时间

- GPU加速可显著提升速度（~10-100倍）
- 单个小结构（<100原子）：~0.1秒（GPU）
- 大型结构（>1000原子）：~1-5秒（GPU）

### 3. 周期性边界条件

- 模型自动处理周期性边界条件（PBC）
- 确保输入结构的PBC设置正确
- 对于分子（非周期性），设置`atoms.set_pbc(False)`

### 4. Cutoff半径

- 默认cutoff=4.0Å
- 增大cutoff可以捕获更远的相互作用，但计算量增加
- 减小cutoff可以加速计算，但可能损失信息

## 常见问题

### Q1: 提取的嵌入可以用于其他任务吗？

**A:** 可以！这些嵌入编码了丰富的化学环境信息，可以用于：
- 能量预测
- 力场构建
- 性质预测（带隙、形成能等）
- 结构相似性分析
- 材料筛选

### Q2: 嵌入的物理意义是什么？

**A:** 嵌入通过球谐函数（L=0,1,2,3,4）编码了电子云的形状信息：
- L=0: 球对称分布（s轨道）
- L=1: 方向性（p轨道）
- L=2: 键的方向性（d轨道）
- L=3,4: 更复杂的电子云形状

这些特征是**环境自适应的**，会根据原子的局部化学环境变化。

### Q3: 为什么不同原子的嵌入维度相同？

**A:** 所有原子都使用相同的992维表示，但：
- 不同元素的嵌入会有不同的模式
- 相同元素在不同环境下的嵌入也会不同
- 这种统一的表示便于下游任务处理

### Q4: 如何处理大型结构？

**A:** 对于超大结构（>10000原子）：
1. 使用GPU加速
2. 考虑分块处理
3. 只提取感兴趣区域的原子嵌入

```python
# 只提取部分原子
indices = [0, 1, 2, 10, 11, 12]  # 感兴趣的原子索引
sub_atoms = atoms[indices]
embeddings = extractor.extract(sub_atoms)
```

### Q5: 嵌入是否旋转不变？

**A:** 模型使用E3等变架构，嵌入是**旋转等变的**（不是不变的）：
- 标量分量（L=0）是旋转不变的
- 矢量和张量分量（L>0）会随旋转而变换
- 这保留了方向信息，对于预测各向异性性质很重要

## 技术细节

### 模型架构

- **输入**: 原子种类（one-hot编码）+ 原子位置
- **消息传递**: 3层E3等变卷积
- **球谐函数**: L=0,1,2,3,4（s,p,d,f,g轨道）
- **输出**: 992维原子嵌入

### 配置参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `cutoff` | 4.0 | 邻居搜索半径（Å） |
| `num_neighbors` | 20 | 最大邻居数 |
| `lmax` | 4 | 最大角动量 |
| `mul` | 500 | irreps倍数 |
| `num_basis` | 10 | 径向基函数数量 |

## 引用

如果使用此模块，请引用DeePAW论文：

```bibtex
@article{deepaw2024,
  title={DeePAW: Deep Learning for PAW Charge Density Prediction},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024}
}
```

## 联系方式

如有问题或建议，请通过GitHub Issues联系。



