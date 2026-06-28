# Surya 数据流全解：从 NetCDF 文件到最终预测

## 前置：你需要理解的一个核心概念——张量（Tensor）

**张量就是多维数组**。你不需要被这个词吓到。

```
一个数字:            3.14                    → 0维张量 (标量)
一排数字:            [1, 2, 3, 4, 5]         → 1维张量 (向量)  
一张表格:            [[1,2],[3,4],[5,6]]     → 2维张量 (矩阵)
一叠表格:            [ 表格1, 表格2, 表格3 ]  → 3维张量
更多维度叠在一起:                                → 4维、5维...
```

在 PyTorch 里，我们用 `shape` 来描述张量有几个维度，每个维度多大：

```python
shape = (3, 256, 256) 
#        │   │    │
#        │   │    └── 第2维大小：256 列（宽）
#        │   └─────── 第1维大小：256 行（高）
#        └─────────── 第0维大小：3 个通道（比如 RGB）
```

**关键**：Surya 代码里到处都是 `(B, C, T, H, W)` 这样的标记，它们只是维度的缩写：

| 缩写 | 含义 | 这个例子中的实际值 |
|------|------|-------------------|
| **B** | Batch，一次处理几张图 | 比如 1 或 8 |
| **C** | Channels，几个观测波段 | Surya 用 13 个通道 |
| **T** | Time，几个时间步 | 比如 2 |
| **H** | Height，图像高度 | 4096 像素 |
| **W** | Width，图像宽度 | 4096 像素 |
| **L** | Length，token 序列长度 | 65536（后面会解释） |
| **D** | Dimension，嵌入维度 | 768 |

---

## 第一步：数据从文件读入

### 文件里有什么？

Surya 的数据是 NetCDF 文件（`.nc`）。一个文件包含**一个时刻**的太阳观测数据。

```
文件: 20140107_1200.nc
├── aia094  → 4096×4096 的数组（94埃波段图像）
├── aia131  → 4096×4096 的数组（131埃波段图像）  
├── aia171  → 4096×4096 的数组（171埃波段图像）
├── ...     → 更多波段...
├── aia304  → 4096×4096 的数组
└── hmi     → 4096×4096 的数组（磁图）
```

每个波段是对太阳不同"颜色"的观测，就像你用不同滤镜看太阳。

### 数据集怎么组织这些文件？

代码在 `surya/datasets/helio.py` 的 `HelioNetCDFDataset` 类中。

**核心逻辑**：模型不是只看"一张图"，而是看"一小段时间序列"来预测未来。

```
                    ┌─────────────────────────────────────────┐
时间轴               │  过去（输入）    │      未来（目标）      │
                    │  T=2 个时间步    │   L=1 个时间步        │
                    │                 │                       │
时刻:  11:48  12:00 │  11:48   12:00  │       13:00           │
       ───── ───── │  ──────  ────── │       ──────          │
文件:  .nc    .nc  │  输入1   输入2   │       目标            │
                    └─────────────────────────────────────────┘
```

`__getitem__` 方法（`helio.py:326`）做这些事：

```python
# 伪代码，展示整个过程
def __getitem__(self, idx):
    # 1. 确定需要哪些时间点的数据
    #    比如需要 [11:48, 12:00, 13:00] 三个时间点
    
    # 2. 逐个读取 NetCDF 文件
    for timestep in required_timesteps:
        data = load_nc_data(filepath)  # → shape (C=13, H=4096, W=4096)
        data = signum_log_transform(data)  # 数值归一化
        sequence_data.append(data)
    
    # 3. 拆分为输入和目标
    inputs = sequence_data[:2]   # 前两个 = 输入
    targets = sequence_data[2:]  # 最后一个 = 目标
    
    # 4. 堆叠为张量
    stacked_inputs = np.stack(inputs, axis=1)  
    # → shape (C=13, T=2, H=4096, W=4096)
    
    stacked_targets = np.stack(targets, axis=1)
    # → shape (C=13, T=1, H=4096, W=4096)
```

### 数据集返回的 batch 是什么？

数据集返回一个字典（`helio.py:465-470`）：

```python
batch = {
    "ts":                # shape (13, 2, 4096, 4096) — 输入图像序列
    "time_delta_input":  # shape (2,)             — 每个输入的时间差（小时）
    "forecast":          # shape (13, 1, 4096, 4096) — 目标图像（用于算损失）
    "lead_time_delta":   # shape (1,)             — 预测的时间差
}
```

**注意**：数据集返回的是单个样本，还没有 B（batch）维度。B 维度是由 DataLoader 的 `collate_fn` 函数加上去的。

### DataLoader 做了什么？

`DataLoader` 一次取多个样本（比如 8 个），用 `custom_collate_fn`（`surya/utils/data.py:11`）把它们**沿第 0 维堆叠**：

```python
# 8 个样本，每个样本的 ts 是 (13, 2, 4096, 4096)
# collate 后 → (8, 13, 2, 4096, 4096)
#               B=8  C=13  T=2  H=4096  W=4096
```

**这就是 B（Batch）维度的来源**。B=8 表示一次同时处理 8 个样本。

---

## 第二步：模型收到 batch 后发生了什么

模型入口是 `HelioSpectFormer.forward()`（`helio_spectformer.py:242`）。

为了好理解，我们假设 **B=1**（一次只处理一个样本）。

### 整个流程的"地铁线路图"

```
batch = {"ts": (1,13,2,4096,4096), ...}
                │
                ▼
┌──────────────────────────────────────────────┐
│                                               │
│  ① Embedding (LinearEmbedding)                │
│     把像素变成"特征 token"                      │
│     (1,13,2,4096,4096) → (1, 65536, 768)      │
│     B,C,T,H,W            B, L,     D          │
│                                               │
│                     │                         │
│                     ▼                         │
│  ② SpectFormer (Backbone / 骨干网络)           │
│     在 token 之间传递信息、提取特征              │
│     (1, 65536, 768) → (1, 65536, 768)          │
│     B, L,     D        B, L,     D            │
│                                               │
│                     │                         │
│                     ▼                         │
│  ③ Decoder (LinearDecoder / 解码器)            │
│     把 token 变回像素                           │
│     (1, 65536, 768) → (1, 13, 4096, 4096)     │
│     B, L,     D        B, C, H,     W         │
│                                               │
└──────────────────────────────────────────────┘
                │
                ▼
        预测图像 (1,13,4096,4096)
```

**三句话总结**：
- **Embedding**：像素 → token（图像压缩）
- **SpectFormer**：token 之间交流信息（特征提取）
- **Decoder**：token → 像素（图像还原）

下面逐个详解。

---

## 第三步：Embedding 详解——像素如何变成 Token

代码在 `surya/models/embedding.py`。

### 3.1 为什么要做 Embedding？

直接看数字：一张 4096×4096 的图像有 **1600 万个像素**。要把这么多像素送进神经网络做计算，计算量太大了。

**解决思路**：把图像切成小块（patch），每个小块做一次压缩。就像你把一张大照片缩略成小图，保留了关键信息但数据量大幅减少。

### 3.2 PatchEmbed3D：切块 + 压缩

`embedding.py:23-52`

```python
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=16, in_chans=13, time_dim=2, embed_dim=768):
        # Conv2d 就是"切块 + 压缩"的工具
        self.proj = nn.Conv2d(
            in_channels=26,    # 13 通道 × 2 时间步 = 26
            out_channels=768,  # 压缩后每个 patch 用 768 个数字表示
            kernel_size=16,    # 每个 patch 是 16×16 的方块
            stride=16,         # 每次移动 16 像素（patch 不重叠）
        )
```

**用具体数字演示**：

```
输入: x = (1, 13, 2, 4096, 4096)
               │   │
               │   └── 2 个时间步
               └────── 13 个通道

第一步: flatten
x = x.flatten(1, 2)  
# → (1, 26, 4096, 4096)   ← 把 C 和 T 维度合并为一个维度
#     B   │    │     │
#         └────┴─────┴────── 26 = 13通道 × 2时间步，看作"26层的图像堆"

第二步: Conv2d 切块
x = self.proj(x)
# Conv2d 做的事：
#   每次取一个 16×16 的小方块（patch）
#   压缩为一个 768 维的向量
#   然后移动到下一个方块（步长=16，不重叠）
#
# 4096 ÷ 16 = 256，所以横竖各 256 个 patch
# 
# → (1, 768, 256, 256)
#     B   D    │    │
#              └────┴──── 256×256 = 65536 个 patch

第三步: rearrange，把空间维度展平
x = rearrange(x, "B D H W -> B (H W) D")
# → (1, 65536, 768)
#     B   L      D
#
# L=65536: 总共有 65536 个 token（patch）
# D=768:   每个 token 用 768 个数字表示
```

**理解 token**：一个 token 就是"一小块图像的压缩表示"。65536 个 token 覆盖了整个太阳图像（每个 token 代表 16×16 像素的方块）。

### 3.3 位置编码（Position Encoding）

`embedding.py:81-113`

**问题**：token 序列（65536 个 token）现在失去了空间位置信息。网络不知道哪个 token 在图像的什么位置。

**解决**：给每个 token 加上一个"位置签名"。

```python
def _generate_position_encoding(self, ...):
    # 对每个位置 (x, y)，生成一个唯一的 768 维向量
    # 使用不同频率的正弦/余弦波的组合
    
    # 比如：第 0 行第 0 列的 token → 对应位置编码的第 0 个向量
    #       第 127 行第 200 列的 token → 对应位置编码的第 (127*256+200) 个向量
    
    self.register_buffer("pos_embed", fourier_signal)  # shape (1, 65536, 768)
```

### 3.4 LinearEmbedding 完整 forward

`embedding.py:115-127`

```python
def forward(self, x, dt):
    # x: (1, 13, 2, 4096, 4096)
    
    x = self.patch_embed(x)      # → (1, 65536, 768)  [切块+压缩]
    x = x + self.pos_embed        # → (1, 65536, 768)  [加上位置信息]
    x = self.pos_drop(x)          # → (1, 65536, 768)  [随机丢弃一些（防过拟合）]
    
    return x
```

**Embedding 阶段的完整 shape 变化**：

```
(1, 13, 2, 4096, 4096)          ← 输入：1个样本，13通道，2时间步，4096×4096图像
        │
        ▼ flatten(1,2)
(1, 26, 4096, 4096)             ← 通道和时间合并为26层
        │
        ▼ Conv2d(kernel=16, stride=16)
(1, 768, 256, 256)              ← 切成65536个patch，每个压缩为768维
        │
        ▼ rearrange
(1, 65536, 768)                 ← 展开为token序列
        │
        ▼ + pos_embed
(1, 65536, 768)                 ← 加上位置编码（形状不变）
```

---

## 第四步：SpectFormer（Backbone）详解——Token 之间如何交流

代码在 `surya/models/spectformer.py`。

### 4.1 什么是 Backbone？

**Backbone = 骨干网络 = 模型的核心**。

如果说 Embedding 是把图像"翻译"成 token，Decoder 是把 token "翻译"回图像，那 Backbone 就是在 token 之间**传递信息、提取特征、建立理解**。

类比：Embedding 是把一本书翻译成一种内部语言，Backbone 是"阅读理解"，Decoder 是用内部语言写出一本新书。

### 4.2 SpectFormer 的 10 层结构

`spectformer.py:183-296`

```
SpectFormer 包含 10 层（depth=10），分两种类型：

层 0-1:  BlockSpectralGating × 2  频谱门控层（在频率域工作）
层 2-9:  BlockAttention × 8       注意力层（在空间域工作）

每层的输入和输出 shape 完全一样：(1, 65536, 768)
```

**为什么输入输出 shape 一样？**
这就像一个"打磨"过程——数据进去，被加工了一下，出来，但大小不变。10 层就是 10 次打磨，每次都让特征变得更好。

### 4.3 BlockSpectralGating：频域处理

`spectformer.py:80-107`

这一层用**傅里叶变换（FFT）**把 token 从"空间域"换到"频率域"，在频率域里做处理，再换回来。

```
空间域 →  FFT  → 频率域 → 乘上可学习的权重 → iFFT → 空间域
(看图)           (看频谱)   (增强/抑制某些频率)       (看图)
```

**直觉理解**：
- 空间域：你看一张照片，看到的是"这里有个活动区，那里有个冕洞"
- 频率域：你把照片分解为"粗纹理 + 细纹理 + 超细纹理"的叠加
- FFT 层的作用：学习哪些频率的纹理是重要的，增强它们，抑制噪声

shape 变化：

```python
def forward(self, x):
    # x: (1, 65536, 768)
    #        │      │
    #        │      └── 每个 token 的 768 维特征
    #        └───────── 65536 个 token
    
    B, N, C = x.shape  # N=65536, C=768
    
    # 把 token 序列还原为 2D 空间布局
    x = x.view(B, 256, 256, C)  # 256=√65536
    # → (1, 256, 256, 768)
    
    # 傅里叶变换到频率域
    x = torch.fft.rfft2(x, dim=(1, 2))
    # → (1, 256, 129, 768)  ← rfft2 会把最后一维减半（对称性）
    
    # 乘上可学习的复数权重（这一层的核心）
    x = x * self.complex_weight  # (256, 129, 768) 的可学习参数
    # → (1, 256, 129, 768)
    
    # 逆傅里叶变换回空间域
    x = torch.fft.irfft2(x, s=(256, 256), dim=(1, 2))
    # → (1, 256, 256, 768)
    
    # 展平回 token 序列
    x = x.reshape(B, N, C)
    # → (1, 65536, 768)
    
    # 再经过 MLP（全连接层）进一步处理
    x = x + self.mlp(...)
    # → (1, 65536, 768)
    
    return x
```

### 4.4 BlockAttention：注意力机制

`spectformer.py:110-180`

这是 Surya 最核心的模块。它使用**长-短注意力（Long-Short Attention）**。

**什么是注意力？**

简单说：每个 token 会"看"其他相关的 token，然后说"我和它们的关系有多密切？我该如何根据它们来更新自己？"

**比如**：一个代表太阳黑子的 token，会特别关注周围同样代表黑子的 token，而不太关注远处的宁静太阳区域的 token。

**长-短注意力**把注意力分成两种：

| 类型 | 看多少 token | 目的 |
|------|-------------|------|
| **短注意力**（窗口） | 只看周围一个小窗口内的 token | 捕捉局部细节（黑子边界、精细纹理） |
| **长注意力**（动态投影） | 看全局所有 token（但压缩后） | 捕捉全局结构（整个活动区的位置、太阳圆面轮廓） |

代码实现在 `transformer_ls.py`。

```python
class BlockAttention(nn.Module):
    def forward(self, x, noise=None):
        # x: (1, 65536, 768)
        
        # 1. LayerNorm（归一化，让数值稳定）
        x_norm = self.norm1(x)
        
        # 2. 长-短注意力（核心）
        #    内部做两件事：
        #    a. 窗口注意力：每个 token 只看周围的 2×2 窗口内的 token
        #    b. 动态投影：把所有 token 压缩到 2 维（dp_rank=2），
        #       然后用这 2 维做全局交互
        x_attn = self.attn(x_norm)  # → (1, 65536, 768)
        
        # 3. 残差连接（把原始输入加回来，防止信息丢失）
        x = x + x_attn
        
        # 4. MLP（两层全连接，进一步处理每个 token）
        x = x + self.mlp(self.norm2(x))
        
        return x  
        # → (1, 65536, 768)  shape 不变！
```

### 4.5 SpectFormer 的完整 forward

`spectformer.py:273-296`

```python
def forward(self, tokens):
    # tokens: (1, 65536, 768)
    
    for i, blk in enumerate(all_10_blocks):
        # 每个 block 处理一次，输入输出 shape 完全相同
        tokens = blk(tokens)
        # shape 始终是 (1, 65536, 768)
    
    return tokens
    # → (1, 65536, 768)
```

**Backbone 阶段总结**：输入 65536 个 token，每个 768 维；经过 10 层处理（2 层频域 + 8 层注意力）；输出仍然是 65536 个 token，每个 768 维。但内容变了——token 现在包含了更丰富的特征信息，token 之间也完成了信息交流。

---

## 第五步：Decoder 详解——Token 如何变回图像

代码在 `surya/models/embedding.py` 的 `LinearDecoder` 类（`embedding.py:130-172`）。

### 5.1 为什么要 Decoder？

Backbone 输出的还是 token（压缩表示），我们需要把它变回真正的图像。Decoder 做的是 Embedding 的逆操作。

### 5.2 LinearDecoder

```python
class LinearDecoder(nn.Module):
    def __init__(self, patch_size=16, out_chans=13, embed_dim=768):
        # Conv2d(kernel=1) 把 768 维放大到 16²×13 = 3328 维
        # 3328 = 256(16×16个像素) × 13(个通道)
        self.unembed = nn.Sequential(
            nn.Conv2d(768, 3328, kernel_size=1),  # 维度扩展
            nn.PixelShuffle(16),                   # 重排为空间像素
        )
```

**用具体数字演示**：

```python
def forward(self, x):
    # x: (1, 65536, 768)
    #     B   L     D
    
    # 第一步：把 token 序列还原为 2D 空间布局
    H_token = W_token = 256  # √65536 = 256
    x = rearrange(x, "B (H W) D -> B D H W", H=256, W=256)
    # → (1, 768, 256, 256)
    #     B   D    │    │
    #              └────┴── token 空间网格，256×256
    
    # 第二步：卷积 → 像素（Conv2d + PixelShuffle）
    x = self.unembed(x)
    # Conv2d(768 → 3328, kernel=1):
    #   对每个 token 位置，把 768 维向量扩展为 3328 维
    #   → (1, 3328, 256, 256)
    #
    # PixelShuffle(16):
    #   把这 3328 维重新排列为 16×16×13 的像素块
    #   3328 = 16×16 × 13
    #   → (1, 13, 4096, 4096)
    #
    # 原理：
    #   原来的每个 token 位置（256×256 网格）→ 扩展为一个 16×16 的像素块
    #   256 × 16 = 4096 像素
    #   每个像素块有 13 个通道
    #   → 最终：4096×4096 的 13 通道图像
    
    return x
    # → (1, 13, 4096, 4096)
```

**Decoder 阶段总结**：65536 个 token → 每个 token 对应 16×16 像素 → 拼接成 4096×4096 的图像 → 输出 13 个通道的预测图。

---

## 第六步：完整 forward 流程一览

用 `HelioSpectFormer.forward()`（`helio_spectformer.py:242-318`）串起来：

```
                      Surya 完整数据流
═══════════════════════════════════════════════════════════════

INPUT: batch = {
    "ts":               (1, 13, 2, 4096, 4096)  ← 输入图像序列
    "time_delta_input": (1, 2)                  ← 时间差
}

    │
    ▼
┌─ EMBEDDING ─────────────────────────────────────────────────┐
│                                                              │
│  LinearEmbedding.forward(x, dt)                             │
│                                                              │
│  x = (1, 13, 2, 4096, 4096)                                │
│       │                                                      │
│       ├─ flatten(C,T) → (1, 26, 4096, 4096)                │
│       │                                                      │
│       ├─ Conv2d(26→768, kernel=16, stride=16)              │
│       │  → (1, 768, 256, 256)                               │
│       │                                                      │
│       ├─ rearrange → (1, 65536, 768)                        │
│       │                                                      │
│       └─ + pos_embed → (1, 65536, 768)                      │
│                                                              │
│  OUTPUT: tokens = (1, 65536, 768)                            │
│                    B  L=65536 D=768                          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ BACKBONE (SpectFormer) ────────────────────────────────────┐
│                                                              │
│  SpectFormer.forward(tokens)                                │
│                                                              │
│  for each of 10 blocks:                                     │
│    ├─ BlockSpectralGating ×2: FFT → 频域滤波 → iFFT       │
│    └─ BlockAttention ×8:     长-短注意力 + MLP              │
│                                                              │
│  每层输入输出 shape 不变：                                     │
│  (1, 65536, 768) → ... → (1, 65536, 768)                   │
│                                                              │
│  OUTPUT: tokens = (1, 65536, 768)                            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ DECODER ───────────────────────────────────────────────────┐
│                                                              │
│  LinearDecoder.forward(tokens)                              │
│                                                              │
│  tokens = (1, 65536, 768)                                   │
│       │                                                      │
│       ├─ rearrange → (1, 768, 256, 256)                     │
│       │                                                      │
│       └─ Conv2d(768→3328) + PixelShuffle(16)               │
│          → (1, 13, 4096, 4096)                              │
│                                                              │
│  OUTPUT: forecast_hat = (1, 13, 4096, 4096)                  │
└──────────────────────────────────────────────────────────────┘

        预测图像: (1, 13, 4096, 4096)
        真实图像: (1, 13, 4096, 4096)  ← batch["forecast"][:, :, 0, :, :]
                    │
                    ▼
         Loss = MSE(预测, 真实)
═══════════════════════════════════════════════════════════════
```

---

## 第七步：Loss 如何计算

在训练代码中（如 `downstream_examples/ar_segmentation/finetune.py:536-537`）：

```python
# 1. 模型前向传播，得到预测
outputs = model(curr_batch)          # → (1, 13, 4096, 4096)

# 2. 从 batch 中取出目标（真实值）
target = curr_batch["forecast"]      # → (1, 13, 1, 4096, 4096)
target = target[:, :, 0, :, :]      # → (1, 13, 4096, 4096)  去掉多余时间维

# 3. 计算损失
loss = F.mse_loss(outputs, target)   # 均方误差 = (预测-真实)² 的平均值
```

`MSE`（Mean Squared Error）就是把预测的每个像素和真实值的每个像素做差，平方，然后全部平均。**数值越小，预测越准。**

---

## 为什么输入变成 (B, L, D)？

```
(B, C, T, H, W)  →  Embedding  →  (B, L, D)

B: batch 大小 (不变)
C: 13 个通道 → 被折叠进 token
T: 2 个时间步 → 被折叠进 token  
H: 4096 高
W: 4096 宽
   这些被压缩：
   ├─ H/patch_size = 4096/16 = 256
   └─ W/patch_size = 4096/16 = 256
   L = 256 × 256 = 65536 个 token

D: 768 = embed_dim，每个 token 的特征维度
```

**一句话**：`(B, L, D)` 就是把图像切成 L 个小块，每个小块用 D 个数字描述。B 个样本一起处理。

---

## 几个术语的一对一解释

| 术语 | 对应代码 | 一句话解释 |
|------|----------|-----------|
| **Embedding** | `LinearEmbedding` / `PerceiverChannelEmbedding` | 把 4096×4096 像素的大图像切成 65536 个小块，每个压缩为 768 个数字 |
| **Backbone** | `SpectFormer`（含 `BlockSpectralGating` + `BlockAttention`） | 10 层网络，让 token 之间交流信息，10 次"打磨"特征 |
| **Decoder** | `LinearDecoder` / `PerceiverDecoder` | 把 65536 个 token 重新组装成 4096×4096 的图像 |
| **Token** | `tokens = (B, L, D)` | 一个小块图像的压缩表示，L 个小块，每个 D 维 |
| **adaLN** | `BlockAttention.adaLN_modulation` | 自适应调制——用外部条件信号去调节 Attention 层的行为 |
| **forward** | 模型的 `forward()` 方法 | "输入→处理→输出"的完整流水线 |

---

## 用一句话概括 Surya 在做什么

> **Surya 接收过去两帧（12 分钟间隔）的 13 通道太阳图像，经过"切块压缩→10 层特征提取→还原放大"，输出来来时刻的 13 通道预测图像。**

```
输入: 过去两帧太阳图像          输出: 未来一帧太阳图像
┌──────────┬──────────┐       ┌──────────────────┐
│ 11:48    │ 12:00    │  →    │ 13:00 预测       │
│ 13通道   │ 13通道   │       │ 13通道 4096×4096 │
│ 4096²    │ 4096²    │       │                  │
└──────────┴──────────┘       └──────────────────┘
```

每个中间步骤（Embedding → Backbone → Decoder）都是为了完成这个"图像→图像"的映射，同时学习太阳物理的深层规律。
