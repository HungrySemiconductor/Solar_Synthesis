# Surya 条件注入分析报告

> 分析目标：在 Surya 基础模型中注入轨道参数（x, y, z, vx, vy, vz）作为条件信号，使模型能够直接学习不同观测位置下的太阳 EUV 图像生成——同时包含距离变化导致的几何缩放和方位变化导致的纹理变化。

---

## 一、代码结构总览

Surya 的数据流如下：

```
batch = {ts: (B,C,T,H,W), time_delta_input: (B,T)}
        │
        ▼
┌─────────────────────────────────┐
│  Embedding                      │  embedding.py:55-127
│  LinearEmbedding / Perceiver    │  (B,C,T,H,W) → (B, L, D)
│  PatchEmbed3D: Conv2d over      │
│  flattened (C×T, H, W)          │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  SpectFormer (Backbone)         │  spectformer.py:183-296
│  10 layers total:               │
│  ├─ BlockSpectralGating ×2      │  (FFT frequency filtering)
│  └─ BlockAttention ×8           │  (Long-Short Attention)
│      └─ AttentionLS             │  transformer_ls.py
│      └─ adaLN (if ensemble)     │  ← 已存在条件注入基础设施!
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Decoder                        │  embedding.py:130-483
│  LinearDecoder / PerceiverDecoder│ (B,L,D) → (B,C,H,W)
└─────────────────────────────────┘
```

### 关键源文件

| 文件 | 作用 |
|------|------|
| `surya/models/helio_spectformer.py` | 顶层模型入口，定义 forward 流程 |
| `surya/models/spectformer.py` | Backbone：Spectral Gating + Attention blocks |
| `surya/models/transformer_ls.py` | Long-Short Attention 实现（AttentionLS）|
| `surya/models/embedding.py` | 输入 Embedding（Linear/Perceiver）+ Decoder |
| `surya/models/flow.py` | HelioFlowModel：基于 grid_sample 的空间变换 |
| `surya/datasets/helio.py` | 数据集类 HelioNetCDFDataset |
| `surya/utils/config.py` | 配置类（DataConfig, ModelConfig, ExperimentConfig）|
| `downstream_examples/ar_segmentation/` | 下游任务范例（展示 finetune 模式）|

---

## 二、问题 1：如何将轨道参数作为条件输入

### 现状

当前 forward 的输入只有时间和图像（`helio_spectformer.py:242-318`）：

```python
# helio_spectformer.py:252-253
x = batch["ts"]                 # B, C, T, H, W
dt = batch["time_delta_input"]  # B, T
```

模型完全不知道观测位置信息。

### 方案 A：扩展 batch 字典（最小侵入）

在数据集返回的 batch 中增加轨道参数，不影响现有数据流：

```python
# helio.py:367-470, 在 __getitem__ 返回中增加
batch = {
    "ts": stacked_inputs,               # B, C, T, H, W
    "time_delta_input": dt,             # B, T
    "orbit_params": orbit_vec,          # B, 6   ← 新增：x,y,z,vx,vy,vz
}
```

轨道参数需要预先归一化。推荐标准化方案：

```python
# 位置 (km) 除以 1 AU (149,597,870.7 km)，速度 (km/s) 除以 29.8 km/s（地球轨道速度）
position = np.array([x, y, z]) / 149597870.7
velocity = np.array([vx, vy, vz]) / 29.8
orbit_vec = np.concatenate([position, velocity])  # 6D
```

### 方案 B：轨道参数编码器（推荐）

在 `HelioSpectFormer.__init__` 中新增一个轻量级 MLP（设计模式参考 `flow.py:31-35` 的 `flow_generator`）：

```python
# 类似 flow.py:31-35 的 flow_generator 设计模式
self.orbit_encoder = nn.Sequential(
    nn.Linear(6, 128),            # 6D orbit → 128D
    nn.GELU(),
    nn.Linear(128, embed_dim),    # 128D → embed_dim（匹配 token 维度）
)
```

**插入位置**：`helio_spectformer.py:268`，在 embedding 之后、backbone 之前。轨道编码可以与 token 序列拼接或通过 adaLN 注入。

---

## 三、问题 2：如何让模型学习太阳大小变化

### 核心原理

观测距离 $d$ 决定太阳视直径 $\theta \propto 1/d$。这是纯粹的几何缩放变换——对模型而言，需要在生成阶段调整输出图像中太阳圆面的表观大小。

### 方案：距离条件化的 HelioFlowModel

**关键发现**：`HelioFlowModel`（`flow.py`）已经通过 `F.grid_sample` 实现了空间变换。目前它只学习一个固定的 flow field（`flow.py:70`）：

```python
# flow.py:70 — 当前：固定 flow field，与输入图像无关
flow_field = self.grid + self.flow_generator(self.higher_modes)
```

可以扩展为**距离条件化的 flow field**：

```python
# 改进方案：用距离 r = sqrt(x²+y²+z²) 调制缩放因子
r = torch.norm(orbit_params[:, :3], dim=1)  # B,  —— 轨道距离
r_ref = 149597870.7  # 1 AU (km) 作为参考距离
scale = r_ref / r  # B — 缩放因子，距离越远 scale 越小

# 用 scale 调制 grid coordinates
scaled_grid = self.grid * scale.view(-1, 1, 1, 1)  # B, H, W, 2
flow_field = scaled_grid + self.flow_generator(self.higher_modes)
```

这样模型天然理解：距离越远 → grid 收缩 → 太阳圆面变小。

### 进一步：学习视角相关的精细 flow

`flow.py:31-35` 的 `flow_generator` 目前只接收空间坐标（2 或 3 个通道：u, v, 可选 latitude）。可扩展为接收轨道编码：

```python
# 新方案：轨道编码 + 空间坐标 → 主动学习的 flow
self.flow_generator = nn.Sequential(
    nn.Linear(embed_dim + 2, 128),   # orbit_embed + (u, v)
    nn.GELU(),
    nn.Linear(128, 2),               # 输出 flow_x, flow_y
)

# forward 中：
orbit_embed = self.orbit_encoder(orbit_params)  # B, embed_dim
orbit_expanded = orbit_embed[:, None, None, :].expand(B, H, W, embed_dim)
flow_input = torch.cat([self.higher_modes.expand(B, -1, -1, -1), 
                         orbit_expanded], dim=-1)
flow_field = self.grid + self.flow_generator(flow_input)
y_hat = F.grid_sample(x, flow_field, mode="bilinear", 
                       padding_mode="border", align_corners=False)
```

这样 flow field 既编码了几何缩放，也编码了视角相关的形变。

---

## 四、问题 3：如何让模型学习纹理变化

纹理变化来自**观测方位变化**（不同的 $(x,y,z)$ 看到太阳的不同面、不同的边缘临边昏暗效应、不同的活动区域朝向）。这比几何缩放更复杂，需要深度注入到特征提取层。

### 方案 A：adaLN 注入轨道信息到 Attention 层（推荐）

**这是最关键的发现**：`BlockAttention`（`spectformer.py:110-157`）已经有 `adaLN`（自适应层归一化调制）基础设施！

```python
# spectformer.py:149-156 — 已存在但只在 ensemble 模式下用于 noise 注入
if adaLN:
    self.adaLN_modulation = nn.Sequential(
        nn.Linear(dim, dim, bias=True),
        act_layer(),
        nn.Linear(dim, 6 * dim, bias=True),  # 6种调制参数
    )
```

当前 `adaLN` 只在 `ensemble` 模式时启用（`spectformer.py:266`）：

```python
# spectformer.py:266 — 当前逻辑
adaLN=True if ensemble is not None else False,
```

**可以直接改造**：让 orbit embedding 替代 noise 作为 adaLN 的条件输入。

adaLN 在 forward 中产生 6 种调制参数（`spectformer.py:158-169`）：

| 参数 | 作用 | 代码位置 |
|------|------|----------|
| `shift_mha` | Multi-Head Attention 前的特征平移 | `spectformer.py:173` |
| `scale_mha` | Multi-Head Attention 前的特征缩放 | `spectformer.py:173` |
| `gate_mha` | Multi-Head Attention 输出的门控 | `spectformer.py:171-175` |
| `shift_mlp` | MLP 前的特征平移 | `spectformer.py:177` |
| `scale_mlp` | MLP 前的特征缩放 | `spectformer.py:177` |
| `gate_mlp` | MLP 输出的门控 | `spectformer.py:176-178` |

这意味着轨道信息可以**精细控制每一层 Attention 的特征表达**，让模型学会"根据观测角度调整哪些纹理特征被强调或抑制"。

**修改方案**：将 `BlockAttention.forward` 的第二个参数从 `noise` 改为 `condition`：

```python
# spectformer.py:158 — 修改后的 forward 签名
def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # c 现在是 orbit_embedding（B, 1, dim），不再是随机 noise
    if self.adaLN_modulation is not None:
        (shift_mha, scale_mha, gate_mha,
         shift_mlp, scale_mlp, gate_mlp
        ) = self.adaLN_modulation(c).chunk(6, dim=2)
    else:
        shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = 6 * (1.0,)
    # ... 后续不变
```

然后在 `SpectFormer.forward` 中传递 orbit_embed：

```python
# spectformer.py:273 — 修改后
def forward(self, tokens: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
    for i, blk in enumerate(chain(self.blocks_spectral_gating, self.blocks_attention)):
        # Spectral Gating 不接收条件（或可扩展）
        # BlockAttention 接收 orbit_embed
        if isinstance(blk, BlockAttention):
            tokens = blk(tokens, condition)
        else:
            tokens = blk(tokens, None)
    return tokens
```

### 方案 B：频谱门控中的轨道条件化

`BlockSpectralGating`（`spectformer.py:80-107`）在傅里叶频域工作，使用可学习的复数权重进行频率过滤：

```python
# spectformer.py:46
self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2) * 0.02)
```

可以改为**轨道条件化的频谱权重**：

```python
class ConditionalSpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8, condition_dim=768):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2) * 0.02)
        # 条件调制网络
        self.weight_modulation = nn.Sequential(
            nn.Linear(condition_dim, dim),
            nn.GELU(),
            nn.Linear(dim, h * w * dim),
        )
        self.w, self.h, self.dim = w, h, dim
    
    def forward(self, x, orbit_embed):
        B, N, C = x.shape
        a = b = int(math.sqrt(N))
        x = x.view(B, a, b, C)
        
        # 轨道条件 → 频率权重调制
        weight_mod = self.weight_modulation(orbit_embed)  # B, h*w*dim
        weight_mod = weight_mod.view(-1, self.h, self.w, self.dim, 1)
        modulated_weight = self.complex_weight * weight_mod
        
        # ... 后续 FFT + 乘法 + iFFT
```

这对纹理学习特别有意义：不同观测角度下，太阳表面特征（如活动区、冕洞）的特定空间频率成分会发生变化——条件化的频谱门控可以自适应地增强或抑制这些频率成分。

---

## 五、问题 5：冻结 Encoder + 轨道 Embedding 是否可行

### 结论：完全可行，且是最佳起始策略

**理由**：

1. **Surya 已有成熟的 fine-tuning 生态**：下游任务（`downstream_examples/ar_segmentation/finetune.py`）展示了冻结 backbone + 添加新模块的模式。当 `finetune=True` 时，backbone 输出 tokens 而不解码（`helio_spectformer.py:278-279`）：

   ```python
   # helio_spectformer.py:278-279
   if self.finetune:
       return tokens  # 不经过 decoder，给下游使用
   ```

2. **通道适配器模式已有先例**（`ar_segmentation/models.py:16-29`）：

   ```python
   class ChannelAdapter(nn.Module):
       def __init__(self, model, num_data_chans, time_dim, out_chans):
           super().__init__()
           self.adapter = nn.Conv3d(self.num_data_chans, self.out_chans, 
                                     kernel_size=1, padding=0)
           self.model = model  # 预训练的 backbone，可冻结
       
       def forward(self, batch):
           batch['ts'] = self.adapter(batch['ts'])
           return self.model(batch)
   ```

3. **LoRA 微调模式已支持**（`finetune.py:306-372`）：使用 `peft` 库的 LoRA 对 Attention 的 qkv、proj、fc1、fc2 等添加低秩适配器，冻结其余参数。

### 推荐实现架构

```
┌──────────────────────────────────────────────────┐
│  冻结部分 (不更新梯度)                             │
│  ┌──────────────┐  ┌────────────────┐            │
│  │  Embedding   │→│  SpectFormer   │            │
│  │  (冻结)      │  │  (冻结/LoRA)   │            │
│  └──────────────┘  └───────┬────────┘            │
│                            │                     │
│                      ↑ adaLN 注入                │
│                 ┌────┴────────┐                  │
│                 │  Orbit      │  ← 可训练        │
│                 │  Encoder    │                  │
│                 │  (MLP)      │                  │
│                 └────┬────────┘                  │
│                      │                           │
│                 ┌────┴────────┐                  │
│                 │ OrbitAware  │  ← 可训练        │
│                 │ FlowModel   │  (几何缩放)      │
│                 └─────────────┘                  │
│                                                  │
│  可训练部分                                       │
│  ┌──────────────────────────┐                    │
│  │  Decoder (解冻或替换)      │                    │
│  └──────────────────────────┘                    │
└──────────────────────────────────────────────────┘
```

### 参数量估算

| 组件 | 参数量 | 是否冻结 |
|------|--------|----------|
| Embedding（LinearEmbedding）| ~100M | ✅ 冻结 |
| SpectFormer Backbone | ~250M | ✅ 冻结（或 LoRA）|
| Orbit Encoder（6→128→768）| ~100K | ❌ 可训练 |
| adaLN 调制层（8 层, 768→6×768）| ~28M | ❌ 可训练 |
| OrbitAwareFlowModel | ~20K | ❌ 可训练 |
| Decoder | ~16M | ❌ 可训练 |
| **总计可训练** | **~44M** | (~12% 的总参数) |
| **总计（含 LoRA）** | **~55M** | (~15% 的总参数) |

---

## 六、问题 6：Condition Injection 是否可行

### 结论：完全可行，且 Surya 的 adaLN 机制就是为此目的设计的

### 现有注入点一览

| 注入位置 | 文件:行号 | 机制 | 适合注入什么 | 侵入性 |
|----------|----------|------|-------------|--------|
| **`BlockAttention.adaLN`** | `spectformer.py:149-167` | Scale+Shift+Gate (6 参数) | 纹理变化（观测方位）| 低 — 机制已存在 |
| **`HelioFlowModel.grid_sample`** | `flow.py:70-73` | 空间变换 (grid sample) | 几何缩放（距离）| 中 — 需扩展架构 |
| **Token 序列拼接** | `helio_spectformer.py:268` | 额外条件 token | 全局轨道上下文 | 低 — 基本操作 |
| **`LinearEmbedding.pos_embed`** | `embedding.py:79-127` | Fourier 位置编码 | 距离感知的空间尺度 | 中 — 需改造 PE |
| **`BlockSpectralGating`** | `spectformer.py:43-77` | 频域复数权重 | 视角相关的频率特征 | 中 — 需扩展 FFT 模块 |
| **Cross-Attention** | `embedding.py:199-246` | Perceiver 交叉注意力 | 轨道信息作为 context | 低 — 机制已存在 |

### 具体实现方案

#### 注入点 1 — adaLN（纹理条件注入）⭐ 最高优先级

修改 `BlockAttention.forward` 的第二个参数语义，从 noise 改为 condition：

```python
# spectformer.py:158 — 修改后
def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Token features (B, N, D)
        c: Condition embedding (B, 1, D) — orbit embedding
    """
    if self.adaLN_modulation is not None and c is not None:
        (shift_mha, scale_mha, gate_mha,
         shift_mlp, scale_mlp, gate_mlp
        ) = self.adaLN_modulation(c).chunk(6, dim=2)
    else:
        shift_mha, scale_mha, gate_mha = 3 * (1.0,)
        shift_mlp, scale_mlp, gate_mlp = 3 * (1.0,)

    x = x + gate_mha * self.drop_path(
        self.attn(self.norm1(x) * scale_mha + shift_mha)
    )
    x = x + gate_mlp * self.drop_path(
        self.mlp(self.norm2(x) * scale_mlp + shift_mlp)
    )
    return x
```

#### 注入点 2 — 轨道感知位置编码（尺度条件）

修改 `embedding.py:81-113`，让 Fourier 位置编码的频率可以受距离调制，使模型感知"同样大小的 patch 在不同距离下对应的太阳物理尺度不同"：

```python
# embedding.py:_generate_position_encoding — 修改后
def _generate_position_encoding(self, img_size, patch_size, embed_dim, 
                                  use_orbit_scale=False):
    x = torch.linspace(0.0, 1.0, img_size // patch_size)
    y = torch.linspace(0.0, 1.0, img_size // patch_size)
    x, y = torch.meshgrid(x, y, indexing="xy")
    fourier_signal = []

    # 基础频率（当 use_orbit_scale=True 时，频率会在 forward 中根据距离动态调整）
    base_frequencies = torch.linspace(1, (img_size // patch_size) / 2.0, 
                                       embed_dim // 4)
    
    if use_orbit_scale:
        # 存储基础频率，等待 forward 中根据距离进行缩放
        self.register_buffer("base_frequencies", base_frequencies)
        # 同时存储坐标网格供 forward 使用
        self.register_buffer("coord_x", x)
        self.register_buffer("coord_y", y)
    else:
        for f in base_frequencies:
            fourier_signal.extend([
                torch.cos(2.0 * torch.pi * f * x),
                torch.sin(2.0 * torch.pi * f * x),
                torch.cos(2.0 * torch.pi * f * y),
                torch.sin(2.0 * torch.pi * f * y),
            ])
        fourier_signal = torch.stack(fourier_signal, dim=2)
        fourier_signal = rearrange(fourier_signal, "h w c -> 1 (h w) c")
        self.register_buffer("pos_embed", fourier_signal)
```

在 `LinearEmbedding.forward` 中动态生成位置编码：

```python
def forward(self, x, dt, orbit_params=None):
    x = self.patch_embed(x)
    
    if self.use_orbit_scale and orbit_params is not None:
        r = torch.norm(orbit_params[:, :3], dim=1, keepdim=True)  # B, 1
        scale = 149597870.7 / r  # 距离比
        # 频率随距离缩放：越远 → 频率越低 → 更大的空间尺度
        pos_embed = self._generate_dynamic_pe(scale)
    else:
        pos_embed = self.pos_embed
    
    x = x + pos_embed
    x = self.pos_drop(x)
    return x
```

#### 注入点 3 — HelioFlowModel 几何缩放（距离条件）

修改 `flow.py:37-81` 的 forward，使其接收轨道参数并产生距离感知的 flow：

```python
# flow.py — 修改后
class HelioFlowModel(nn.Module):
    def __init__(self, img_size=(4096, 4096), use_latitude_in_learned_flow=False,
                 use_orbit_condition=False):
        super().__init__()
        # ... 原有初始化 ...
        
        self.use_orbit_condition = use_orbit_condition
        if use_orbit_condition:
            self.ref_distance = 149597870.7  # 1 AU in km
            # 轨道编码 → 缩放因子的精细调制
            self.scale_modulation = nn.Sequential(
                nn.Linear(6, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

    def forward(self, batch):
        x = batch["ts"]
        orbit_params = batch.get("orbit_params", None)
        B, C, T, H, W = x.shape
        
        if T == 1:
            x = x[:, :, -1, :, :]
        else:
            x = (x[:, :, -1, :, :] + x[:, :, -2, :, :]) / 2

        if self.use_orbit_condition and orbit_params is not None:
            r = torch.norm(orbit_params[:, :3], dim=1, keepdim=True)  # B, 1
            base_scale = self.ref_distance / r.squeeze()  # B
            scale_mod = self.scale_modulation(orbit_params).squeeze()  # B
            scale = (base_scale * scale_mod).view(B, 1, 1, 1)
        else:
            scale = 1.0

        if self.use_latitude_in_learned_flow:
            # ... latitude-aware flow ...
            pass
        else:
            scaled_grid = self.grid * scale if self.use_orbit_condition else self.grid
            flow_field = scaled_grid + self.flow_generator(self.higher_modes)
        
        flow_field = flow_field.expand(B, H, W, 2)
        y_hat = F.grid_sample(x, flow_field, mode="bilinear",
                               padding_mode="border", align_corners=False)
        return y_hat
```

---

## 七、推荐的完整修改方案

### 需要修改的文件和具体行

| 优先级 | 文件 | 行号 | 修改内容 |
|--------|------|------|----------|
| **P0** | `surya/models/helio_spectformer.py` | 29-52 | `__init__` 增加 `use_orbit_condition` 参数和 `orbit_encoder` |
| **P0** | `surya/models/helio_spectformer.py` | 242-268 | `forward` 中增加轨道参数提取、编码、条件传递 |
| **P0** | `surya/models/spectformer.py` | 183-296 | `SpectFormer.forward` 传递 `condition` 到 `BlockAttention` |
| **P1** | `surya/models/spectformer.py` | 110-180 | `BlockAttention.forward` 改用 orbit_embed 驱动 adaLN |
| **P1** | `surya/models/flow.py` | 6-81 | `HelioFlowModel` 增加距离感知缩放 |
| **P2** | `surya/models/embedding.py` | 55-127 | `LinearEmbedding` 增加轨道感知的 Fourier 位置编码 |
| **P2** | `surya/datasets/helio.py` | 367-470 | `_get_index_data` 增加轨道参数字段 |
| **P3** | `surya/utils/config.py` | 64-116 | `ModelConfig` 增加轨道条件相关配置项 |

### 修改后的 HelioSpectFormer 核心逻辑（完整伪代码）

```python
class HelioSpectFormer(nn.Module):
    def __init__(self, ..., use_orbit_condition=False):
        super().__init__()
        # ... 原有初始化代码 ...
        
        self.use_orbit_condition = use_orbit_condition
        
        if use_orbit_condition:
            # 1. 轨道编码器：6D (x,y,z,vx,vy,vz) → embed_dim
            self.orbit_encoder = nn.Sequential(
                nn.Linear(6, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            
            # 2. 轨道感知的 Flow 模型（几何缩放）
            self.orbit_flow = HelioFlowModel(
                img_size=(img_size, img_size),
                use_orbit_condition=True,
            )
            
            # 3. backbone 中的 adaLN 将在初始化时启用
            #    SpectFormer(..., adaLN_for_orbit=True)
    
    def forward(self, batch):
        x = batch["ts"]                        # B, C, T, H, W
        dt = batch["time_delta_input"]         # B, T
        orbit = batch.get("orbit_params")      # B, 6  (可选字段)
        
        # === 步骤 0：轨道编码 ===
        orbit_embed = None
        if self.use_orbit_condition and orbit is not None:
            orbit_embed = self.orbit_encoder(orbit)       # B, embed_dim
            orbit_embed = orbit_embed.unsqueeze(1)         # B, 1, embed_dim
        
        # === 步骤 1：几何变换（距离 → 缩放）===
        if self.use_orbit_condition and orbit is not None:
            x_flow = self.orbit_flow(batch)               # B, C, H, W
            x = torch.concat((x, x_flow.unsqueeze(2)), dim=2)  # B, C, T+1, H, W
            # (如果需要 Perceiver 嵌入，同时调整 dt)
        
        # === 步骤 2：Token 嵌入 ===
        tokens = self.embedding(x, dt, orbit_params=orbit 
                                if self.use_orbit_condition else None)
        
        # === 步骤 3：Backbone 处理（条件注入）===
        tokens = self.backbone(tokens, condition=orbit_embed)
        
        # === 步骤 4：解码 ===
        if self.finetune:
            return tokens
        forecast_hat = self.unembed(tokens)
        
        return forecast_hat
```

### SpectFormer 修改

```python
class SpectFormer(nn.Module):
    def __init__(self, ..., use_orbit_adaLN=False):
        # ...
        for i in range(depth):
            if i < n_spectral_blocks:
                layer = BlockSpectralGating(...)
            else:
                layer = BlockAttention(
                    ...,
                    adaLN=True if use_orbit_adaLN else 
                            (True if ensemble is not None else False),
                )
    
    def forward(self, tokens, condition=None):
        for i, blk in enumerate(chain(self.blocks_spectral_gating, 
                                        self.blocks_attention)):
            if isinstance(blk, BlockAttention):
                tokens = blk(tokens, condition)  # ← 传递轨道条件
            else:
                tokens = blk(tokens, None)
        return tokens
```

---

## 八、训练策略

### 阶段 1：冻结 Backbone + 训练条件注入层（推荐起步）

```
训练组件：
  ✅ Embedding     — 冻结
  ✅ SpectFormer   — 冻结（含 attention 权重）
  ❌ adaLN 调制层   — 训练
  ❌ Orbit Encoder — 训练
  ❌ Orbit Flow    — 训练
  ❌ Decoder       — 训练/替换

目标：让条件模块学会基本的距离→大小、方位→纹理映射
预计可训练参数：~30M (8%)
学习率：1e-4 ~ 5e-4
```

### 阶段 2：LoRA 微调 Backbone

```
新增训练：
  ❌ LoRA on BlockAttention.qkv
  ❌ LoRA on BlockAttention.proj
  ❌ LoRA on BlockAttention.mlp.fc1, fc2

目标：让 backbone 学会利用轨道信息调整内部表征
预计新增可训练参数：~10M
总可训练参数：~40M (11%)
学习率：5e-5 ~ 1e-4
```

### 阶段 3（可选）：全参数微调

```
训练组件：全部参数

目标：最终性能优化
学习率：1e-5 ~ 5e-5（很小，避免灾难性遗忘）
Epochs：10-20
```

---

## 九、数据集改造要点

### HelioNetCDFDataset 修改（`helio.py:367-470`）

```python
def _get_index_data(self, idx: int) -> dict:
    # ... 原有逻辑 ...
    
    # === 新增：读取轨道参数 ===
    # 方案 1：从 NetCDF 属性中读取
    orbit_params = self._load_orbit_params(required_timesteps)
    
    # 方案 2：从单独轨道文件读取（推荐，解耦数据）
    # orbit_params = self.orbit_index.loc[reference_timestep]
    
    return {
        "ts": stacked_inputs,
        "time_delta_input": time_delta_input_float,
        "forecast": stacked_targets,
        "lead_time_delta": lead_time_delta_float,
        "orbit_params": orbit_params,  # ← 新增 [B, 6]
    }, metadata
```

### 轨道参数归一化

```python
# 在数据预处理时统一归一化
def normalize_orbit_params(df):
    """归一化轨道参数"""
    pos_cols = ['x_km', 'y_km', 'z_km']
    vel_cols = ['vx_kms', 'vy_kms', 'vz_kms']
    
    df[pos_cols] = df[pos_cols] / 149597870.7   # 除以 1 AU
    df[vel_cols] = df[vel_cols] / 29.8          # 除以地球轨道速度
    
    return df
```

---

## 十、总结

| 问题 | 答案 | 关键代码位置 |
|------|------|-------------|
| 轨道参数输入 | 扩展 batch 字典 + MLP 编码器 (6→128→768) | `helio_spectformer.py:242` |
| 学习大小变化 | 扩展 HelioFlowModel 的 grid_sample 支持距离感知缩放 | `flow.py:70-73` |
| 学习纹理变化 | 启用 adaLN 将轨道信息注入每个 Attention 层（6 种调制参数）| `spectformer.py:149-167` |
| 最佳修改模块 | BlockAttention (adaLN) + HelioFlowModel + LinearEmbedding | 上述全文件 |
| 冻结 Encoder | ✅ 完全可行，推荐作为第一阶段策略 | `finetune=True` 模式 + LoRA |
| Condition Injection | ✅ 完全可行，adaLN 机制已就位，只需改为 orbit_embed 输入 | 无需从零设计 |

**核心优势**：Surya 的 adaLN 和 HelioFlowModel 已经为条件注入提供了完美的架构基础。你不需要从零设计注入机制——只需"激活"并重新定向这些已有模块。具体来说：

1. **adaLN**（`spectformer.py:149-156`）：已实现完整的 Scale+Shift+Gate 调制流水线——当前只用于 ensemble noise 注入，可以直接用于轨道条件注入。
2. **HelioFlowModel**（`flow.py`）：已实现基于 `grid_sample` 的空间变换——可以自然扩展为距离感知的缩放变换。
3. **Fine-tuning 模式**（`helio_spectformer.py:278-279`）：已支持冻结 backbone 只返回 tokens——完美适配条件模块的训练。
4. **LoRA 支持**（`finetune.py:306-372`）：已集成 PEFT LoRA——可在冻结大部分参数的同时微调 attention 权重。
