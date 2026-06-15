# surya_orbit 代码架构详解

> 本文档逐类解释 `surya_orbit/` 中每个文件的数据流、类职责、以及相对于 Surya 原始代码的具体改动。

---

## 一、文件总览

```
surya_orbit/
├── orbit_dataset.py          ← 数据入口: NC/CSV → batch 字典
├── orbit_flow.py             ← 几何引擎: 距离 → grid_sample 缩放
├── orbit_spectformer.py      ← 模型外壳: 替换 FlowModel + 重写 forward
├── train_phase1.py           ← 训练脚本: 冻结/解冻 + 训练循环
├── infer_phase1.py           ← 推理脚本: 加载模型 + 生成多距离图像
├── test_phase1.py            ← 验证脚本: 5 项自动化检查
├── config_phase1.yaml        ← 配置
├── orbit_dataset_fast.py     ← 加速版 Dataset (.npy)
├── preprocess_nc.py          ← NC → .npy 预处理
├── check_nc.py               ← NC 文件诊断工具
├── README.md                 ← 使用说明
└── README_PREPROCESS.md      ← 预处理说明
```

核心数据流只涉及其中 4 个文件：

```
orbit_dataset.py  →  orbit_spectformer.py  →  orbit_flow.py  →  Surya 原始模块
   (数据准备)          (模型入口)              (几何缩放)        (Embedding/Backbone/Decoder)
```

---

## 二、数据全流程：每一步的 Shape 变化

以当前配置为例：`pooling=4, img_size=1024, batch_size=1, embed_dim=1280`。

```
═══════════════════════════════════════════════════════════════════════════
Step 0 — orbit_dataset.py: OrbitDataset.__getitem__()
═══════════════════════════════════════════════════════════════════════════

NC 文件 (20140107_1500.nc)
  ├── 13 个通道变量, 每个 4096×4096 float32
  └── meta_0/meta_1 中的 dsun_obs 字段

操作:
  1. xr.open_dataset → 读 13 通道 → (13, 4096, 4096)
  2. dsun_obs: 147097819 m → 147097819 km (source_distance_km)
  3. signum-log 归一化: data * sl_scale → sign*log1p(|x|) → (x-mean)/(std+eps)
  4. pool (4×4 mean): (13, 4096, 4096) → (13, 1024, 1024)
  5. 随机采样 SPO 距离: target_distance_km ∈ [0.98, 3.18] AU
  6. 几何伪真值: zoom(image, source_km/target_km) → (13, 1024, 1024)
  7. 时间序列: stack([image, image], axis=1) → (13, 2, 1024, 1024)

返回 batch (单样本):
{
    "ts":                  (13, 2, 1024, 1024)  float32   ← 输入
    "time_delta_input":    (2,)                  float32   ← [-0.2, 0.0]
    "source_distance_km":  scalar                float32   ← ~147097819
    "target_distance_km":  scalar                float32   ← 随机 SPO
    "forecast":            (13, 1, 1024, 1024)  float32   ← 伪真值
    "lead_time_delta":     (1,)                  float32   ← [0.2]
}

DataLoader collate → 加 B 维度:
    "ts":                  (1, 13, 2, 1024, 1024)   ← B=1
    "forecast":            (1, 13, 1, 1024, 1024)


═══════════════════════════════════════════════════════════════════════════
Step 1 — orbit_spectformer.py: OrbitHelioSpectFormer.forward()
═══════════════════════════════════════════════════════════════════════════

batch 进入 forward():

① OrbitAwareFlowModel.forward(batch)           ← orbit_flow.py
   输入: batch["ts"] = (1, 13, 2, 1024, 1024)
   操作: 取最后帧 → 读 source_km / target_km → 算 scale → grid_sample 缩放
   输出: y_hat_flow = (1, 13, 1024, 1024)

② 拼接
   x = concat([ts, y_hat_flow.unsqueeze(2)], dim=2)
   (1, 13, 2+1, 1024, 1024) = (1, 13, 3, 1024, 1024)
   ↑ T 从 2 变成 3 (原始 2 帧 + flow 输出 1 帧)

③ LinearEmbedding.forward(x, dt)               ← Surya 原始, 冻结
   (1, 13, 3, 1024, 1024) → flatten → (1, 39, 1024, 1024)
   → Conv2d(39→1280, kernel=16) → (1, 1280, 64, 64)
   → rearrange → (1, 4096, 1280)
   → + pos_embed → (1, 4096, 1280)
   L = (1024/16)² = 4096, D = 1280

④ SpectFormer.forward(tokens)                  ← Surya 原始, 冻结
   (1, 4096, 1280) → 10 层 (2×SpectralGating + 8×Attention) → (1, 4096, 1280)
   shape 不变

⑤ LinearDecoder.forward(tokens)                ← Surya 原始, 训练
   (1, 4096, 1280) → rearrange → (1, 1280, 64, 64)
   → Conv2d(1280→3328) + PixelShuffle(16) → (1, 13, 1024, 1024)

⑥ 残差连接
   forecast_hat = ⑤ + ①(y_hat_flow)
   (1, 13, 1024, 1024)

输出 → Loss = MSE(forecast_hat, batch["forecast"][:,:,0,:,:])
═══════════════════════════════════════════════════════════════════════════
```

---

## 三、逐类详解

### 3.1 orbit_dataset.py — `OrbitDataset`

**继承关系**：直接继承 `torch.utils.data.Dataset`，不继承 Surya 的任何类。

**为什么独立**：Surya 的 `HelioNetCDFDataset` 是按"时间序列预测"组织的（需要连续时间戳），而我们需要的是"随机配对"（任意 SDO 图 + 任意 SPO 距离）。

**职责**：

```
┌─────────────────────────────────────────────┐
│  __init__():                                │
│    1. 加载 SPO CSV → self.orbit_df          │
│    2. 缓存 signum-log 归一化参数             │
│       (mean, std, epsilon, sl_scale_factor)  │
│    3. 每个通道独立: shape (13, 1, 1)         │
├─────────────────────────────────────────────┤
│  __getitem__(idx):                          │
│    1. 选 NC 文件 (idx // samples_per_file)  │
│    2. _load_and_transform():                │
│       a. xr.open_dataset → (13, 4096, 4096) │
│       b. _extract_dsun_obs():               │
│          读 meta_0/meta_1 JSON → dsun_obs(m) │
│          → /1000 → source_distance_km        │
│       c. signum-log 归一化                   │
│       d. mean pool (4096→1024)               │
│    3. 随机取 SPO 距离                        │
│       row = orbit_df.sample(1)              │
│       target_distance_km = row["distance_km"]│
│    4. _geometric_scale():                   │
│       zoom = source_km / target_km          │
│       scipy.ndimage.zoom(每通道, order=3)   │
│    5. 组装 batch 字典                        │
└─────────────────────────────────────────────┘
```

**返回的 batch 字典字段**：

| 键 | Shape | 类型 | 含义 |
|----|-------|------|------|
| `ts` | (13, 2, H, W) | float32 | 输入图像序列（同一张图复制两次）|
| `time_delta_input` | (2,) | float32 | 时间增量，固定 [-0.2, 0.0] |
| `source_distance_km` | scalar | float32 | SDO 对日距离，从 NC 元数据提取 |
| `target_distance_km` | scalar | float32 | SPO 目标距离，随机采样 |
| `forecast` | (13, 1, H, W) | float32 | 几何缩放伪真值 |
| `lead_time_delta` | (1,) | float32 | 预测时间差，固定 [0.2] |

**距离提取逻辑**（`_extract_dsun_obs`）：

与 `single_scale.py:38-71` 完全一致——遍历通道变量的 `meta_0` 和 `meta_1` JSON 属性，提取 `dsun_obs`（米）→ 除以 1000 得公里。若所有通道都找不到，回退到 1 AU (149,597,870.7 km)。

**为什么同一张图复制两次？**

Surya 的输入格式要求 `(C, T, H, W)` 且 T≥2（它设计为多帧时序预测）。Phase 1 只做几何变换，不依赖时序信息——同一张图复制两次，模型看到的是"静止的太阳"，它只需关注轨道距离参数。

---

### 3.2 orbit_flow.py — `OrbitAwareFlowModel`

**继承关系**：

```
nn.Module
  └── HelioFlowModel          (surya/models/flow.py)
        └── OrbitAwareFlowModel  (surya_orbit/orbit_flow.py)   ← 我们写的
```

**父类 `HelioFlowModel` 做了什么**：

```python
# flow.py:12-16 — 创建归一化坐标网格
u = linspace(-1, 1, H)      # 像素列坐标, 归一化到 [-1, 1]
v = linspace(-1, 1, W)      # 像素行坐标, 归一化到 [-1, 1]
self.grid = meshgrid(u, v)  # (1, H, W, 2), 最后一个维度是 (u,v) 对

# flow.py:27-29 — 坐标基函数 (用于学习微小的空间偏移)
self.higher_modes = stack([u, v])  # (1, H, W, 2)

# flow.py:31-35 — 可学习的微小变形
self.flow_generator = Sequential(
    Linear(2 → 128), GELU, Linear(128 → 2)
)
```

父类的 forward（`flow.py:37-81`）：

```python
# 固定 flow: grid 不动, 只加一个学习到的微小偏移
flow_field = self.grid + self.flow_generator(self.higher_modes)
#           (1,H,W,2)     (1,H,W,2)    [可学习, 但值很小]

# 用 flow_field 对输入图像做空间采样
y_hat = F.grid_sample(x, flow_field)
```

**关键理解**：父类的 `flow_field` 几乎就是 identity（`self.grid` 本身）。`flow_generator` 只产生微小的修正（数据无关的、固定的）。

**子类 `OrbitAwareFlowModel` 改了什么**：

```python
class OrbitAwareFlowModel(HelioFlowModel):

    def __init__(self, img_size):
        super().__init__(img_size)
        # ── 新增: 距离 → 缩放修正 MLP ──
        self.scale_corrector = Sequential(
            Linear(1 → 16),   # 输入: 目标距离 (AU)
            GELU(),
            Linear(16 → 1),   # 输出: 缩放修正值
        )

    def forward(self, batch):
        # Step 1: 取最后一帧 (与父类相同)
        x = batch["ts"][:, :, -1, :, :]  # (B, C, H, W)

        # Step 2: 读距离 — 父类没有这一步!
        r_src = batch["source_distance_km"]  # (B,)
        r_tgt = batch["target_distance_km"]  # (B,)

        # Step 3: 计算缩放 — 父类没有!
        base_scale = r_tgt / r_src            # 确定性几何缩放
        correction = self.scale_corrector(r_tgt_AU)  # 可学习微调 ±10%
        scale = base_scale * (1 + 0.1 * tanh(correction))

        # Step 4: 缩放 grid — 父类的 grid 是固定的!
        scaled_grid = self.grid * scale       # (B, H, W, 2)

        # Step 5: 加上微小局部修正 (复用父类 flow_generator)
        flow_field = scaled_grid + self.flow_generator(self.higher_modes)

        # Step 6: 采样 (同父类)
        y_hat = F.grid_sample(x, flow_field)
        return y_hat
```

**父类 vs 子类对比**：

```
┌─────────────────────────────┬──────────────────────────────────┐
│  父类 HelioFlowModel        │  子类 OrbitAwareFlowModel        │
├─────────────────────────────┼──────────────────────────────────┤
│  flow_field = grid          │  flow_field = grid × scale       │
│             + fixed_offset  │             + fixed_offset       │
│                             │                                 │
│  grid 固定不变               │  grid 按距离比缩放               │
│  不需要轨道参数              │  需要 source_km + target_km      │
│  对任何输入都产生相同变形    │  不同距离产生不同缩放            │
│                             │                                 │
│  scale: 始终 ≈ 1.0          │  scale: 0.31 ~ 3.2              │
│                             │  (0.98AU→3.18AU 时 scale≈3.2)   │
└─────────────────────────────┴──────────────────────────────────┘
```

**F.grid_sample 的几何直觉**：

```
grid 是"输出像素在输入图像中对应的归一化坐标"

原始 grid (identity):
  输出位置 (0,0) → 采样输入位置 (0,0)
  输出位置 (1,1) → 采样输入位置 (1,1)

缩放后 grid × 3.2:
  输出位置 (0,0) → 采样输入位置 (0,0)     ← 中心不变
  输出位置 (0.5,0) → 采样输入位置 (1.6,0)  ← 超出边界! (padding_mode='border')
  输出位置 (0.2,0) → 采样输入位置 (0.64,0) ← 从更内部的区域采样

结果：输出图像的"有效内容区域"收缩了 → 太阳看起来变小了
     (因为太阳只占图像中心一小部分, 外围是太空)
```

---

### 3.3 orbit_spectformer.py — `OrbitHelioSpectFormer`

**继承关系**：

```
nn.Module
  └── HelioSpectFormer          (surya/models/helio_spectformer.py)
        └── OrbitHelioSpectFormer  (surya_orbit/orbit_spectformer.py)  ← 我们写的
```

**这是最关键的改动所在地**。修改了两处：`__init__` 和 `forward`。

#### 改动 1: `__init__` — 替换 FlowModel

```python
class OrbitHelioSpectFormer(HelioSpectFormer):
    def __init__(self, **kwargs):
        # 先调父类 __init__
        # 父类会创建:
        #   self.learned_flow_model = HelioFlowModel(...)   ← 原始版本
        #   self.embedding = LinearEmbedding(...)           ← 不变
        #   self.backbone = SpectFormer(...)                ← 不变
        #   self.unembed = LinearDecoder(...)               ← 不变
        super().__init__(**kwargs)

        # 替换: 把父类创建的 HelioFlowModel 扔掉, 换成我们的
        if self.learned_flow:
            img_size = kwargs.get("img_size", 4096)
            self.learned_flow_model = OrbitAwareFlowModel(
                img_size=(img_size, img_size),
            )
```

**为什么可以这样**：`super().__init__()` 会创建完整的 Surya 模型（含原始的 `HelioFlowModel`），然后我们立刻把 `self.learned_flow_model` 指向一个新的 `OrbitAwareFlowModel` 实例。原始的 FlowModel 被 Python 垃圾回收，不影响其余模块（Embedding/Backbone/Decoder 完全不变）。

#### 改动 2: `forward` — 去掉"跳过 pipeline"的逻辑

**父类的 forward（`helio_spectformer.py:242-318`）有一个关键行为**：

```python
# 父类 forward 关键片段:
if self.learned_flow:
    y_hat_flow = self.learned_flow_model(batch)

    # ⚠️ 父类的逻辑:
    if any(FlowModel 的 requires_grad):
        return y_hat_flow   # ← 直接返回! 跳过 Embedding/Backbone/Decoder!
    else:
        # 只有 FlowModel 冻结时才走完整 pipeline
        x = concat(ts, y_hat_flow)
        ...

tokens = self.embedding(x, dt)
tokens = self.backbone(tokens)
forecast_hat = self.unembed(tokens)
forecast_hat = forecast_hat + y_hat_flow   # 残差连接
```

这个设计是 Surya 原作者为"FlowModel 单独训练模式"准备的——如果 FlowModel 可训练，就只训练它，Encoder/Decoder 完全不走。

**我们的 forward 做了什么**：

```python
def forward(self, batch):
    x = batch["ts"]
    dt = batch["time_delta_input"]
    B, C, T, H, W = x.shape

    # ① FlowModel: 距离感知缩放 (始终执行)
    y_hat_flow = self.learned_flow_model(batch)  # (B, C, H, W)

    # ② 粘贴 flow 输出 (不检查 requires_grad, 始终走完整 pipeline)
    x = torch.concat((x, y_hat_flow.unsqueeze(2)), dim=2)
    # (B, C, T+1, H, W)

    # ③→④→⑤ 完整 Surya pipeline
    tokens = self.embedding(x, dt)       # (B, L, D)
    tokens = self.backbone(tokens)       # (B, L, D)
    forecast_hat = self.unembed(tokens)  # (B, C, H, W)

    # ⑥ 残差连接
    forecast_hat = forecast_hat + y_hat_flow

    return forecast_hat
```

**区别总结**：

```
┌──────────────────────────────┬──────────────────────────────────┐
│  父类 HelioSpectFormer       │  子类 OrbitHelioSpectFormer      │
├──────────────────────────────┼──────────────────────────────────┤
│  FlowModel 可训练时:         │  FlowModel 可训练时:              │
│    return flow_output        │    仍然走完整 pipeline            │
│    (Encoder/Decoder 不走)    │    (Encoder/Decoder 也参与)       │
│                              │                                  │
│  只能单独训练 FlowModel      │  FlowModel + Decoder 可同时训练   │
│  或单独训练 Encoder/Decoder  │                                  │
│                              │  残差连接让 Decoder 只需学习      │
│  无残差连接                  │  "FlowModel 没处理好的部分"       │
└──────────────────────────────┴──────────────────────────────────┘
```

**为什么需要残差连接**：`y_hat_flow` 已经通过 grid_sample 做了基础几何缩放。Decoder 的输出在此基础上做精细修正。如果 Decoder 什么也不做（输出 ≈ 0），最终结果至少是几何上正确的（归功于 FlowModel）。这提供了训练的"安全网"。

---

## 四、训练时的冻结/解冻策略

```python
# train_phase1.py 中的逻辑:

# Step 1: 全部冻结
for param in model.parameters():
    param.requires_grad = False

# Step 2: 只解冻两个模块
for param in model.unembed.parameters():           # LinearDecoder
    param.requires_grad = True
for param in model.learned_flow_model.parameters(): # OrbitAwareFlowModel
    param.requires_grad = True
```

**各模块的冻结状态**：

```
OrbitHelioSpectFormer
├── learned_flow_model (OrbitAwareFlowModel)   ✅ 训练 (~5K 参数)
│   ├── scale_corrector (MLP: 1→16→1)          ✅ 训练
│   └── flow_generator (Linear: 2→128→2)       ✅ 训练 (父类继承)
│
├── embedding (LinearEmbedding)                ❄️ 冻结 (~100M 参数)
│   ├── patch_embed.proj (Conv2d)              ❄️
│   └── pos_embed (Fourier 位置编码)           ❄️
│
├── backbone (SpectFormer)                     ❄️ 冻结 (~250M 参数)
│   ├── blocks_spectral_gating[0,1]            ❄️
│   └── blocks_attention[0..7]                 ❄️ (adaLN 也不训练)
│
└── unembed (LinearDecoder)                    ✅ 训练 (~16M 参数)
    └── unembed.0 (Conv2d 1280→3328)          ✅ 训练
```

---

## 五、关键设计决策

### 5.1 为什么用距离标量而不是 6D 向量？

最初的设计用了完整的 6D 轨道参数 `(x, y, z, vx, vy, vz)`。后来改为直接用对日直线距离 (km)。

**原因**：`F.grid_sample` 操作的是**图像像素归一化坐标 `(u, v) ∈ [-1, 1]`**。缩放图像只需要一个数——距离比。6D 向量中的方向信息（x/y/z 的比例）描述的是"从哪个方向看太阳"，影响的是**纹理**（Phase 2 才涉及），而不是**缩放**。

Phase 1 只做几何缩放，用距离标量是最简洁的。

### 5.2 为什么 scale_corrector 输入 1 维而不是 6 维？

同上。Phase 1 中 MLP 的输入是目标距离 (AU)，输出是一个缩放修正值。距离越远 → 修正可能略微不同（比如 0.98 AU 和 3.2 AU 的边缘效应不同），MLP 可以学习这个差异。但本质上一个标量就够了。

### 5.3 为什么 `zoom_scale` 和 `grid_scale` 互为倒数？

```
Dataset 中 (生成伪真值):
  zoom_scale = source_km / target_km
  用 scipy.ndimage.zoom(image, zoom_scale)
  zoom=0.5 表示图像缩小到 50%

FlowModel 中 (grid_sample):
  grid_scale = target_km / source_km
  grid × 2.0 表示采样范围扩张 → 内容收缩 → 效果等同于 zoom=0.5

两者互为倒数, 但最终效果一致——只是实现方式不同。
```

### 5.4 为什么第一轮训练 10 分钟？

时间全耗在 CPU 上：

```
每个样本 (~2.5 秒):
  xr.open_dataset + h5netcdf 读取  →  ~1.5 秒
  signum-log 归一化               →  ~0.3 秒
  mean pool (13×4096×4096→1024²)  →  ~0.2 秒
  scipy zoom (13 通道, order=3)   →  ~0.3 秒
  numpy stack/copy                 →  ~0.2 秒

230 样本 × 2.5 秒 ≈ 9.6 分钟 CPU + ~2 分钟 GPU = ~12 分钟/轮
```

解决：`preprocess_nc.py` 把 NC→归一化→pool 预处理为 `.npy`，训练时 `np.load()` ~50ms，每轮降至 2-3 分钟。

### 5.5 为什么 pooling=4 (1024×1024)？

Surya 预训练是 `embed_dim=1280, img_size=4096`，在 24GB 显存上训练不够。pooling=4 将分辨率降到 1024×1024，token 数从 65536 降到 4096，显存占用从 ~40GB 降到 ~18GB。同时通过插值适配了位置编码和频谱门控权重。
