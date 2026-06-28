# Phase 1：几何缩放训练 — 完整实施方案

> **目标**：让 Surya 学会根据观测距离变化，生成太阳视直径不同的图像。
>
> **原则**：不修改 Surya 任何现有代码，所有新增代码放在独立目录 `surya_orbit/` 中。
>
> **硬件**：矩尺云 RTX 4090 (24GB)

---

## 一、关键前提：先澄清三个"不需要担心"的问题

### 1.1 时间不匹配不是问题

```
SDO 数据时间：2014 年 10 月 23 日（地球轨道，约 1 AU）
SPO 轨道时间：2029 年 ~ 2042 年（0.83 AU ~ 5.13 AU）
```

**这两者的时间差了 15~28 年，但完全不需要匹配。**

原因：Phase 1 只训练几何缩放。

```
几何缩放公式：scale_factor = r_src / r_tgt

  其中 r_src = 地球轨道距离 ≈ 1 AU（和 SDO 图像绑定）
       r_tgt = SPO 轨道距离（和 SDO 图像的时间无关）

  一张 2014 年的太阳图 + 2035 年 SPO 距离 3.2 AU
  → 伪真值 = zoom(2014年图, 1/3.2 ≈ 0.31)
  → 这就是从 3.2 AU 处看到的太阳应有的表观大小
```

几何缩放是一个**纯数学运算**——它不关心中间隔了多少年。所以：

> **任意时间的 SDO 图像 + 任意时间的 SPO 距离 = 一个有效的训练样本**

### 1.2 SPO 数据非常充足

当前 SPO 轨道参数文件 `spo_orbit_param_valid.csv` 的实际规模：

```
Total rows:       131,603
Unique distances: 131,220（几乎每行都不一样）
Distance range:   0.83 AU ~ 5.13 AU

距离分布:
  0.80-0.90 AU:  1,284 行
  0.95-1.02 AU:  7,877 行（地球轨道附近）
  1.10-1.50 AU: 25,289 行
  1.50-2.00 AU: 30,271 行
  2.00-2.50 AU: 20,799 行
  2.50-3.00 AU:  9,599 行
  3.00-3.50 AU: 10,559 行
  3.50-5.13 AU: 18,925 行（最远）
```

每个距离区间都有大量采样。训练时随机抽取，样本的距离跨度可以覆盖整个 0.83~5.13 AU 范围。

### 1.3 训练样本数量足够

```
24 个 SDO 文件 × 随机采样 SPO 行 = 无限不重复的训练流

每个 epoch 可以产生任意数量的训练样本：
  - 如果 epoch = 遍历 24 个 SDO 文件各 1 次，每个配 1 个随机 SPO → 24 个样本
  - 如果每个 SDO 配 N 个随机 SPO → 24 × N 个样本
  - 推荐：每个 SDO 配 10 个随机 SPO → 240 样本/epoch
  - 20 epochs → 4800 次参数更新
```

---

## 二、整体架构

### 2.1 训练目标

```
输入:  SDO 图像 (1 AU 处的太阳, 13 通道全波段)
      + 源轨道参数 (地球位置 ~1 AU)
      + 目标轨道参数 (SPO 位置, 随机采样于 0.83~5.13 AU)

输出:  从 SPO 位置看到的太阳图像 (13 通道)

监督:  用纯几何缩放 (scipy.ndimage.zoom) 生成的"伪真值"
```

### 2.2 为什么需要训练？

几何缩放是纯数学运算（`scipy.ndimage.zoom`），但为什么还要训练模型而非直接用 zoom？

| 训练谁 | 目的 |
|--------|------|
| `OrbitAwareFlowModel` | 学会从轨道参数自动计算缩放因子，集成到 Surya 的 forward 流水线中 |
| `Decoder` | 学会解码"任意距离下太阳的 token"——它原来只解码过 1 AU 的 token |

**Decoder 尤其重要**：Surya 的 Decoder 在预训练时只见过"1 AU 的太阳特征"，现在需要它学会"输入 token 代表的太阳可能大也可能小，但我都能正确还原为像素"。这需要在缩放过的数据上重新训练 Decoder。

### 2.3 数据流（以 pooling=2, img_size=2048 为例）

```
SDO NC 文件 (2014-10-23, 13通道, 4096×4096)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  OrbitDataset.__getitem__()                                  │
│                                                              │
│  1. 加载 SDO 图像 → (13, 4096, 4096)                         │
│  2. Signum-log 归一化（使用 Surya 的 scalers）                 │
│  3. 池化 → (13, 2048, 2048)                                  │
│  4. 源轨道 = 地球近似位置 (~1 AU)                              │
│            归一化：pos/1AU, vel/29.8 km/s                     │
│  5. 目标轨道 = 从 SPO CSV 随机采样 (0.83~5.13 AU 中的任意值)   │
│  6. 几何缩放伪真值 = zoom(SDO图, r_src / r_tgt)               │
│     → (13, 2048, 2048)                                       │
│  7. 构建时间序列：同一张图复制 2 帧                             │
│     → ts = (13, 2, 2048, 2048)                               │
│                                                              │
│  返回 batch 字典:                                             │
│  {                                                           │
│    "ts":                  (13, 2, 2048, 2048) ← 输入          │
│    "time_delta_input":    (2,)                ← 时间差        │
│    "source_orbit_params": (6,)                ← 源轨道        │
│    "target_orbit_params": (6,)                ← 目标轨道      │
│    "forecast":            (13, 1, 2048, 2048) ← 伪真值        │
│    "lead_time_delta":     (1,)                ← 预测时间差    │
│  }                                                           │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼ DataLoader collate 添加 B 维度
                     │
               batch["ts"] = (B, 13, 2, 2048, 2048)
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  OrbitHelioSpectFormer.forward(batch)                        │
│                                                              │
│                                                              │
│  第 0 步：提取轨道编码（Phase 2 用，Phase 1 预留接口）        │
│    orbit_embed = self.orbit_encoder(target_orbit_params)     │
│    → (B, 1, 768)                                             │
│                                                              │
│                                                              │
│  ① OrbitAwareFlowModel (可训练, ~5K 参数)                     │
│     batch["ts"] → 取最后一帧 x = ts[:,:,-1,:,:]              │
│     → x: (B, 13, 2048, 2048)                                 │
│     source_orbit_params → r_src                               │
│     target_orbit_params → r_tgt                               │
│     base_scale = r_src / r_tgt                                │
│     correction = MLP(target_orbit)  [可学习的微调]             │
│     scale = base_scale * (1 + 0.1·tanh(correction))          │
│     scaled_grid = self.grid * scale                           │
│     flow_field = scaled_grid + flow_generator(higher_modes)  │
│     y_hat_flow = F.grid_sample(x, flow_field)                │
│     → (B, 13, 2048, 2048)                                    │
│                                                              │
│     然后与原始 ts 拼接:                                        │
│     x = torch.cat([ts, y_hat_flow.unsqueeze(2)], dim=2)     │
│     → (B, 13, 3, 2048, 2048)  ← T 从 2 变为 3               │
│                                                              │
│                                                              │
│  ② LinearEmbedding (冻结, ~100M)                             │
│     x → PatchEmbed3D → tokens                                │
│     (B, 13, 3, 2048, 2048) → (B, 16384, 768)                │
│     [2048/16=128, 128×128=16384 tokens]                      │
│                                                              │
│                                                              │
│  ③ SpectFormer Backbone (冻结, ~250M)                        │
│     (B, 16384, 768) → 10 层处理 → (B, 16384, 768)            │
│     shape 不变，但 token 内容被精炼                            │
│                                                              │
│                                                              │
│  ④ LinearDecoder (可训练, ~16M)                               │
│     (B, 16384, 768) → LinearDecoder → (B, 13, 2048, 2048)   │
│                                                              │
│                                                              │
│  ⑤ 残差连接:                                                  │
│     output = ④ + ① (y_hat_flow)                              │
│     → (B, 13, 2048, 2048)                                    │
│     残差连接让模型只需学习"flow 没处理好的部分"                 │
│                                                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
         pred = (B, 13, 2048, 2048)
         target = batch["forecast"][:, :, 0, :, :]
                = (B, 13, 2048, 2048)

         Loss = MSE(pred, target)
```

### 2.4 参数分布

| 模块 | 参数量 | Phase 1 状态 | 原因 |
|------|--------|-------------|------|
| OrbitAwareFlowModel | ~5,000 | ✅ 训练 | 学习几何变换 |
| LinearEmbedding | ~100M | ❄️ 冻结 | 已学会提取太阳特征，无需修改 |
| SpectFormer Backbone | ~250M | ❄️ 冻结 | 已学会理解太阳物理，Phase 2 再微调 |
| LinearDecoder | ~16M | ✅ 训练 | 学习解码变换后的 token |

**可训练参数约 16M，占总参数 366M 的 4.4%**。

---

## 三、新建文件清单

所有新代码放在 `Surya-main/surya_orbit/` 目录下，**不修改 Surya 任何现有文件**。

```
surya_orbit/
├── __init__.py                  # 包初始化
├── orbit_flow.py                # ① OrbitAwareFlowModel 类
├── orbit_spectformer.py         # ② OrbitHelioSpectFormer 类
├── orbit_dataset.py             # ③ OrbitDataset 类
├── train_phase1.py              # ④ 训练脚本
└── config_phase1.yaml           # ⑤ 训练配置
```

---

## 四、每个文件详解

### 4.1 `orbit_flow.py` — OrbitAwareFlowModel

**继承自**：`surya.models.flow.HelioFlowModel`

**核心改动**：原来的 `HelioFlowModel` 使用固定的 flow field（不依赖输入内容，不依赖轨道参数）。我们改为**从轨道参数计算距离比，生成对应的缩放 flow field**。

```python
class OrbitAwareFlowModel(HelioFlowModel):
    """
    与父类 HelioFlowModel 的区别：
    - 父类：flow_field = fixed_grid + small_learned_offset
    - 本类：flow_field = distance_scaled_grid + small_learned_offset
      distance_scaled_grid = fixed_grid × (r_src / r_tgt)
    """

    def __init__(self, img_size=(2048, 2048)):
        super().__init__(img_size)
        # 小 MLP：6D 轨道参数 → 1 个缩放修正因子
        # 这个修正因子在 ±10% 范围内（受 tanh 约束）
        self.scale_corrector = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, batch):
        # 1. 取最后一帧输入图像
        x = batch["ts"]                         # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = x[:, :, -1, :, :]                  # (B, C, H, W)

        # 2. 提取距离
        src = batch["source_orbit_params"]      # (B, 6)
        tgt = batch["target_orbit_params"]      # (B, 6)
        # 源轨道 = 地球位置（约 1 AU），目标轨道 = 随机采样的 SPO 位置
        r_src = torch.norm(src[:, :3], dim=1)   # (B,)
        r_tgt = torch.norm(tgt[:, :3], dim=1)   # (B,)

        # 3. 缩放因子 = 基础几何 × 可学习修正
        base_scale = r_src / r_tgt              # (B,)  纯几何
        correction = self.scale_corrector(tgt).squeeze(-1)  # (B,)
        scale = base_scale * (1.0 + 0.1 * torch.tanh(correction))
        # tanh 输出 ∈ [-1, 1]，所以修正 ∈ [-10%, +10%]

        # 4. 缩放 grid + 采样
        # self.grid 来自父类：shape (1, H, W, 2)，每个像素的归一化 (u,v) 坐标
        # scale > 1 → grid 收缩 → 太阳表观尺寸变小（目标比源远）
        # scale < 1 → grid 扩张 → 太阳表观尺寸变大（目标比源近）
        scaled_grid = self.grid * scale.view(B, 1, 1, 1)  # (B, H, W, 2)

        # 父类的 flow_generator 提供微小的局部修正（如边缘处理）
        flow_field = scaled_grid + self.flow_generator(self.higher_modes)
        # (B, H, W, 2)

        y_hat = F.grid_sample(
            x, flow_field,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return y_hat  # (B, C, H, W)
```

**设计要点**：

- **`base_scale` 是确定性的**：纯几何距离比，保证即使 MLP 完全未训练（输出 ≈ 0），缩放也是正确的
- **`correction` 是可学习的小幅微调**：±10% 范围，用于处理几何缩放无法完美覆盖的边缘效应
- **`flow_generator` 来自父类**：提供微小的非均匀 flow 修正

---

### 4.2 `orbit_spectformer.py` — OrbitHelioSpectFormer

**继承自**：`surya.models.helio_spectformer.HelioSpectFormer`

**改动量极小**：只需在 `__init__` 中替换 FlowModel 类型，forward 几乎不变。

```python
class OrbitHelioSpectFormer(HelioSpectFormer):
    """
    与父类的唯一区别：learned_flow_model 使用 OrbitAwareFlowModel
    而非原始的 HelioFlowModel。

    父类的 forward() 已经通过 learned_flow 路径支持额外输入，
    包括:
    - 调用 self.learned_flow_model(batch)
    - 将 y_hat_flow 拼接到 ts 中
    - 残差连接 forecast_hat = unembed(tokens) + y_hat_flow

    这些逻辑完全不需要修改。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 父类在 __init__ 中已经创建了 self.learned_flow_model
        # 但创建的是 HelioFlowModel。
        # 我们在调用 super().__init__() 之后替换它：
        if self.learned_flow:
            img_size = kwargs["img_size"]
            self.learned_flow_model = OrbitAwareFlowModel(
                img_size=(img_size, img_size),
            )
```

**具体来说，父类 `HelioSpectFormer.__init__`（`helio_spectformer.py:91-95`）中的这段代码**：

```python
# helio_spectformer.py:91-95 (原始代码)
if learned_flow:
    self.learned_flow_model = HelioFlowModel(
        img_size=(img_size, img_size),
        use_latitude_in_learned_flow=use_latitude_in_learned_flow,
    )
```

**会被 `OrbitHelioSpectFormer` 在 `super().__init__()` 调用后覆盖**。所有其他代码（Embedding、Backbone、Decoder、forward 逻辑）完全不变。

**我们只需要确保 batch 字典中包含 `source_orbit_params` 和 `target_orbit_params`** 这两个键，OrbitAwareFlowModel 内部自己去读。

---

### 4.3 `orbit_dataset.py` — OrbitDataset

**不继承** Surya 的 `HelioNetCDFDataset`——数据结构完全不同。

**核心设计**：随机配对。每个 `__getitem__` 调用：

1. 取一个 SDO 文件（提供 1 AU 的太阳图像）
2. 随机取一个 SPO 轨道行（提供目标距离）
3. 几何缩放生成伪真值

```python
class OrbitDataset(Dataset):
    """
    训练样本构造方式：

    每个 SDO 文件 = 一张 1 AU 处的太阳图
    每个 SPO 行 = 一个目标轨道位置（0.83~5.13 AU）

    随机配对 → 无限不重复的训练样本
    """

    def __init__(
        self,
        nc_files: list[str],           # SDO NC 文件路径列表 (如 24 个)
        orbit_csv_path: str,           # SPO 轨道 CSV (131K 行)
        scalers: dict,                 # Signum-log 归一化参数
        channels: list[str],           # 13 通道名称列表
        samples_per_file: int = 10,    # 每个 SDO 文件配多少个 SPO 样本
        pooling: int = 2,              # 池化因子 (4096/pooling = img_size)
    ):
        self.nc_files = nc_files
        self.orbit_df = pd.read_csv(orbit_csv_path)
        self.scalers = scalers
        self.channels = channels
        self.samples_per_file = samples_per_file
        self.pooling = pooling

        # 归一化常数
        self.AU_KM = 149597870.7       # 1 AU in km
        self.EARTH_VEL = 29.8          # 地球轨道速度 km/s

    def __len__(self):
        # 每个 SDO 文件配 N 个随机 SPO = 总共 len × N 个样本
        return len(self.nc_files) * self.samples_per_file

    def __getitem__(self, idx):
        # ── 1. 确定是哪个 SDO 文件 ──
        file_idx = idx // self.samples_per_file
        filepath = self.nc_files[file_idx]

        # ── 2. 加载并变换 SDO 图像 ──
        image = self._load_sdo_image(filepath)
        # image: (13, 4096, 4096) float32, 已做 signum-log 归一化

        # ── 3. 池化降低分辨率 ──
        if self.pooling > 1:
            image = block_reduce(
                image,
                block_size=(1, self.pooling, self.pooling),
                func=np.mean,
            )
        # image: (13, 2048, 2048)

        # ── 4. 源轨道参数（SDO/地球 ≈ 1 AU）──
        # 简化处理：近似为地球在 y 轴方向 1 AU 处
        source_orbit = np.array(
            [0.0, 1.0, 0.0,   # 位置 (AU)
             0.0, 0.0, 0.0],  # 速度 (归一化)
            dtype=np.float32
        )

        # ── 5. 目标轨道参数（从 SPO CSV 随机采样）──
        row = self.orbit_df.sample(1).iloc[0]
        target_pos_km = np.array([row["x_km"], row["y_km"], row["z_km"]],
                                 dtype=np.float64)
        target_vel_kms = np.array([row["vx_kms"], row["vy_kms"],
                                    row["vz_kms"]], dtype=np.float64)
        target_orbit = np.concatenate([
            (target_pos_km / self.AU_KM).astype(np.float32),
            (target_vel_kms / self.EARTH_VEL).astype(np.float32),
        ])
        # target_orbit: (6,) float32

        # ── 6. 生成几何伪真值 ──
        r_src = self.AU_KM                    # 1 AU (km)
        r_tgt = float(np.linalg.norm(target_pos_km))  # SPO 距离 (km)
        scale_factor = r_src / r_tgt           # 1 AU / SPO距离

        gt_image = self._geometric_scale(image, scale_factor)
        # gt_image: (13, 2048, 2048)

        # ── 7. 构建时间序列 ──
        # Surya 需要 (C, T, H, W) 格式，T ≥ 2
        # Phase 1 不依赖时序信息，同一张图复制两次
        ts = np.stack([image, image], axis=1)  # (13, 2, H, W)

        # 目标格式：(C, 1, H, W) — Surya 期望的 forecast 格式
        forecast = gt_image[np.newaxis, :, :, :]  # (13, 1, H, W)

        return {
            "ts": ts.astype(np.float32),
            "time_delta_input": np.array([-0.2, 0.0], dtype=np.float32),
            "source_orbit_params": source_orbit,
            "target_orbit_params": target_orbit,
            "forecast": forecast.astype(np.float32),
            "lead_time_delta": np.array([0.2], dtype=np.float32),
        }, {
            "filepath": filepath,
            "scale_factor": scale_factor,
            "r_src_km": r_src,
            "r_tgt_km": r_tgt,
        }

    def _load_sdo_image(self, filepath):
        """加载 SDO NC 文件并做 signum-log 归一化"""
        with xr.open_dataset(filepath, engine="h5netcdf") as ds:
            data = ds[self.channels].to_array().load().to_numpy()
        # data: (13, 4096, 4096)

        # Signum-log 归一化（与 Surya 预训练一致）
        means, stds, epsilons, sl_scale_factors = self._normalization_params()
        data = data * sl_scale_factors.reshape(-1, 1, 1)
        data = np.sign(data) * np.log1p(np.abs(data))
        data = (data - means.reshape(-1, 1, 1)) / (
            stds.reshape(-1, 1, 1) + epsilons.reshape(-1, 1, 1)
        )
        return data.astype(np.float32)

    def _normalization_params(self):
        """缓存归一化参数（与 Surya helio.py 的 transformation_inputs 一致）"""
        if not hasattr(self, "_cached_params"):
            means = np.array([self.scalers[ch].mean for ch in self.channels],
                             dtype=np.float32)
            stds = np.array([self.scalers[ch].std for ch in self.channels],
                            dtype=np.float32)
            epsilons = np.array([self.scalers[ch].epsilon for ch in self.channels],
                                dtype=np.float32)
            sl_scale_factors = np.array(
                [self.scalers[ch].sl_scale_factor for ch in self.channels],
                dtype=np.float32
            )
            self._cached_params = (means, stds, epsilons, sl_scale_factors)
        return self._cached_params

    def _geometric_scale(self, image, scale_factor):
        """
        纯几何缩放，生成伪真值。

        复用你 single_scale.py 中的 scipy.ndimage.zoom 逻辑。
        对 13 个通道分别缩放。
        """
        C, H, W = image.shape
        result = np.zeros_like(image)

        for c in range(C):
            ch = image[c]
            scaled = zoom(ch, zoom=scale_factor, order=3)

            sh, sw = scaled.shape
            if scale_factor >= 1.0:
                # 放大 → 裁剪中心
                start_y = (sh - H) // 2
                start_x = (sw - W) // 2
                result[c] = scaled[start_y:start_y+H, start_x:start_x+W]
            else:
                # 缩小 → 放在中心，周围补 0
                start_y = (H - sh) // 2
                start_x = (W - sw) // 2
                result[c, start_y:start_y+sh, start_x:start_x+sw] = scaled

        return result
```

**关于 `__len__` 和训练样本数量**：

```python
# 示例：24 个 SDO 文件，samples_per_file=10
dataset = OrbitDataset(nc_files=glob("data/SDO_20141023/*.nc"),  # 24 个文件
                       samples_per_file=10)

len(dataset)  # = 240

# DataLoader 每次取 batch_size=2，shuffle=True
# 每个 epoch = 240/2 = 120 次参数更新
# 20 epochs = 2400 次参数更新

# 每个样本的目标距离是随机采样的（131K 个候选），
# 所以 2400 次更新中几乎不会重复使用相同的 (SDO图, SPO距离) 组合
```

---

### 4.4 `train_phase1.py` — 训练脚本

**结构参考**：`downstream_examples/ar_segmentation/finetune.py`

```python
def main(config):
    # ========================================
    # 初始化
    # ========================================
    device = torch.device("cuda")
    scalers = build_scalers(config["data"]["scalers_path"])

    # ========================================
    # 数据集
    # ========================================
    nc_files = sorted(Path(config["data"]["sdo_data_dir"]).glob("*.nc"))
    print(f"Found {len(nc_files)} SDO files")

    dataset = OrbitDataset(
        nc_files=[str(f) for f in nc_files],
        orbit_csv_path=config["data"]["orbit_csv"],
        scalers=scalers,
        channels=config["data"]["sdo_channels"],
        samples_per_file=config["data"]["samples_per_file"],
        pooling=config["data"]["pooling"],
    )
    print(f"Dataset size: {len(dataset)} samples "
          f"({len(nc_files)} files × {config['data']['samples_per_file']})")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # ========================================
    # 模型
    # ========================================
    model = OrbitHelioSpectFormer(
        img_size=config["data"]["img_size_after_pool"],
        patch_size=config["model"]["patch_size"],
        in_chans=len(config["data"]["sdo_channels"]),
        embed_dim=config["model"]["embed_dim"],
        time_embedding={
            "type": "linear",
            "time_dim": config["model"]["time_dim"],
        },
        depth=config["model"]["depth"],
        n_spectral_blocks=config["model"]["n_spectral_blocks"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        drop_rate=0.0,
        window_size=config["model"]["window_size"],
        dp_rank=config["model"]["dp_rank"],
        learned_flow=True,
        finetune=False,
        nglo=0,
    )

    # ── 加载预训练权重 ──
    weights_path = Path(config["model"]["sdo_model_repo"]) / \
                   config["model"]["pretrained_weights"]
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)

    # strict=False: OrbitAwareFlowModel 的参数不在预训练权重中
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"Missing keys (expected - new FlowModel params): {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    # ── 冻结/解冻 ──
    for param in model.parameters():
        param.requires_grad = False

    # 解冻 Decoder
    for param in model.unembed.parameters():
        param.requires_grad = True

    # 解冻 OrbitAwareFlowModel
    for param in model.learned_flow_model.parameters():
        param.requires_grad = True

    # 统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable/1e6:.2f}M trainable / "
          f"{total/1e6:.2f}M total ({100*trainable/total:.1f}%)")

    model.to(device)

    # ========================================
    # 优化器
    # ========================================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
    )

    # ========================================
    # 训练循环
    # ========================================
    model.train()
    for epoch in range(config["training"]["epochs"]):
        epoch_loss = 0.0
        for batch_idx, (batch, meta) in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(batch)                             # (B, 13, H, W)
                target = batch["forecast"][:, :, 0, :, :]       # (B, 13, H, W)
                loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % config["training"]["log_interval"] == 0:
                print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d} | "
                      f"Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:3d} complete | Avg Loss: {avg_loss:.6f}")

        # ── 保存 checkpoint ──
        if (epoch + 1) % config["training"]["save_every"] == 0:
            ckpt_path = Path(config["output"]["checkpoint_dir"]) / \
                        f"phase1_epoch{epoch+1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")

    print("Phase 1 training complete.")
```

---

### 4.5 `config_phase1.yaml` — 训练配置

```yaml
# ============================================================
# Phase 1: 几何缩放训练配置
# ============================================================

training:
  epochs: 20
  batch_size: 2
  learning_rate: 0.0001        # 1e-4
  num_workers: 2
  log_interval: 10             # 每 10 个 batch 打印一次 loss
  save_every: 5                # 每 5 个 epoch 保存一次

data:
  sdo_data_dir: "data/SDO_20141023"
  orbit_csv: "data/spo_orbit_param_valid.csv"
  scalers_path: "data/Surya-1.0/scalers.yaml"
  sdo_channels:                # Surya 的 13 个全波段
    - aia094
    - aia131
    - aia171
    - aia193
    - aia211
    - aia304
    - aia335
    - aia1600
    - hmi_m
    - hmi_bx
    - hmi_by
    - hmi_bz
    - hmi_v
  pooling: 2                   # 4096 → 2048
  img_size_after_pool: 2048
  samples_per_file: 10         # 每个 SDO 文件配 10 个随机 SPO 距离

model:
  sdo_model_repo: "data/Surya-1.0"
  pretrained_weights: "surya.366m.v1.pt"
  # 以下参数与 Surya 预训练一致
  patch_size: 16
  embed_dim: 768
  depth: 10
  n_spectral_blocks: 2
  num_heads: 12
  mlp_ratio: 4.0
  window_size: 2
  dp_rank: 2
  time_dim: 2

output:
  checkpoint_dir: "checkpoints/phase1"
```

---

## 五、在云服务器上的操作步骤

### Step 0：确认环境（2 分钟）

```bash
ssh your-server
cd Surya-main

nvidia-smi                          # 确认 RTX 4090 可见
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
ls data/Surya-1.0/                  # 确认有权重文件
# 应看到: config.yaml  scalers.yaml  surya.366m.v1.pt
```

### Step 1：下载 SDO 数据（约 10 分钟）

```bash
# 在云服务器上 Surya-main 目录下执行
python -c "
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

out_dir = Path('data/SDO_20141023')
out_dir.mkdir(parents=True, exist_ok=True)
bucket = 'nasa-surya-bench'
t = datetime(2014, 10, 23, 0, 0)

for h in range(24):
    ts = t + timedelta(hours=h)
    fname = ts.strftime('%Y%m%d_%H%M.nc')
    s3_path = f's3://{bucket}/2014/10/{fname}'
    local_path = out_dir / fname
    if not local_path.exists():
        subprocess.run(['aws', 's3', 'cp', s3_path, str(local_path),
                       '--no-sign-request', '--only-show-errors'],
                       check=True)
        print(f'Downloaded: {fname}')
    else:
        print(f'Skipped: {fname} (exists)')

print(f'Done. {len(list(out_dir.glob(\"*.nc\")))} files downloaded.')
"
```

### Step 2：上传轨道 CSV（1 分钟）

```bash
# 在本地 Windows 终端执行：
scp "D:/Solar_Images/Solar_Scaler/data/Outputs/CSV/spo_orbit_param_valid.csv" \
    your-server:Surya-main/data/

# 在云服务器上确认：
head -3 data/spo_orbit_param_valid.csv
# 应看到: timestamp,x_km,y_km,z_km,vx_kms,vy_kms,vz_kms,distance_km
wc -l data/spo_orbit_param_valid.csv
# 应输出: 131604
```

### Step 3：上传 surya_orbit 代码（1 分钟）

```bash
# 在云服务器上创建目录：
cd Surya-main
mkdir -p surya_orbit

# 在本地 Windows 终端上传代码：
scp -r "D:/Solar_Images/Surya-main/surya_orbit/"* \
    your-server:Surya-main/surya_orbit/
```

### Step 4：开始训练

```bash
cd Surya-main
python surya_orbit/train_phase1.py --config surya_orbit/config_phase1.yaml
```

---

## 六、预期结果

### 6.1 训练日志

```
Found 24 SDO files
Dataset size: 240 samples (24 files × 10)
Missing keys (expected - new FlowModel params): 8
Unexpected keys: 0
Parameters: 16.01M trainable / 366.19M total (4.4%)

  Epoch   0 | Batch    0 | Loss: 0.087432
  Epoch   0 | Batch   10 | Loss: 0.065127
  ...
Epoch   0 complete | Avg Loss: 0.071234

  Epoch   1 | Batch    0 | Loss: 0.048291
  ...
Epoch   4 complete | Avg Loss: 0.015423
  Saved: checkpoints/phase1/phase1_epoch5.pt

...

Epoch  19 complete | Avg Loss: 0.002187
  Saved: checkpoints/phase1/phase1_epoch20.pt

Phase 1 training complete.
```

Loss 从 ~0.08 下降到 ~0.002（signum-log 归一化空间）。

### 6.2 验证：推理测试

```python
# 在云服务器上快速测试训练好的模型
import torch
from surya_orbit.orbit_spectformer import OrbitHelioSpectFormer
# ... (加载模型和一张测试图像) ...

model.eval()
with torch.no_grad():
    for dist_au in [0.83, 1.0, 1.5, 2.5, 3.2]:
        batch = make_test_batch(sdo_image, dist_au)
        pred = model(batch)
        # pred 中的太阳圆面应随 dist_au 增大而变小
```

---

## 七、潜在问题和解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| OOM (显存不足) | 2048×2048 对于 24GB 显存仍有压力 | pooling 改为 4 (1024×1024)，或 batch_size=1 |
| Loss 不下降 | 学习率不当 | 先试 1e-3，不行再试 1e-5 |
| `strict=False` 警告 missing keys | OrbitAwareFlowModel 参数不在预训练权重中 | **正常现象**，这些参数是随机初始化的，会被训练 |
| 输出图像模糊 | Decoder 刚开始学习处理缩放后的 token | 增加 epochs，前几个 epoch 模糊是正常的 |
| 缩放方向反了 | `r_src/r_tgt` 顺序问题 | 验证：r_tgt=3.2 AU 时太阳应变小（scale≈0.31）|

---

## 八、Phase 1 完成后通向 Phase 2

```
Phase 1 完成标志:
  ✅ OrbitAwareFlowModel 正确进行几何缩放
  ✅ Decoder 能正确还原缩放后的图像
  ✅ Loss 稳定收敛到 < 0.005

加载 Phase 1 checkpoint → Phase 2 训练:

  1. 加载 phase1_epoch20.pt
  2. OrbitAwareFlowModel: 冻结（几何能力保留）
  3. 解冻 adaLN 调制层（BlockAttention 中的 adaLN_modulation）
  4. 添加 LoRA 到 Attention 权重（qkv, proj, fc1, fc2）
  5. 新增物理约束损失:
     L_total = 0.5*L_geo + 0.1*L_flux + 0.05*L_limb + 0.1*L_struct
  6. 更低学习率 (5e-5) 继续训练
```

---

## 九、时间估算

| 步骤 | 时间 |
|------|------|
| 环境确认 + 下载 24 个 SDO 文件 | ~15 分钟 |
| 上传 CSV + 上传代码 | ~2 分钟 |
| 首次运行 (验证不报错) | ~5 分钟 |
| Phase 1 完整训练 (20 epochs × 120 batches) | ~2-4 小时 |
| 验证结果 | ~30 分钟 |
