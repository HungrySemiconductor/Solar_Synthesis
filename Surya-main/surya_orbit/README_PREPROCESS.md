# NC 预处理说明

## 为什么需要预处理

训练时 GPU 利用率 0%、CPU 200%，是因为 DataLoader 每个样本都要做：

```
NC 文件读取 (h5netcdf) → signum-log 归一化 → 均值池化 → scipy zoom
        ↑                        ↑              ↑            ↑
      ~1.5 秒                  ~0.3 秒        ~0.2 秒      ~0.3 秒
                                       总计: ~2-3 秒/样本
```

GPU 干 0.1 秒就等 2 秒，利用率不到 5%。

## 预处理做了什么

把所有 NC 文件**一次性**转换为 `.npy`：

```
NC 文件 → signum-log → pool → 保存 .npy
                               ↑
                        只跑一次，存到磁盘
```

之后 Dataset 直接 `np.load()` .npy（~0.05 秒），GPU 利用率从 5% 升到 ~80%。

## 文件大小

| pooling | 输出尺寸 | 每文件大小 | 23 个文件 | 72 个文件 |
|---------|----------|-----------|----------|----------|
| 1 | 4096×4096 | ~800 MB | ~18 GB | ~56 GB |
| 2 | 2048×2048 | ~200 MB | ~4.6 GB | ~14 GB |
| 4 | 1024×1024 | ~52 MB | ~1.2 GB | ~3.7 GB |

**Phase 1 推荐 pooling=4**（与训练 config 一致）。

## 用法

### Step 1：预处理

```bash
cd /root/Surya-main

python surya_orbit/preprocess_nc.py \
    --nc-dir data/Surya-1.0_validation_data \
    --npy-dir data/Surya-1.0_validation_npy \
    --pooling 4
```

输出：

```
NC dir:    data/Surya-1.0_validation_data
NPY dir:   data/Surya-1.0_validation_npy
Files:     23
Pooling:   4 (4096 → 1024)

  [  1/23] 20140107_1500.nc  → 20140107_1500__147097819km.npy  (0.983 AU, 1024×1024, 3.2s)
  [  2/23] 20140107_1512.nc  → 20140107_1512__147097818km.npy  (0.983 AU, 1024×1024, 3.1s)
  ...
  [ 23/23] 20140107_1924.nc  → 20140107_1924__147097820km.npy  (0.983 AU, 1024×1024, 3.0s)

Done.
  Files: 23 .npy  (1196 MB total)
  Manifest: data/Surya-1.0_validation_npy/manifest.csv
```

23 个文件约耗时 70 秒。

### Step 2：切换训练脚本使用快速 Dataset

预处理后生成的是 `.npy` 文件，原来的 `orbit_dataset.py` 不识别 `.npy`。
需手动将 `train_phase1.py` 的 import 从原始版改为快速版。

在 `train_phase1.py` 中找到这行：

```python
from surya_orbit.orbit_dataset import OrbitDataset
```

替换为：

```python
from surya_orbit.orbit_dataset_fast import OrbitDatasetFast as OrbitDataset
```

不要删除 `orbit_dataset.py`（留作 .nc 回退）。改完后训练命令不变。

### Step 3：训练

```bash
python surya_orbit/train_phase1.py --config surya_orbit/config_phase1.yaml
```

## 输出文件结构

```
data/Surya-1.0_validation_npy/
├── manifest.csv                          ← 索引文件
│   npy_file, source_nc, source_distance_km
├── 20140107_1500__147097819km.npy        ← (13, 1024, 1024) float32
├── 20140107_1512__147097818km.npy
├── ...
└── 20140107_1924__147097820km.npy
```

文件名格式：`{原NC名}__{dsun_obs距离}km.npy`，距离直接从文件名解析，不需要再读 NC 元数据。

## 清理

预处理完成后可以删除原始 NC 文件以节省空间：

```bash
# 可选：删除原始 NC 文件（~5 GB）
rm -rf data/Surya-1.0_validation_data/

# 保留 .npy 目录（~1.2 GB）
ls data/Surya-1.0_validation_npy/
```
