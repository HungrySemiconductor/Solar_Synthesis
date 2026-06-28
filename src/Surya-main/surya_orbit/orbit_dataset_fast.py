"""
OrbitDataset (fast) — 几何缩放训练数据集。

与 orbit_dataset.py 的区别:
  - 自动检测文件类型: .npy (预处理过) vs .nc (原始)
  - .npy: np.load() 直接读, ~50ms/样本
  - .nc:  全流程 (h5netcdf → signum-log → pool), ~2-3s/样本
  - 用 np.random.choice 替代 pd.sample(), 减少 Python 开销

用法:
  先运行 preprocess_nc.py → 生成 .npy 文件,
  然后用本文件替代 orbit_dataset.py (改 train_phase1.py 中的 import 即可).
"""

import json, re
import numpy as np
from pathlib import Path

import hdf5plugin  # NC 文件 blosc 压缩插件
import pandas as pd
import xarray as xr
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class OrbitDatasetFast(Dataset):
    """
    参数同 OrbitDataset, 新增自动检测 .npy / .nc。
    """

    def __init__(
        self,
        nc_files: list[str],
        orbit_csv_path: str,
        scalers: dict,
        channels: list[str],
        samples_per_file: int = 10,
        pooling: int = 2,
    ):
        super().__init__()
        self.nc_files = [str(Path(f)) for f in nc_files]
        self.channels = list(channels)
        self.samples_per_file = int(samples_per_file)
        self.pooling = int(pooling)
        self.scalers = scalers

        # ── 检测文件类型 ──
        first = self.nc_files[0]
        self._use_npy = first.endswith(".npy")
        self._re_npy = re.compile(r"__(\d+)km\.npy$")

        # 归一化参数 (.nc 模式才需要)
        if not self._use_npy:
            self._cache_normalization_params()

        # 轨道 CSV — 预转 numpy, 省去 pd.sample() 开销
        self.orbit_df = pd.read_csv(orbit_csv_path)
        if "distance_km" not in self.orbit_df.columns:
            raise ValueError("Orbit CSV missing column: distance_km")
        self._orbit_dist = self.orbit_df["distance_km"].values.astype(np.float64)

        self._log_init()

    # ================================================================
    # 初始化辅助
    # ================================================================

    def _cache_normalization_params(self):
        means_list, stds_list, eps_list, sl_list = [], [], [], []
        for ch in self.channels:
            s = self.scalers[ch]
            means_list.append(s.mean)
            stds_list.append(s.std)
            eps_list.append(s.epsilon)
            sl_list.append(s.sl_scale_factor)
        self._means = np.array(means_list, dtype=np.float32).reshape(-1, 1, 1)
        self._stds = np.array(stds_list, dtype=np.float32).reshape(-1, 1, 1)
        self._epsilons = np.array(eps_list, dtype=np.float32).reshape(-1, 1, 1)
        self._sl_scales = np.array(sl_list, dtype=np.float32).reshape(-1, 1, 1)

    def _log_init(self):
        d = self._orbit_dist
        mode = "NPY (preprocessed)" if self._use_npy else "NC (raw, slow)"
        print(f"[OrbitDatasetFast] Mode: {mode}")
        print(f"[OrbitDatasetFast] {len(self.nc_files)} files × "
              f"{self.samples_per_file} = {len(self)} samples/epoch")
        print(f"[OrbitDatasetFast] SPO distances: {d.min()/149597870.7:.2f}"
              f"–{d.max()/149597870.7:.2f} AU ({len(d)} rows)")

    # ================================================================
    # PyTorch Dataset 接口
    # ================================================================

    def __len__(self):
        return len(self.nc_files) * self.samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file

        # ── 1. 加载 SDO 图像 + 源距离 ──
        fp = self.nc_files[file_idx]
        if self._use_npy:
            image, source_km = self._load_npy(fp)
        else:
            image, source_km = self._load_nc(fp)

        # ── 2. 目标距离 (随机采样) ──
        target_km = float(np.random.choice(self._orbit_dist))

        # ── 3. 几何伪真值 ──
        zoom_scale = source_km / target_km
        gt = self._geometric_scale(image, zoom_scale)

        # ── 4. 构建 batch ──
        ts = np.stack([image, image], axis=1)  # (C, 2, H, W)

        return {
            "ts": ts.astype(np.float32, copy=False),
            "time_delta_input": np.array([-0.2, 0.0], dtype=np.float32),
            "source_distance_km": np.float32(source_km),
            "target_distance_km": np.float32(target_km),
            "forecast": gt[np.newaxis, ...].astype(np.float32, copy=False),
            "lead_time_delta": np.array([0.2], dtype=np.float32),
        }, {
            "filepath": fp,
            "source_distance_km": source_km,
            "target_distance_km": target_km,
            "zoom_scale": zoom_scale,
            "grid_scale": target_km / source_km,
        }

    # ================================================================
    # .npy 快速加载 (~50ms)
    # ================================================================

    def _load_npy(self, path: str) -> tuple[np.ndarray, float]:
        """np.load() 直接读取, 数据已归一化+已池化。"""
        image = np.load(path)  # (C, H, W) float32
        # 从文件名解析距离:  xxx__147097819km.npy
        m = self._re_npy.search(Path(path).name)
        distance_km = float(m.group(1)) if m else 149597870.7
        return image, distance_km

    # ================================================================
    # .nc 慢速加载 (~2-3s, 回退用)
    # ================================================================

    def _load_nc(self, filepath: str) -> tuple[np.ndarray, float]:
        with xr.open_dataset(filepath, engine="h5netcdf", chunks=None) as ds:
            data = ds[self.channels].to_array().load().to_numpy()
            distance_km = self._extract_dsun_obs(ds)

        data = data.astype(np.float32)
        data = data * self._sl_scales
        data = np.sign(data) * np.log1p(np.abs(data))
        data = (data - self._means) / (self._stds + self._epsilons)

        if self.pooling > 1:
            C, H, W = data.shape
            h, w = H // self.pooling, W // self.pooling
            data = data.reshape(C, h, self.pooling, w, self.pooling)
            data = data.mean(axis=(2, 4))

        return data.astype(np.float32, copy=False), distance_km

    def _extract_dsun_obs(self, ds: xr.Dataset) -> float:
        for ch_name in self.channels:
            if ch_name not in ds:
                continue
            var = ds[ch_name]
            for mk in ["meta_0", "meta_1"]:
                if mk not in var.attrs:
                    continue
                try:
                    meta = json.loads(var.attrs[mk])
                    if "dsun_obs" in meta:
                        return float(meta["dsun_obs"]) / 1000.0
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return 149597870.7

    # ================================================================
    # 几何伪真值
    # ================================================================

    def _geometric_scale(self, image: np.ndarray,
                         zoom_scale: float) -> np.ndarray:
        C, H, W = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(C):
            ch = image[c]
            scaled = zoom(ch, zoom=zoom_scale, order=3)
            sh, sw = scaled.shape
            if zoom_scale >= 1.0:
                sy, sx = (sh - H) // 2, (sw - W) // 2
                result[c] = scaled[sy:sy + H, sx:sx + W]
            else:
                sy, sx = (H - sh) // 2, (W - sw) // 2
                result[c, sy:sy + sh, sx:sx + sw] = scaled
        return result
