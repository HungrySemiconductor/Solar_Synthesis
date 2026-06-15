"""
OrbitDataset — 几何缩放训练数据集。

每个样本 = (SDO 图像 + 其对日直线距离) + (SPO 轨道距离)。
缩放完全基于对日直线距离比，与任何地理坐标系无关。

距离来源：
  - 源距离 (source_distance_km): 从 NC 文件通道变量的 meta_0/meta_1 中
    提取 dsun_obs 字段（米），转为公里。与 single_scale.py 逻辑一致。
  - 目标距离 (target_distance_km): 从 SPO CSV 的 distance_km 列直接读取。

几何伪真值生成：
  zoom_scale = source_distance_km / target_distance_km
  用 scipy.ndimage.zoom 对每个通道独立缩放。
"""

import json
import numpy as np
from pathlib import Path

import hdf5plugin  # 必须：NC 文件使用 blosc 压缩，h5py 需要此插件才能读取
import pandas as pd
import xarray as xr
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class OrbitDataset(Dataset):
    """
    参数:
        nc_files:           SDO NetCDF 文件路径列表
        orbit_csv_path:     SPO 轨道参数 CSV 路径
        scalers:            Surya 的 StandardScaler 字典 {channel_name: scaler}
        channels:           通道名称列表（13 个 SDO 通道）
        samples_per_file:   每个 SDO 文件配多少个随机 SPO 距离
        pooling:            池化因子 (4096 // pooling = 输出分辨率)
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

        # 加载轨道 CSV
        self.orbit_df = pd.read_csv(orbit_csv_path)
        self._validate_orbit_df()

        # 缓存归一化参数
        self.scalers = scalers
        self._cache_normalization_params()

        self._log_init()

    # ================================================================
    # 初始化辅助方法
    # ================================================================

    def _validate_orbit_df(self):
        required = {"distance_km"}
        missing = required - set(self.orbit_df.columns)
        if missing:
            raise ValueError(f"Orbit CSV missing columns: {missing}")

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
        self._sl_scale_factors = np.array(sl_list, dtype=np.float32).reshape(-1, 1, 1)

    def _log_init(self):
        n_files = len(self.nc_files)
        n_orbit = len(self.orbit_df)
        distances_km = self.orbit_df["distance_km"].values
        print(f"[OrbitDataset] {n_files} SDO files × {self.samples_per_file}"
              f" samples/file = {len(self)} samples/epoch")
        print(f"[OrbitDataset] SPO orbits: {n_orbit} rows, "
              f"distance range: {distances_km.min()/149597870.7:.2f}"
              f"–{distances_km.max()/149597870.7:.2f} AU")
        print(f"[OrbitDataset] Channels: {len(self.channels)}, "
              f"Pooling: {self.pooling}")

    # ================================================================
    # PyTorch Dataset 接口
    # ================================================================

    def __len__(self):
        return len(self.nc_files) * self.samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        filepath = self.nc_files[file_idx]

        # ── 1. 加载 SDO 图像并提取源距离 ──
        image, source_distance_km = self._load_and_transform(filepath)
        # image: (C, H, W) float32, signum-log 归一化
        # source_distance_km: float

        # ── 2. 目标距离 (从 SPO CSV 随机采样) ──
        row = self.orbit_df.sample(1).iloc[0]
        target_distance_km = float(row["distance_km"])

        # ── 3. 生成几何伪真值 ──
        zoom_scale = source_distance_km / target_distance_km
        gt_image = self._geometric_scale(image, zoom_scale)
        # gt_image: (C, H, W) float32

        # ── 4. 构建时间序列 ──
        # Surya 输入格式: (C, T, H, W), T=2
        ts = np.stack([image, image], axis=1)

        # Surya forecast 格式: (C, L, H, W), L=1
        forecast = gt_image[np.newaxis, :, :, :]

        return {
            "ts": ts.astype(np.float32),
            "time_delta_input": np.array([-0.2, 0.0], dtype=np.float32),
            "source_distance_km": np.float32(source_distance_km),
            "target_distance_km": np.float32(target_distance_km),
            "forecast": forecast.astype(np.float32),
            "lead_time_delta": np.array([0.2], dtype=np.float32),
        }, {
            "filepath": str(filepath),
            "source_distance_km": source_distance_km,
            "target_distance_km": target_distance_km,
            "zoom_scale": zoom_scale,
            "grid_scale": target_distance_km / source_distance_km,
        }

    # ================================================================
    # 图像加载与预处理
    # ================================================================

    def _load_and_transform(self, filepath: str) -> tuple[np.ndarray, float]:
        """
        加载 NC 文件 → signum-log 归一化 → 池化。
        同时提取 dsun_obs (对日直线距离)。

        返回:
            image: (C, H, W) float32
            distance_km: float
        """
        with xr.open_dataset(filepath, engine="h5netcdf", chunks=None) as ds:
            # 1. 读取图像数据
            data = ds[self.channels].to_array().load().to_numpy()
            # data: (C=13, 4096, 4096)

            # 2. 提取 dsun_obs (与 single_scale.py:38-71 逻辑一致)
            distance_km = self._extract_dsun_obs(ds)

        # 3. Signum-log 归一化
        data = data.astype(np.float32, copy=False)
        data = data * self._sl_scale_factors
        data = np.sign(data) * np.log1p(np.abs(data))
        data = (data - self._means) / (self._stds + self._epsilons)

        # 4. 池化降低分辨率
        if self.pooling > 1:
            C, H, W = data.shape
            new_h, new_w = H // self.pooling, W // self.pooling
            data = data.reshape(C, new_h, self.pooling, new_w, self.pooling)
            data = data.mean(axis=(2, 4))

        return data.astype(np.float32, copy=False), distance_km

    def _extract_dsun_obs(self, ds: xr.Dataset) -> float:
        """
        从 NC 文件的通道元数据中提取 dsun_obs (对日直线距离)。

        逻辑与 single_scale.py:38-71 完全一致：
          1. 遍历通道变量
          2. 尝试从 meta_0 或 meta_1 属性中解析 JSON
          3. 提取 dsun_obs 字段（单位：米）
          4. 转换为公里

        如果提取失败，回退到 1 AU ≈ 149,597,870.7 km。
        """
        for ch_name in self.channels:
            if ch_name not in ds:
                continue
            var = ds[ch_name]
            for meta_key in ["meta_0", "meta_1"]:
                if meta_key not in var.attrs:
                    continue
                try:
                    meta = json.loads(var.attrs[meta_key])
                    if "dsun_obs" in meta:
                        dsun_obs_m = float(meta["dsun_obs"])
                        return dsun_obs_m / 1000.0  # 米 → 公里
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

        # 回退值：SDO 在地球同步轨道，约 1 AU
        fallback_km = 149597870.7
        # 只在第一次回退时打印警告
        if not hasattr(self, "_fallback_warned"):
            print(f"  [WARN] dsun_obs not found in NC metadata, "
                  f"using fallback: {fallback_km:.0f} km (1 AU)")
            self._fallback_warned = True
        return fallback_km

    # ================================================================
    # 几何伪真值生成
    # ================================================================

    def _geometric_scale(
        self, image: np.ndarray, zoom_scale: float
    ) -> np.ndarray:
        """
        用 scipy.ndimage.zoom 做纯几何缩放，生成伪真值。

        zoom_scale = source_distance_km / target_distance_km
        - zoom_scale < 1 → 目标更远 → 太阳变小
        - zoom_scale > 1 → 目标更近 → 太阳变大
        - zoom_scale = 1 → 距离相同 → 不变

        与 single_scale.py 的 scale_single_channel 逻辑一致。
        """
        C, H, W = image.shape
        result = np.zeros_like(image, dtype=np.float32)

        for c in range(C):
            ch = image[c]                              # (H, W)
            scaled = zoom(ch, zoom=zoom_scale, order=3)  # 三次样条
            sh, sw = scaled.shape

            if zoom_scale >= 1.0:
                # 放大 → 裁剪中心区域
                start_y = (sh - H) // 2
                start_x = (sw - W) // 2
                result[c] = scaled[start_y:start_y + H,
                                    start_x:start_x + W]
            else:
                # 缩小 → 放入中心，四周填充 0（太空背景）
                start_y = (H - sh) // 2
                start_x = (W - sw) // 2
                result[c, start_y:start_y + sh,
                        start_x:start_x + sw] = scaled

        return result
