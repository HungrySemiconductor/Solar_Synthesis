#!/usr/bin/env python3
"""
预处理 NC 文件：load → signum-log → pool → 保存为 .npy。

每个 NC 文件只需跑一次。后续 Dataset 直接 np.load() .npy，
跳过 NC 解析 + signum-log + pool 的 CPU 瓶颈（原来 ~2s/样本 → ~0.1s/样本）。

用法:
    python surya_orbit/preprocess_nc.py \
        --nc-dir data/Surya-1.0_validation_data \
        --npy-dir data/Surya-1.0_validation_npy \
        --pooling 4

输出:
    data/Surya-1.0_validation_npy/
    ├── 20140107_1500__147097819km.npy    ← (13, 1024, 1024) float32, ~52 MB
    ├── 20140107_1512__147097818km.npy
    ├── ...
    └── manifest.csv                      ← 文件索引 (npy_name, source_distance_km)
"""

import argparse, json, sys, os
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import hdf5plugin
import xarray as xr
import yaml


def main():
    parser = argparse.ArgumentParser(description="Preprocess NC → .npy for fast training")
    parser.add_argument("--nc-dir", required=True)
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--scalers-path", default="data/Surya-1.0/scalers.yaml")
    parser.add_argument("--pooling", type=int, default=4,
                        help="Pooling factor: 4096/pooling = output size")
    args = parser.parse_args()

    nc_dir = Path(args.nc_dir)
    npy_dir = Path(args.npy_dir)
    npy_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 scalers ──
    with open(args.scalers_path) as f:
        scalers_info = yaml.safe_load(f)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from surya.utils.data import build_scalers
    scalers = build_scalers(scalers_info)

    # Surya 的 13 个通道
    channels = [
        "aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335",
        "aia1600", "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v",
    ]

    # 归一化参数 (signum-log: 与 Surya helio.py 一致)
    means = np.array([scalers[ch].mean for ch in channels],
                     dtype=np.float32).reshape(-1, 1, 1)
    stds = np.array([scalers[ch].std for ch in channels],
                    dtype=np.float32).reshape(-1, 1, 1)
    epsilons = np.array([scalers[ch].epsilon for ch in channels],
                        dtype=np.float32).reshape(-1, 1, 1)
    sl_scales = np.array([scalers[ch].sl_scale_factor for ch in channels],
                         dtype=np.float32).reshape(-1, 1, 1)

    # ── 处理每个 NC 文件 ──
    nc_files = sorted(nc_dir.glob("*.nc"))
    print(f"NC dir:    {nc_dir}")
    print(f"NPY dir:   {npy_dir}")
    print(f"Files:     {len(nc_files)}")
    print(f"Pooling:   {args.pooling} (4096 → {4096 // args.pooling})")
    print()

    manifest = []
    for i, fp in enumerate(nc_files):
        t0 = perf_counter()

        # 1. 读取 NC + 提取 dsun_obs
        with xr.open_dataset(fp, engine="h5netcdf", chunks=None) as ds:
            data = ds[channels].to_array().load().to_numpy()
            distance_km = _extract_dsun_obs(ds, channels)

        # 2. Signum-log 归一化
        data = data.astype(np.float32)
        data = data * sl_scales
        data = np.sign(data) * np.log1p(np.abs(data))
        data = (data - means) / (stds + epsilons)

        # 3. 池化
        p = args.pooling
        if p > 1:
            C, H, W = data.shape
            h, w = H // p, W // p
            data = data.reshape(C, h, p, w, p).mean(axis=(2, 4))

        # 4. 保存
        npy_name = f"{fp.stem}__{distance_km:.0f}km.npy"
        np.save(npy_dir / npy_name, data.astype(np.float32))

        manifest.append({
            "npy_file": npy_name,
            "source_nc": str(fp),
            "source_distance_km": distance_km,
        })

        elapsed = perf_counter() - t0
        print(f"  [{i+1:3d}/{len(nc_files)}] {fp.name}  "
              f"→ {npy_name}  ({distance_km/149597870.7:.3f} AU, "
              f"{data.shape[1]}×{data.shape[2]}, {elapsed:.1f}s)")

    # ── 保存 manifest ──
    manifest_path = npy_dir / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    total_mb = sum(f.stat().st_size for f in npy_dir.glob("*.npy")) / (1024**2)
    print(f"\nDone.")
    print(f"  Files: {len(manifest)} .npy  ({total_mb:.0f} MB total)")
    print(f"  Manifest: {manifest_path}")


def _extract_dsun_obs(ds, channels):
    """从 meta_0/meta_1 中提取 dsun_obs (米) → 公里"""
    for ch_name in channels:
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


if __name__ == "__main__":
    main()
