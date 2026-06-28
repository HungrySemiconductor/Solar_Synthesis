#!/usr/bin/env python3
"""
Phase 1 推理脚本 (CSV 模式) — 从 SPO CSV 中读取目标距离进行推理。

与 infer_phase1.py 的区别:
    infer_phase1.py:  手动指定距离 (--distances 0.83 1.0 ...)
    本文件:            从 SPO CSV 中按时间戳或行号自动取距离

用法:
    # 按时间戳推理
    python surya_orbit/infer_from_csv.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/inference_test.nc \
        --orbit-csv data/spo_orbit_values.csv \
        --timestamp "2030-01-31 16:23:17" \
        --output-dir outputs/phase1_inference_csv

    # 推理 CSV 中所有行（建议限制范围）
    python surya_orbit/infer_from_csv.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/inference_test.nc \
        --orbit-csv data/spo_orbit_values.csv \
        --start-row 0 --end-row 8 \
        --output-dir outputs/phase1_inference_csv

    # 推理 CSV 中随机 N 行
    python surya_orbit/infer_from_csv.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/inference_test.nc \
        --orbit-csv data/spo_orbit_values.csv \
        --random 8 \
        --output-dir outputs/phase1_inference_csv

输出:
    outputs/phase1_inference_csv/
    ├── prediction.nc              ← 所有推理结果的 NetCDF (完整科学数据)
    ├── sun_diameter.csv           ← 太阳直径统计
    └── previews/                  ← PNG 预览图 (每个推理样本一张)
        ├── 2030-01-31_162317_1.54AU.png
        ├── ...
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
import hdf5plugin
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surya.utils.data import build_scalers
from surya_orbit.orbit_spectformer import OrbitHelioSpectFormer


# ============================================================
# 工具函数 (与 infer_phase1.py 共用)
# ============================================================

def load_model(checkpoint_path: str, config: dict, device: torch.device):
    cfg = config["model"]
    data_cfg = config["data"]
    model = OrbitHelioSpectFormer(
        img_size=data_cfg["img_size_after_pool"],
        patch_size=cfg["patch_size"],
        in_chans=len(data_cfg["sdo_channels"]),
        embed_dim=cfg["embed_dim"],
        time_embedding={"type": "linear", "time_dim": cfg["time_dim"]},
        depth=cfg["depth"],
        n_spectral_blocks=cfg["n_spectral_blocks"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        drop_rate=0.0,
        window_size=cfg["window_size"],
        dp_rank=cfg["dp_rank"],
        learned_flow=True,
        use_latitude_in_learned_flow=False,
        init_weights=False,
        checkpoint_layers=None,
        rpe=False,
        ensemble=None,
        finetune=False,
        nglo=0,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] {n_params:.1f}M parameters loaded to {device}")
    return model


def extract_dsun_obs(ds: xr.Dataset, channels: list[str]) -> float:
    for ch_name in channels:
        if ch_name not in ds:
            continue
        var = ds[ch_name]
        for meta_key in ["meta_0", "meta_1"]:
            if meta_key not in var.attrs:
                continue
            try:
                meta = json.loads(var.attrs[meta_key])
                if "dsun_obs" in meta:
                    return float(meta["dsun_obs"]) / 1000.0
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return 149597870.7


def load_and_transform_sdo(
    filepath: str, scalers: dict, channels: list[str], pooling: int = 2
) -> tuple[np.ndarray, float]:
    with xr.open_dataset(filepath, engine="h5netcdf", chunks=None) as ds:
        data = ds[channels].to_array().load().to_numpy().astype(np.float32)
        distance_km = extract_dsun_obs(ds, channels)

    means = np.array([scalers[ch].mean for ch in channels],
                     dtype=np.float32).reshape(-1, 1, 1)
    stds = np.array([scalers[ch].std for ch in channels],
                    dtype=np.float32).reshape(-1, 1, 1)
    epsilons = np.array([scalers[ch].epsilon for ch in channels],
                        dtype=np.float32).reshape(-1, 1, 1)
    sl_scales = np.array([scalers[ch].sl_scale_factor for ch in channels],
                         dtype=np.float32).reshape(-1, 1, 1)

    data = data * sl_scales
    data = np.sign(data) * np.log1p(np.abs(data))
    data = (data - means) / (stds + epsilons)

    if pooling > 1:
        C, H, W = data.shape
        h, w = H // pooling, W // pooling
        data = data.reshape(C, h, pooling, w, pooling).mean(axis=(2, 4))

    return data.astype(np.float32), distance_km


def denormalize_image(data: np.ndarray, scalers: dict,
                      channels: list[str]) -> np.ndarray:
    """反归一化：signum-log 空间 → 物理 DN/s"""
    means = np.array([scalers[ch].mean for ch in channels],
                     dtype=np.float32).reshape(-1, 1, 1)
    stds = np.array([scalers[ch].std for ch in channels],
                    dtype=np.float32).reshape(-1, 1, 1)
    epsilons = np.array([scalers[ch].epsilon for ch in channels],
                        dtype=np.float32).reshape(-1, 1, 1)
    sl_scales = np.array([scalers[ch].sl_scale_factor for ch in channels],
                         dtype=np.float32).reshape(-1, 1, 1)

    data = data * (stds + epsilons) + means
    data = np.sign(data) * np.expm1(np.abs(data))
    return (data / sl_scales).astype(np.float32)


def run_single_inference(model, image_norm, source_km, target_km, device):
    """跑一次推理, 返回归一化空间的 numpy 预测 (C, H, W)"""
    C, H, W = image_norm.shape
    ts_np = np.stack([image_norm, image_norm], axis=1)  # (C, 2, H, W)
    batch = {
        "ts": torch.from_numpy(ts_np).unsqueeze(0).to(device),
        "time_delta_input": torch.tensor([[-0.2, 0.0]], device=device),
        "source_distance_km": torch.tensor([source_km], device=device),
        "target_distance_km": torch.tensor([target_km], device=device),
    }
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            pred = model(batch)
    return pred[0].cpu().numpy().astype(np.float32)


# ============================================================
# CSV 查询
# ============================================================

def get_rows_from_csv(orbit_df, timestamp=None, start_row=None, end_row=None,
                      random_n=None):
    """从 orbit DataFrame 中筛选要推理的行"""
    if timestamp is not None:
        # 精确匹配时间戳
        match = orbit_df[orbit_df["timestamp"] == timestamp]
        if len(match) == 0:
            # 尝试模糊匹配
            match = orbit_df[orbit_df["timestamp"].str.startswith(
                timestamp[:16])]
        if len(match) == 0:
            raise ValueError(f"No row found for timestamp: {timestamp}")
        return match

    if random_n is not None:
        return orbit_df.sample(n=min(random_n, len(orbit_df)))

    if start_row is not None:
        end = end_row if end_row is not None else start_row + 1
        return orbit_df.iloc[start_row:end]

    # 默认: 取前 8 行
    print("[CSV] No selection mode specified, using first 8 rows")
    return orbit_df.head(8)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        "Phase 1: Inference from SPO CSV")
    parser.add_argument("--config", default="surya_orbit/config_phase1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-nc", required=True,
                        help="SDO NC 文件路径 (用于推理, 不应该用训练数据)")
    parser.add_argument("--orbit-csv", required=True,
                        help="SPO 轨道 CSV 路径")
    parser.add_argument("--output-dir", default="outputs/phase1_inference_csv")

    # CSV 行选择方式 (三选一)
    parser.add_argument("--timestamp", default=None,
                        help="精确时间戳, 如 '2030-01-31 16:23:17'")
    parser.add_argument("--start-row", type=int, default=None,
                        help="起始行号 (配合 --end-row)")
    parser.add_argument("--end-row", type=int, default=None,
                        help="结束行号 (配合 --start-row, 不含)")
    parser.add_argument("--random", type=int, default=None,
                        help="随机取 N 行")

    args = parser.parse_args()

    # 验证: 至少选一种方式
    if not any([args.timestamp, args.start_row is not None, args.random]):
        print("[INFO] No selection mode specified. Using --random 8 as default.")
        args.random = 8

    # ── 加载配置 ──
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 加载模型 ──
    model = load_model(args.checkpoint, config, device)

    # ── 加载 scalers ──
    with open(config["data"]["scalers_path"], "r") as f:
        scalers = build_scalers(info=yaml.safe_load(f))

    channels = config["data"]["sdo_channels"]
    pooling = config["data"]["pooling"]

    # ── 加载输入图像 ──
    print(f"[Data] Loading {args.input_nc}")
    image_norm, source_distance_km = load_and_transform_sdo(
        args.input_nc, scalers, channels, pooling
    )
    image_phys_input = denormalize_image(image_norm, scalers, channels)
    print(f"[Data] Source distance: {source_distance_km:.0f} km "
          f"= {source_distance_km/149597870.7:.3f} AU")

    # ── 加载 CSV ──
    print(f"[CSV] Loading {args.orbit_csv}")
    orbit_df = pd.read_csv(args.orbit_csv)
    if "distance_km" not in orbit_df.columns:
        raise ValueError("CSV must have 'distance_km' column")

    # ── 选择行 ──
    rows = get_rows_from_csv(
        orbit_df,
        timestamp=args.timestamp,
        start_row=args.start_row,
        end_row=args.end_row,
        random_n=args.random,
    )
    n_samples = len(rows)
    print(f"[CSV] Selected {n_samples} row(s) for inference")

    # ════════════════════════════════════════════════════════════
    # 逐个推理
    # ════════════════════════════════════════════════════════════
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    results = []
    predictions_phys = {}   # {sample_idx: np.ndarray (C, H, W)}

    for sample_idx, (_, row) in enumerate(rows.iterrows()):
        target_km = float(row["distance_km"])
        target_au = target_km / 149597870.7
        timestamp = str(row.get("timestamp", f"row_{sample_idx}"))

        print(f"\n[{sample_idx + 1}/{n_samples}] "
              f"timestamp={timestamp[:19]}, "
              f"distance={target_au:.3f} AU ({target_km:.0f} km)")

        pred_norm = run_single_inference(
            model, image_norm, source_distance_km, target_km, device)
        pred_phys = denormalize_image(pred_norm, scalers, channels)
        predictions_phys[sample_idx] = pred_phys

        # ── 快速统计 ──
        diam = _estimate_diameter(pred_phys[0])

        results.append({
            "sample_idx": sample_idx,
            "timestamp": timestamp,
            "target_distance_km": target_km,
            "target_distance_au": target_au,
            "grid_scale": target_km / source_distance_km,
            "sun_diameter_px": diam,
        })

        # ── PNG 预览 (只显示 3 个通道) ──
        safe_ts = timestamp[:19].replace(" ", "_").replace(":", "")
        png_path = preview_dir / f"{safe_ts}_{target_au:.2f}AU.png"
        _save_preview(
            pred_phys, channels,
            plot_channels=["aia171", "aia193", "aia304"],
            path=png_path,
            title=f"Predicted: {timestamp[:19]}\n"
                  f"{source_distance_km/149597870.7:.3f} AU → {target_au:.3f} AU")

    # ════════════════════════════════════════════════════════════
    # CSV 统计
    # ════════════════════════════════════════════════════════════
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "sun_diameter.csv", index=False)
    print(f"\n[Saved] {out_dir / 'sun_diameter.csv'}")

    # ════════════════════════════════════════════════════════════
    # NetCDF — 完整科学数据
    # ════════════════════════════════════════════════════════════
    H, W = image_phys_input.shape[-2], image_phys_input.shape[-1]
    nc_path = out_dir / "prediction.nc"
    import h5netcdf
    with h5netcdf.File(str(nc_path), "w") as f:
        f.dimensions = {"y": H, "x": W, "sample": n_samples}
        f.attrs["title"] = "Surya-Orbit Phase 1 — CSV-based Inference"
        f.attrs["description"] = (
            "Model predictions at SPO trajectory points. "
            "All *_<ch> variables are in physical units (DN/s). "
            "input_<ch>: SDO image at source distance. "
            "pred_<ch>: model prediction at each SPO distance. "
            "target_distance_km / target_distance_au: per-sample metadata."
        )
        f.attrs["source_distance_km"] = source_distance_km
        f.attrs["source_distance_au"] = source_distance_km / 149597870.7
        f.attrs["input_file"] = args.input_nc
        f.attrs["checkpoint"] = args.checkpoint
        f.attrs["orbit_csv"] = args.orbit_csv

        # 元数据
        f.create_variable("target_distance_km", ("sample",),
                          dtype="f8")[:] = df["target_distance_km"].values
        f.create_variable("target_distance_au", ("sample",),
                          dtype="f4")[:] = df["target_distance_au"].values.astype(np.float32)

        # 输入图
        for i, ch in enumerate(channels):
            f.create_variable(f"input_{ch}", ("y", "x"),
                              dtype="f4")[...] = image_phys_input[i].astype(np.float32)

        # 各样本预测
        for i, ch in enumerate(channels):
            var = f.create_variable(f"pred_{ch}", ("sample", "y", "x"),
                                    dtype="f4")
            for j in range(n_samples):
                var[j, :, :] = predictions_phys[j][i].astype(np.float32)

    print(f"[Saved] {nc_path}")
    print(f"\n[Done] {n_samples} inferences → {out_dir}/")


# ============================================================
# 快速辅助函数 (不依赖 infer_phase1.py)
# ============================================================

def _estimate_diameter(img_1ch, frac=0.1):
    thr = frac * img_1ch.max()
    binary = (img_1ch > thr)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        return 0
    return int(max(rows.sum(), cols.sum()))


def _save_preview(image_phys, all_channels, plot_channels, path, title):
    n = len(plot_channels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, ch_name in zip(axes, plot_channels):
        idx = all_channels.index(ch_name)
        img = image_phys[idx]
        vmax = np.percentile(img[img > 0], 98) if (img > 0).any() else img.max()
        im = ax.imshow(img, cmap='inferno', origin='lower', vmin=0, vmax=vmax)
        ax.set_title(ch_name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
