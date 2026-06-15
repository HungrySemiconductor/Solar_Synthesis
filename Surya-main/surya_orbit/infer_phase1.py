#!/usr/bin/env python3
"""
Phase 1 推理脚本 — 加载训练好的模型，生成不同对日距离的太阳图像。

缩放依据：
    对日直线距离 (km) — 从 NC 文件的 dsun_obs 元数据提取。

用法:
    python surya_orbit/infer_phase1.py \
        --checkpoint checkpoints/phase1/phase1_epoch20.pt \
        --input-nc data/SDO_20141023/20141023_1200.nc \
        --distances 0.83 1.0 1.5 2.5 3.2 \
        --output-dir outputs/phase1_inference

输出:
    outputs/phase1_inference/
    ├── input_1AU.png              ← 输入图像（指定通道网格）
    ├── scaled_0.83AU.png          ← 各距离的预测图
    ├── scaled_1.00AU.png
    ├── scaled_1.50AU.png
    ├── scaled_2.50AU.png
    ├── scaled_3.20AU.png
    ├── comparison.png             ← 并排对比
    ├── sun_diameter.csv           ← 太阳像素直径统计
    └── prediction.nc              ← NetCDF 格式的预测
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import hdf5plugin  # NC 文件 blosc 压缩插件
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surya.utils.data import build_scalers
from surya_orbit.orbit_spectformer import OrbitHelioSpectFormer


# ============================================================
# 工具函数
# ============================================================

def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """加载 Phase 1 训练好的模型"""
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
    """
    从 NC 元数据中提取对日直线距离 (km)。
    与 single_scale.py 逻辑一致。
    """
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
    return 149597870.7  # fallback: 1 AU


def load_and_transform_sdo(
    filepath: str, scalers: dict, channels: list[str], pooling: int = 2
) -> tuple[np.ndarray, float]:
    """加载 SDO NC 文件 → signum-log 归一化 → 池化。返回 (image, distance_km)。"""
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
    """反归一化：signum-log 空间 → 原始物理值 (DN/s)。所有通道批量处理。"""
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


def denormalize_single_channel(data_1ch: np.ndarray,
                               scaler) -> np.ndarray:
    """反归一化单个通道：signum-log → 物理 DN/s"""
    data = data_1ch * (scaler.std + scaler.epsilon) + scaler.mean
    data = np.sign(data) * np.expm1(np.abs(data))
    return (data / scaler.sl_scale_factor).astype(np.float32)


def estimate_sun_diameter(image_1ch: np.ndarray,
                          threshold_frac: float = 0.1) -> int:
    """估算太阳圆面像素直径"""
    thr = threshold_frac * image_1ch.max()
    binary = (image_1ch > thr)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        return 0
    return int(max(rows.sum(), cols.sum()))


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser("Phase 1: Orbit-Conditioned Surya Inference")
    parser.add_argument("--config", default="surya_orbit/config_phase1.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Phase 1 checkpoint 路径 (.pt)")
    parser.add_argument("--input-nc", required=True,
                        help="输入 SDO NC 文件路径")
    parser.add_argument("--distances", nargs="+", type=float,
                        default=[0.83, 1.0, 1.5, 2.5, 3.2],
                        help="目标距离 (AU), 空格分隔")
    parser.add_argument("--channels-to-plot", nargs="+", type=str,
                        default=["aia171", "aia193", "aia304"],
                        help="要显示的通道名")
    parser.add_argument("--output-dir", default="outputs/phase1_inference")
    args = parser.parse_args()

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
    img_size = config["data"]["img_size_after_pool"]

    # ── 加载并预处理输入图像 ──
    print(f"[Data] Loading {args.input_nc}")
    image_norm, source_distance_km = load_and_transform_sdo(
        args.input_nc, scalers, channels, pooling
    )
    print(f"[Data] Source distance (dsun_obs): {source_distance_km:.0f} km "
          f"= {source_distance_km/149597870.7:.3f} AU")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 保存输入图像 PNG ──
    image_phys = denormalize_image(image_norm, scalers, channels)
    save_preview(image_phys, channels, args.channels_to_plot,
                 out_dir / "input_1AU.png",
                 title=f"Input (dsun_obs={source_distance_km/149597870.7:.3f} AU)")

    # ════════════════════════════════════════════════════════════
    # 不同距离推理
    # ════════════════════════════════════════════════════════════
    ts_np = np.stack([image_norm, image_norm], axis=1)  # (C, 2, H, W)
    results = []
    predictions_norm = {}   # 保存归一化空间结果, 用于 NC 输出
    predictions_phys = {}   # 保存物理空间结果, 用于 PNG

    for dist_au in args.distances:
        target_km = dist_au * 149597870.7
        print(f"[Infer] Distance = {dist_au:.2f} AU ({target_km:.0f} km)")

        batch = {
            "ts": torch.from_numpy(ts_np).unsqueeze(0).to(device),
            "time_delta_input": torch.tensor([[-0.2, 0.0]], device=device),
            "source_distance_km": torch.tensor([source_distance_km],
                                               device=device),
            "target_distance_km": torch.tensor([target_km], device=device),
        }

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                pred = model(batch)

        pred_norm = pred[0].cpu().numpy().astype(np.float32)
        predictions_norm[dist_au] = pred_norm
        pred_phys = denormalize_image(pred_norm, scalers, channels)
        predictions_phys[dist_au] = pred_phys

        diam = estimate_sun_diameter(pred_phys[0])
        diam_expected = img_size / dist_au * 0.95
        print(f"  Sun diameter: {diam} px (expected ~{diam_expected:.0f})")

        # ── PNG 输出 ──
        label = f"{dist_au:.2f} AU"
        save_preview(pred_phys, channels, args.channels_to_plot,
                     out_dir / f"scaled_{label.replace(' ','')}.png",
                     title=f"Predicted at {label}")

        results.append({
            "distance_au": dist_au,
            "distance_km": target_km,
            "sun_diameter_px": diam,
            "expected_diameter_px": diam_expected,
            "grid_scale": target_km / source_distance_km,
        })

    # ── CSV ──
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "sun_diameter.csv", index=False)
    print(f"\n[Saved] {out_dir / 'sun_diameter.csv'}")

    # ════════════════════════════════════════════════════════════
    # NetCDF 输出 — 完整的科学数据
    # ════════════════════════════════════════════════════════════
    distances_au = args.distances
    n_dist = len(distances_au)
    H, W = image_phys.shape[-2], image_phys.shape[-1]

    nc_path = out_dir / "prediction.nc"
    import h5netcdf
    with h5netcdf.File(str(nc_path), "w") as f:
        f.dimensions = {"y": H, "x": W, "distance": n_dist}
        f.attrs["title"] = "Surya-Orbit Phase 1 Geometric Scaling Inference"
        f.attrs["description"] = (
            "Model predictions at multiple heliocentric distances. "
            "All *_<ch> variables are in physical units (DN/s) after "
            "inverse signum-log transform. "
            "input_<ch>: SDO image at source distance. "
            "pred_<ch>: model prediction at each target distance."
        )
        f.attrs["source_distance_km"] = source_distance_km
        f.attrs["source_distance_au"] = source_distance_km / 149597870.7
        f.attrs["input_file"] = args.input_nc
        f.attrs["checkpoint"] = args.checkpoint

        # 距离坐标
        f.create_variable("distance_au", ("distance",),
                          dtype="f4")[...] = np.array(distances_au, dtype=np.float32)

        # 输入图 (每个通道一个 2D 变量)
        for i, ch in enumerate(channels):
            f.create_variable(f"input_{ch}", ("y", "x"),
                              dtype="f4")[...] = image_phys[i].astype(np.float32)

        # 各距离预测 (每个通道一个 3D 变量: distance × y × x)
        for i, ch in enumerate(channels):
            var = f.create_variable(f"pred_{ch}", ("distance", "y", "x"),
                                    dtype="f4")
            for j, d in enumerate(distances_au):
                var[j, :, :] = predictions_phys[d][i].astype(np.float32)

    print(f"[Saved] {nc_path}")
    print(f"\n[Done] All outputs in {out_dir}/")
    print(f"\n[Done] All outputs in {out_dir}/")


# ============================================================
# 可视化
# ============================================================

def save_preview(image_phys, all_channels, plot_channels, path, title):
    """保存多通道网格预览图"""
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

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


if __name__ == "__main__":
    main()
