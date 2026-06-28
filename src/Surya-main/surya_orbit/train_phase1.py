"""
Phase 1 训练脚本 — 几何缩放训练 (基于对日直线距离)。

训练目标:
    让 Surya 学会根据观测距离变化，生成太阳视直径不同的图像。

训练策略:
    - 冻结: Embedding + SpectFormer Backbone (共 350M 参数)
    - 训练: OrbitAwareFlowModel (~5K) + LinearDecoder (~16M)
    - 监督: 几何缩放伪真值（scipy.ndimage.zoom 生成）
    - 损失: MSE(模型输出, 伪真值)

缩放依据:
    对日直线距离 (km):
      - 源距离从 NC 文件的 dsun_obs 元数据提取
      - 目标距离从 SPO CSV 的 distance_km 列读取
      - 缩放因子 = target_distance_km / source_distance_km

用法:
    python surya_orbit/train_phase1.py --config surya_orbit/config_phase1.yaml
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surya.utils.data import build_scalers, custom_collate_fn
from surya_orbit.orbit_dataset import OrbitDataset
from surya_orbit.orbit_spectformer import OrbitHelioSpectFormer


# ================================================================
# 工具函数
# ================================================================

def count_params(model, tag=""):
    """统计可训练/总参数"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"[{tag}] Parameters: {trainable/1e6:.2f}M trainable "
          f"/ {total/1e6:.2f}M total ({pct:.1f}%)")


def make_printable(v):
    """将各种类型转为可打印字符串"""
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return f"{v.item():.6f}" if v.dtype.is_floating_point else str(v.item())
        return f"tensor(shape={list(v.shape)}, dtype={v.dtype})"
    if isinstance(v, np.ndarray):
        return f"ndarray(shape={list(v.shape)}, dtype={v.dtype})"
    return str(v)


def debug_batch(batch, meta, prefix="[First Batch]"):
    """打印第一个 batch 的详细信息（验证数据管道用）"""
    print(f"\n{prefix} — Data Pipeline Verification")
    print(f"  ts:                  {make_printable(batch['ts'])}")
    print(f"  time_delta_input:    {make_printable(batch['time_delta_input'])}")
    print(f"  source_distance_km:  {make_printable(batch['source_distance_km'])}")
    print(f"  target_distance_km:  {make_printable(batch['target_distance_km'])}")
    print(f"  forecast:            {make_printable(batch['forecast'])}")

    # 提取标量值
    src = float(batch["source_distance_km"])
    tgt = float(batch["target_distance_km"])
    zoom_s = src / tgt
    grid_s = tgt / src
    print(f"\n  Source (SDO dsun_obs):    {src:.0f} km = {src/149597870.7:.3f} AU")
    print(f"  Target (SPO distance_km): {tgt:.0f} km = {tgt/149597870.7:.3f} AU")
    print(f"  → zoom scale (GT):   {zoom_s:.4f}")
    print(f"  → grid scale (model): {grid_s:.4f}")


# ================================================================
# 主函数
# ================================================================

def main(config_path: str):
    # ── 0. 加载配置 ──
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cfg_train = config["training"]
    cfg_data = config["data"]
    cfg_model = config["model"]
    cfg_out = config["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── 1. 加载 scalers ──
    scalers_path = Path(cfg_data["scalers_path"])
    with open(scalers_path, "r") as f:
        scalers = build_scalers(info=yaml.safe_load(f))

    # ── 2. 数据集 ──
    nc_files = sorted(Path(cfg_data["sdo_data_dir"]).glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(
            f"No .nc files found in {cfg_data['sdo_data_dir']}. "
            f"Did you download the SDO data?"
        )
    print(f"\n[Data] Found {len(nc_files)} NC files")

    dataset = OrbitDataset(
        nc_files=[str(f) for f in nc_files],
        orbit_csv_path=cfg_data["orbit_csv"],
        scalers=scalers,
        channels=cfg_data["sdo_channels"],
        samples_per_file=cfg_data["samples_per_file"],
        pooling=cfg_data["pooling"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        num_workers=cfg_train.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    # ── 3. 模型 ──
    img_size = cfg_data["img_size_after_pool"]
    n_channels = len(cfg_data["sdo_channels"])
    print(f"\n[Model] OrbitHelioSpectFormer (img_size={img_size}, "
          f"channels={n_channels})")

    model = OrbitHelioSpectFormer(
        img_size=img_size,
        patch_size=cfg_model["patch_size"],
        in_chans=n_channels,
        embed_dim=cfg_model["embed_dim"],
        time_embedding={
            "type": "linear",
            "time_dim": cfg_model["time_dim"],
        },
        depth=cfg_model["depth"],
        n_spectral_blocks=cfg_model["n_spectral_blocks"],
        num_heads=cfg_model["num_heads"],
        mlp_ratio=cfg_model["mlp_ratio"],
        drop_rate=0.0,
        window_size=cfg_model["window_size"],
        dp_rank=cfg_model["dp_rank"],
        learned_flow=True,
        use_latitude_in_learned_flow=False,
        init_weights=False,
        checkpoint_layers=None,
        rpe=False,
        ensemble=None,
        finetune=False,
        nglo=0,
    )

    # ── 4. 加载预训练权重 + 插值适配分辨率 ──
    weights_path = Path(cfg_model["sdo_model_repo"]) / \
                   cfg_model["pretrained_weights"]
    print(f"\n[Weights] Loading from {weights_path}")
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)

    # 插值位置编码：预训练 (1, 256², 1280) → 当前 (1, (1024/16)², 1280)
    old_grid = 256  # 4096 / 16
    new_grid = img_size // cfg_model["patch_size"]
    if old_grid != new_grid and "embedding.pos_embed" in weights:
        pe = weights["embedding.pos_embed"]  # (1, 65536, 1280)
        pe = pe.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)  # (1, 1280, 256, 256)
        pe = torch.nn.functional.interpolate(pe, size=(new_grid, new_grid),
                                              mode="bilinear", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
        weights["embedding.pos_embed"] = pe
        print(f"  Interpolated pos_embed: {old_grid}×{old_grid} → {new_grid}×{new_grid}")

    # 插值频谱门控权重：预训练 (256, 129, 1280, 2) → 当前 (new_grid, new_grid//2+1, 1280, 2)
    old_h, old_w = old_grid, old_grid // 2 + 1  # 256, 129
    new_h, new_w = new_grid, new_grid // 2 + 1  # 64, 33
    for i in range(cfg_model["n_spectral_blocks"]):
        key = f"backbone.blocks_spectral_gating.{i}.filter.complex_weight"
        if key in weights and (old_h != new_h or old_w != new_w):
            cw = weights[key]  # (256, 129, 1280, 2)
            cw = cw.permute(3, 2, 0, 1)  # (2, 1280, 256, 129)
            cw = torch.nn.functional.interpolate(cw, size=(new_h, new_w),
                                                  mode="bilinear", align_corners=False)
            cw = cw.permute(2, 3, 1, 0)  # (64, 33, 1280, 2)
            weights[key] = cw
            print(f"  Interpolated complex_weight[{i}]: "
                  f"{old_h}×{old_w} → {new_h}×{new_w}")

    # 填充 patch_embed.proj.weight:
    #   预训练 learned_flow=False → 26 通道 (13×2)
    #   我们   learned_flow=True  → 39 通道 (13×3)
    #   额外 13 通道拷贝自输入通道
    conv_key = "embedding.patch_embed.proj.weight"
    if conv_key in weights:
        old_w = weights[conv_key]  # (1280, 26, 16, 16)
        new_ic = n_channels * (cfg_model["time_dim"] + 1)  # 13 * 3 = 39
        if old_w.shape[1] < new_ic:
            pad = old_w[:, :n_channels, :, :].clone()
            weights[conv_key] = torch.cat([old_w, pad], dim=1)
            print(f"  Padded {conv_key}: {old_w.shape[1]} → "
                  f"{weights[conv_key].shape[1]} ch")

    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"  Missing keys  (new FlowModel params): {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    # ── 5. 冻结/解冻 ──
    for param in model.parameters():
        param.requires_grad = False

    for param in model.unembed.parameters():
        param.requires_grad = True
    print(f"  [Unfrozen] Decoder (LinearDecoder)")

    for param in model.learned_flow_model.parameters():
        param.requires_grad = True
    print(f"  [Unfrozen] OrbitAwareFlowModel")

    count_params(model, tag="Model")
    model.to(device)

    # ── 6. 优化器 ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg_train["learning_rate"],
    )
    print(f"\n[Optimizer] AdamW, lr={cfg_train['learning_rate']}")

    # ── 7. 训练 ──
    n_epochs = cfg_train["epochs"]
    n_batches = len(dataloader)
    print(f"\n[Training] {n_epochs} epochs × {n_batches} batches "
          f"(batch_size={cfg_train['batch_size']})")
    print(f"[Training] Total updates: {n_epochs * n_batches}\n")

    checkpoint_dir = Path(cfg_out["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        first_batch = (epoch == 0)

        for batch_idx, (batch, meta) in enumerate(dataloader):
            if first_batch:
                debug_batch(batch, meta)
                first_batch = False

            # 移动到 GPU
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # 前向
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                pred = model(batch)
                target = batch["forecast"][:, :, 0, :, :]
                loss = F.mse_loss(pred, target)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            # 日志
            log_every = cfg_train.get("log_interval", 10)
            if batch_idx % log_every == 0 or batch_idx == n_batches - 1:
                try:
                    src = float(batch["source_distance_km"][0])
                    tgt = float(batch["target_distance_km"][0])
                    dst_au = tgt / 149597870.7
                    gs = tgt / src
                except Exception:
                    dst_au, gs = 0.0, 0.0
                print(f"  [E{epoch:3d}|B{batch_idx:4d}/{n_batches}] "
                      f"loss={loss_val:.6f}  "
                      f"dst={dst_au:.2f}AU  grid_scale={gs:.3f}")

        avg_loss = epoch_loss / n_batches
        print(f"── Epoch {epoch:3d} complete | Avg loss: {avg_loss:.6f} ──")

        save_every = cfg_train.get("save_every", 5)
        if (epoch + 1) % save_every == 0 or epoch == n_epochs - 1:
            ckpt_name = f"phase1_epoch{epoch + 1}.pt"
            ckpt_path = checkpoint_dir / ckpt_name
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            print(f"  [Saved] {ckpt_path}")

    print("\n[Done] Phase 1 training complete.")


# ================================================================
# 入口
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Phase 1: Geometric Scaling Training")
    parser.add_argument(
        "--config",
        default="surya_orbit/config_phase1.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)

    main(config_path)
