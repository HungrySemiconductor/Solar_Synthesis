"""
OrbitAwareFlowModel — 基于对日直线距离的空间缩放模块。

继承自 surya.models.flow.HelioFlowModel，在原始 FlowModel 基础上增加：
- 直接从 batch 中读取 source_distance_km 和 target_distance_km
- 根据距离比缩放 grid（F.grid_sample 操作的是图像归一化像素坐标 [-1,1]）
- 可学习的缩放修正因子（±10% 范围）

坐标系说明：
  F.grid_sample 的 grid 是图像像素归一化坐标 (u,v) ∈ [-1,1]。
  缩放 grid 就是在缩放图像——与日心坐标系 / 经纬度完全无关。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from surya.models.flow import HelioFlowModel


class OrbitAwareFlowModel(HelioFlowModel):
    """
    距离感知的空间流模型。

    缩放原理:
        grid_sample 的 grid 是输出像素对应输入图像中的归一化坐标。
        grid × scale > 1 → 采样范围扩张 → 太阳表观变小 (目标更远)
        grid × scale < 1 → 采样范围收缩 → 太阳表观变大 (目标更近)

        目标距离 : 源距离 = target_km : source_km
        scale = target_km / source_km

    与 single_scale.py 的关系:
        single_scale  用 scipy.ndimage.zoom(image, r_src/r_tgt)
        grid_sample   用 grid × (r_tgt/r_src)
        两者互为倒数、结果等效 ── 只是实现方式不同。
    """

    def __init__(self, img_size=(2048, 2048)):
        # 调用父类初始化（创建 self.grid, self.higher_modes, self.flow_generator）
        super().__init__(img_size=img_size)

        # 缩放修正 MLP：目标距离 (1 个标量) → 1 个缩放修正值
        # 输入是目标距离归一化到 AU（除以 149,597,870.7 km）
        self.scale_corrector = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, batch):
        """
        Args:
            batch: dict, 必须包含:
                - ts: (B, C, T, H, W) 输入图像序列
                - source_distance_km: (B,) 源距离 (km), 从 NC 文件 dsun_obs 提取
                - target_distance_km: (B,) 目标距离 (km), 从 SPO CSV distance_km 列
        Returns:
            y_hat: (B, C, H, W) 经距离缩放后的图像
        """
        # ── 1. 提取最后一帧图像 ──
        x = batch["ts"]                         # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        if T == 1:
            x = x[:, :, -1, :, :]               # (B, C, H, W)
        else:
            x = (x[:, :, -1, :, :] + x[:, :, -2, :, :]) / 2.0

        # ── 2. 提取对日直线距离 ──
        r_src = batch["source_distance_km"]      # (B,) km
        r_tgt = batch["target_distance_km"]      # (B,) km

        # ── 3. 计算缩放因子 ──
        # scale = r_tgt / r_src
        #   r_tgt > r_src (更远) → scale > 1 → grid 扩张 → 太阳变小 ✓
        #   r_tgt < r_src (更近) → scale < 1 → grid 收缩 → 太阳变大 ✓
        base_scale = r_tgt / (r_src + 1e-8)      # (B,)

        # 可学习的修正 (±10%)：用目标距离 AU 值作为条件
        tgt_au = (r_tgt / 149597870.7).view(-1, 1)  # (B, 1) 归一化到 AU
        correction = self.scale_corrector(tgt_au).squeeze(-1)  # (B,)
        scale = base_scale * (1.0 + 0.1 * torch.tanh(correction))

        # ── 4. 缩放 grid 并采样 ──
        # self.grid: (1, H, W, 2), 像素归一化坐标 (u,v) ∈ [-1,1]
        scaled_grid = self.grid * scale.view(B, 1, 1, 1)   # (B, H, W, 2)

        # 父类的 flow_generator 提供微小的局部修正（如边缘像素级微调）
        flow_field = scaled_grid + self.flow_generator(self.higher_modes)
        # (B, H, W, 2)

        y_hat = F.grid_sample(
            x, flow_field,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        return y_hat  # (B, C, H, W)
