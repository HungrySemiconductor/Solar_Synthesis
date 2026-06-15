"""
OrbitHelioSpectFormer — 支持轨道条件注入的 Surya 模型。

继承自 HelioSpectFormer。
Phase 1（几何缩放训练）的改动：
  1. 将 learned_flow_model 替换为 OrbitAwareFlowModel
  2. 重写 forward() 去掉"FlowModel 训练时直接返回 flow 输出"的逻辑，
     改为始终走完整流水线：FlowModel → Embedding → Backbone → Decoder

Phase 2（纹理训练，预留）的改动：
  3. 解冻 adaLN 调制层
  4. 添加 LoRA 到 Attention 权重
"""

import torch
import torch.nn as nn
from einops import rearrange

from surya.models.helio_spectformer import HelioSpectFormer
from surya_orbit.orbit_flow import OrbitAwareFlowModel


class OrbitHelioSpectFormer(HelioSpectFormer):
    """
    轨道条件化的 HelioSpectFormer。

    Phase 1 使用方式:
        model = OrbitHelioSpectFormer(learned_flow=True, ...)
        # 加载 Surya 预训练权重 (strict=False)
        # 冻结除 FlowModel 和 Decoder 外的所有参数
        # 训练

    Phase 2 使用方式:
        # 加载 Phase 1 checkpoint
        # 解冻 adaLN 层 + 添加 LoRA
        # 继续训练
    """

    def __init__(self, **kwargs):
        # ── 调用父类初始化 ──
        # 父类会：
        #   1. 创建 self.learned_flow_model = HelioFlowModel(...)
        #   2. 根据 time_embedding 创建 self.embedding
        #   3. 创建 self.backbone = SpectFormer(...)
        #   4. 创建 self.unembed
        super().__init__(**kwargs)

        # ── 替换 FlowModel 为轨道感知版本 ──
        if self.learned_flow:
            img_size = kwargs.get("img_size", 4096)
            self.learned_flow_model = OrbitAwareFlowModel(
                img_size=(img_size, img_size),
            )

    def forward(self, batch):
        """
        重写 forward，与父类的区别：
        ┌──────────────────────────────────────────────────────────┐
        │ 父类:                                                     │
        │   if FlowModel.requires_grad:                            │
        │       return y_hat_flow  ← 跳过 Embedding/Backbone/Decoder│
        │                                                          │
        │ 本类:                                                     │
        │   始终走完整流水线，这样 FlowModel 和 Decoder 可以同时训练  │
        └──────────────────────────────────────────────────────────┘
        """
        x = batch["ts"]
        dt = batch["time_delta_input"]
        B, C, T, H, W = x.shape

        # ── ① FlowModel: 距离感知的空间缩放 ──
        y_hat_flow = self.learned_flow_model(batch)   # (B, C, H, W)

        # 粘贴 flow 输出到图像序列（不跳过 pipeline）
        x = torch.concat((x, y_hat_flow.unsqueeze(2)), dim=2)
        # (B, C, T+1, H, W)

        # Perceiver 模式需要调整时间 delta
        if self.time_embedding["type"] == "perceiver":
            dt = torch.cat(
                (dt, batch["lead_time_delta"].reshape(-1, 1)), dim=1
            )

        # ── ② Embedding: 像素 → tokens ──
        tokens = self.embedding(x, dt)
        # (B, L, D)

        # ── ③ Backbone: SpectFormer 特征提取 ──
        if self.ensemble:
            tokens = torch.repeat_interleave(
                tokens, repeats=self.ensemble, dim=0
            )

        tokens = self.backbone(tokens)
        # (B, L, D)

        if self.finetune:
            return tokens

        # ── ④ Decoder: tokens → 像素 ──
        forecast_hat = self.unembed(tokens)
        # (B, C, H, W)

        # ── ⑤ 残差连接 ──
        # y_hat_flow 提供了基础的几何缩放，
        # Decoder 输出的是 token 空间中的修正
        if self.ensemble:
            y_hat_flow = torch.repeat_interleave(
                y_hat_flow, repeats=self.ensemble, dim=0
            )

        forecast_hat = forecast_hat + y_hat_flow

        # ── Ensemble reshape ──
        if self.ensemble:
            forecast_hat = rearrange(
                forecast_hat, "(B E) C H W -> B E C H W",
                B=B, E=self.ensemble
            )

        return forecast_hat
