# -*- coding: utf-8 -*-
"""掩码数据损失（M1 + B2 相对误差项）：按域×变量掩码的加权 MSE，权重 1/σ²（数据标准差）。

B2（阶段 B）：增加可选"相对误差"项，对量级跨度大的源项（qav_an/qav_ca/htQ）
用 |pred-true|/(|true|+eps) 均衡低电流/高电流工况，避免全局归一化 MSE 偏向大值点。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskedDataLoss(nn.Module):
    """L = L_abs + lambda_rel * L_rel

    L_abs = Σ w_v·M·(pred−target)² / Σ w_v·M     （归一化空间）
    L_rel = Σ_v λ_v · mean( M·|pred−target| / (|target|+eps) )  （仅 relative_vars）

    var_weights: (n_out,) = 1/σ_v² 或可学习。
    relative_vars: {var_index: λ_v}，在归一化空间计算（eps 默认 0.1σ，避免 true≈0 爆炸）。
    """

    def __init__(self, var_weights: torch.Tensor, learnable: bool = False,
                 relative_vars: dict | None = None, lambda_rel: float = 1.0,
                 eps: float = 0.1):
        super().__init__()
        if learnable:
            log_sigma = nn.Parameter(torch.zeros_like(var_weights))
            self.register_parameter("log_sigma", log_sigma)
            self.fixed = None
        else:
            self.register_buffer("fixed", var_weights)
            self.log_sigma = None
        self.relative_vars = dict(relative_vars or {})
        self.lambda_rel = float(lambda_rel)
        self.eps = float(eps)

    def weights(self) -> torch.Tensor:
        if self.log_sigma is not None:
            # 不确定性加权：w = exp(-2 log σ) = 1/σ²
            return torch.exp(-2.0 * self.log_sigma)
        return self.fixed

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = self.weights()[None, :]
        wm = w * mask
        target = torch.nan_to_num(target, nan=0.0)
        L_abs = ((pred - target) ** 2 * wm).sum() / wm.sum().clamp(min=1.0)
        L_rel = torch.zeros((), device=pred.device)
        if self.relative_vars:
            for j, lam in self.relative_vars.items():
                mj = mask[:, j].to(pred.dtype)
                denom = target[:, j].abs() + self.eps
                num = (pred[:, j] - target[:, j]).abs() * mj
                L_rel = L_rel + lam * num.sum() / denom.sum().clamp(min=1.0)
        return L_abs + self.lambda_rel * L_rel