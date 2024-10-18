from typing import Tuple

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from src.primitives.model import BaseOptim, BaseBackbone, BaseNeck, BaseHead


class DETROptim(BaseOptim):
    def __init__(
            self,
            lr: float = 0.00001,
            betas: Tuple[float, float] = (0.9, 0.999),
            weight_decay: float = 0.0001,
            gamma: float = 0.1,
            neck_coef: float = 10.0,
            head_coef: float = 10.0,
    ):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.neck_coef = neck_coef
        self.head_coef = head_coef

    def configure_optimizers(
            self,
            backbone: BaseBackbone,
            neck: BaseNeck,
            head: BaseHead,
    ):
        p_backbone_normal, p_backbone_wd = [], []
        p_neck_normal, p_neck_wd = [], []
        p_head_normal, p_head_wd = [], []
        for module, pn, pw in zip(
                [backbone, neck, head],
                [p_backbone_normal, p_neck_normal, p_head_normal],
                [p_backbone_wd, p_neck_wd, p_head_wd],
        ):
            for k, v in module.named_modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                    pn.append(v.bias)
                if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)) or 'bn' in k:
                    pn.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pw.append(v.weight)  # apply decay
        optimizer = torch.optim.AdamW(
            [
                {'params': p_backbone_normal},
                {'params': p_backbone_wd, 'weight_decay': self.weight_decay},
                {'params': p_neck_normal, 'lr': self.lr * self.neck_coef},
                {'params': p_neck_wd, 'lr': self.lr * self.neck_coef, 'weight_decay': self.weight_decay},
                {'params': p_head_normal, 'lr': self.lr * self.head_coef},
                {'params': p_head_wd, 'lr': self.lr * self.head_coef, 'weight_decay': self.weight_decay},
            ],
            lr=self.lr, betas=self.betas,
        )

        scheduler = MultiStepLR(
            optimizer,
            milestones=[1000],
            gamma=self.gamma,
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'AdamW_lr'}]
