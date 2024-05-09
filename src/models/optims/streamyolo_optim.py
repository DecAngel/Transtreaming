import torch
from torch import nn

from src.primitives.model import BaseOptim, BaseBackbone, BaseNeck, BaseHead


class StreamYOLOScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer, exp_steps: int, last_epoch=-1):
        super(StreamYOLOScheduler, self).__init__(
            optimizer,
            lambda steps: pow(steps/exp_steps, 2) if steps < exp_steps else 0.05,
            last_epoch=last_epoch
        )


class StreamYOLOOptim(BaseOptim):
    def __init__(
            self,
            lr: float = 0.001,
            momentum: float = 0.9,
            weight_decay: float = 5e-4,
            batch_size: int = 1,
    ):
        super().__init__()
        self.lr = lr / 64 * batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay

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
        optimizer = torch.optim.SGD(
            [
                {'params': p_backbone_normal},
                {'params': p_backbone_wd, 'weight_decay': self.weight_decay},
                {'params': p_neck_normal, 'lr': self.lr},
                {'params': p_neck_wd, 'lr': self.lr, 'weight_decay': self.weight_decay},
                {'params': p_head_normal, 'lr': self.lr},
                {'params': p_head_wd, 'lr': self.lr, 'weight_decay': self.weight_decay},
            ],
            lr=self.lr, momentum=self.momentum, nesterov=True
        )

        scheduler = StreamYOLOScheduler(
            optimizer,
            int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'SGD_lr'}]
