from typing import Tuple, Optional

import torch
from torch import nn

from src.models.layers.network_blocks import BaseConv
from src.primitives.batch import PYRAMID, TIME, BatchDict
from src.primitives.model import BaseNeck


class DFPNeck(BaseNeck):
    state_dict_replace = [
        ('backbone.jian2', 'neck.convs.0'),
        ('backbone.jian1', 'neck.convs.1'),
        ('backbone.jian0', 'neck.convs.2'),
    ]
    input_frames: int = 2
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            BaseConv(f, f // 2, ksize=1, stride=1)
            for f in in_channels
        ])

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        B, TP = batch['past_clip_ids'].size()
        _, TF = batch['future_clip_ids'].size()

        outputs = []
        for i, conv in enumerate(self.convs):
            features_conv = conv(features[i].flatten(0, 1)).unflatten(0, (B, TP)).flip(1)
            features_conv = torch.cat([features_conv[:, :1].expand(-1, TP-1, -1, -1, -1), features_conv[:, 1:]], dim=2)
            outputs.append(features[i][:, -1:]+features_conv)

        batch['intermediate']['features_f'] = tuple(outputs)
        return batch
