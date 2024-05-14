from typing import Tuple, Optional

import torch
from torch import nn

from src.models.layers.network_blocks import BaseConv
from src.primitives.batch import PYRAMID, TIME
from src.primitives.model import BaseNeck


class LongShortNeck(BaseNeck):
    state_dict_replace = [
        ('long_backbone.group_0_jian2', 'neck.long_convs.0'),
        ('long_backbone.group_0_jian1', 'neck.long_convs.1'),
        ('long_backbone.group_0_jian0', 'neck.long_convs.2'),
        ('short_backbone.group_0_jian2', 'neck.short_convs.0'),
        ('short_backbone.group_0_jian1', 'neck.short_convs.1'),
        ('short_backbone.group_0_jian0', 'neck.short_convs.2'),
        ('jian2', 'neck.long_2_convs.0'),
        ('jian1', 'neck.long_2_convs.1'),
        ('jian0', 'neck.long_2_convs.2'),
    ]
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
    ):
        """Implement LongShortNet with N=3, delta=1, LSFM-Lf-Dil. (The best setting)

        :param in_features: The channels of FPN features.
        """
        super().__init__()
        self.short_convs = nn.ModuleList([
            BaseConv(f, f // 2, ksize=1, stride=1)
            for f in in_channels
        ])
        self.long_convs = nn.ModuleList([
            BaseConv(f, f // 6, ksize=1, stride=1)
            for f in in_channels
        ])
        self.long_2_convs = nn.ModuleList([
            BaseConv((f // 6) * 3, f - (f // 2), ksize=1, stride=1)
            for f in in_channels
        ])

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ) -> PYRAMID:
        B, TP, _, _, _ = features[0].size()
        _, TF = future_clip_ids.size()

        assert TP == 4
        outputs = []
        for i, f in enumerate(features):
            l3, l2, l1, short = f.unbind(1)
            short = self.short_convs[i](short)
            l1 = self.long_convs[i](l1)
            l2 = self.long_convs[i](l2)
            l3 = self.long_convs[i](l3)
            long = torch.cat([l1, l2, l3], dim=1)
            long = self.long_2_convs[i](long)
            outputs.append((torch.cat([short, long], dim=1) + f[:, -1]).unsqueeze(1))

        return tuple(outputs)
