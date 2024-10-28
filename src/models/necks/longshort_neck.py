from typing import Tuple, Optional, Literal

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
        ('long_backbone.group_1_jian2', 'neck.long_convs_r.0'),
        ('long_backbone.group_1_jian1', 'neck.long_convs_r.1'),
        ('long_backbone.group_1_jian0', 'neck.long_convs_r.2'),
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
            remap: bool = True,
            residue: bool = False,
    ):
        """Implement LongShortNet with N=3, delta=1, LSFM-Lf-Dil. (The best setting)

        :param in_features: The channels of FPN features.
        :param remap: align long_convs channels with short_convs
        :param residue: add long_convs_r
        """
        super().__init__()
        self.in_channels = in_channels
        self.remap = remap
        self.residue = residue

        self.short_convs = nn.ModuleList([
            BaseConv(f, f // 2, ksize=1, stride=1)
            for f in in_channels
        ])
        self.long_convs = nn.ModuleList([
            BaseConv(f, f // 6, ksize=1, stride=1)
            for f in in_channels
        ])
        if self.residue:
            self.long_convs_r = nn.ModuleList([
                BaseConv(f, f - f // 2 - 2 * (f // 6), ksize=1, stride=1)
                for f in in_channels
            ])

        if self.remap:
            self.long_2_convs = nn.ModuleList([
                BaseConv(f - f // 2 if self.residue else (f // 6) * 3, f - (f // 2), ksize=1, stride=1)
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

        assert TP == 4
        outputs = []
        for i, f in enumerate(features):
            l3, l2, l1, short = f.unbind(1)
            short = self.short_convs[i](short)
            l1 = self.long_convs[i](l1)
            l2 = self.long_convs[i](l2)
            if self.residue:
                l3 = self.long_convs_r[i](l3)
            else:
                l3 = self.long_convs[i](l3)
            long = torch.cat([l1, l2, l3], dim=1)
            if self.remap:
                long = self.long_2_convs[i](long)
            outputs.append((torch.cat([short, long], dim=1) + f[:, -1]).unsqueeze(1))

        batch['intermediate']['features_f'] = tuple(outputs)
        return batch
