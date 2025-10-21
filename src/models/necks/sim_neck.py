import contextlib
import functools
from typing import Tuple, Optional, Literal, List

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from src.models.layers.pafpn.network_blocks import BaseConv
from src.primitives.batch import PYRAMID, TIME, BatchDict
from src.primitives.model import BaseNeck
from src.models.layers.sim_parts import Sim, Sim2, Sim3, Diff, Diff2, LS, Corr, Corr2, Enhance


class SimNeck(BaseNeck):
    state_dict_replace = [
        ('long_backbone.group_0_jian2', 'neck.ls_long_convs.0'),
        ('long_backbone.group_0_jian1', 'neck.ls_long_convs.1'),
        ('long_backbone.group_0_jian0', 'neck.ls_long_convs.2'),
        ('long_backbone.group_1_jian2', 'neck.ls_long_convs_r.0'),
        ('long_backbone.group_1_jian1', 'neck.ls_long_convs_r.1'),
        ('long_backbone.group_1_jian0', 'neck.ls_long_convs_r.2'),
        ('short_backbone.group_0_jian2', 'neck.ls_short_convs.0'),
        ('short_backbone.group_0_jian1', 'neck.ls_short_convs.1'),
        ('short_backbone.group_0_jian0', 'neck.ls_short_convs.2'),
        ('jian2', 'neck.ls_long_2_convs.0'),
        ('jian1', 'neck.ls_long_2_convs.1'),
        ('jian0', 'neck.ls_long_2_convs.2'),
    ]
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            in_strides: Tuple[int, ...],
            sim_stride: int = 1,
            sim_radius: int = 4,
            sim_hidden: int = 4,
            corr_kernel: int = 4,
            corr_patch: int = 4,
            ls_remap: bool = False,
            ls_residue: bool = False,
            use_sim: bool = False,
            use_sim2: bool = False,
            use_sim3: bool = False,
            use_diff: bool = False,
            use_diff2: bool = False,
            use_ls: bool = False,
            use_corr: bool = False,
            use_corr2: bool = False,
            use_enhance: bool = False,
    ):
        """High-level-feature-similarity based temporal fusion and forecasting

        :param in_channels: The channels of FPN features.
        """
        super().__init__()
        self.parts = []
        if use_sim:
            self.sim = Sim(in_channels, in_strides, sim_stride, sim_radius)
            self.parts.append(self.sim)
        if use_sim2:
            self.sim2 = Sim2(in_channels, in_strides, sim_stride, sim_radius, sim_hidden)
            self.parts.append(self.sim2)
        if use_sim3:
            self.sim3 = Sim3(in_channels, in_strides, sim_stride, sim_radius)
            self.parts.append(self.sim3)
        if use_diff:
            self.diff = Diff(in_channels)
            self.parts.append(self.diff)
        if use_diff2:
            self.diff2 = Diff2(in_channels)
            self.parts.append(self.diff2)
        if use_ls:
            self.ls = LS(in_channels, ls_remap, ls_residue)
            self.parts.append(self.ls)
        if use_corr:
            self.corr = Corr(in_channels, in_strides, corr_kernel, corr_patch)
            self.parts.append(self.corr)
        if use_corr2:
            self.corr2 = Corr2(in_channels, in_strides, corr_kernel, corr_patch)
            self.parts.append(self.corr2)
        if use_enhance:
            self.enhance = Enhance(in_channels)
            self.parts.append(self.enhance)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        past_clip_ids = batch['past_clip_ids'].half()
        future_clip_ids = batch['future_clip_ids'].half()
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        # B, TF, C, H, W
        outputs = [f[:, -1:].expand(-1, TF, -1, -1, -1) for f in features]
        for part in self.parts:
            with self.record_time(f'neck_{part.__class__.__name__}'):
                outputs = [
                    f + df
                    for f, df in zip(
                        outputs,
                        part(features, past_clip_ids, future_clip_ids)
                    )
                ]

        batch['intermediate']['features_f'] = tuple(outputs)
        return batch
