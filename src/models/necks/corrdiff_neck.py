import contextlib
import functools
from typing import Tuple, Optional, Literal, List

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn
from torch.nn import ModuleList
import spatial_correlation_sampler as scs

from src.models.layers.pafpn.network_blocks import BaseConv
from src.primitives.batch import PYRAMID, TIME, BatchDict
from src.primitives.model import BaseNeck, BlockMixin
from src.utils.collection_operations import concat_pyramids

# from src.models.layers.sim_parts import Diff, Corr2, Sim2


'''
class CorrDiffNeck(BaseNeck):
    state_dict_replace = [
    ]
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            in_strides: Tuple[int, ...],
            corr_kernel: int = 1,
            corr_patch: int = 1,
    ):
        """High-level-feature-similarity based temporal fusion and forecasting

        :param in_channels: The channels of FPN features.
        """
        super().__init__()
        self.diff = Diff(in_channels)
        self.corr2 = Corr2(in_channels, in_strides, corr_kernel, corr_patch)
        # self.corr2 = Sim2(in_channels, in_strides)

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
        with self.record_time('neck_Diff'):
            features_1 = [
                f[:, -1:] + df
                for f, df in zip(
                    features,
                    self.diff(features, past_clip_ids, future_clip_ids)
                )
            ]
        with self.record_time('neck_Corr'):
            features_2 = [
                f + df
                for f, df in zip(
                    features_1,
                    self.corr2(features, past_clip_ids, future_clip_ids)
                )
            ]

        batch['intermediate']['features_f'] = tuple(features_2)
        return batch
'''


class Diff(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.diff_convs = nn.ModuleList([
            BaseConv(c, c, ksize=1, stride=1)
            for c in in_channels
        ])
        self.out_convs = nn.ModuleList([
            BaseConv(c, c, ksize=1, stride=1)
            for c in in_channels
        ])

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ):
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        # weights = F.softmax(past_clip_ids[:, :-1], dim=1).reshape(B, TP - 1, 1, 1, 1)
        proj_features = tuple([
            conv(f[:, -2:].flatten(0, 1)).unflatten(0, (B, 2))
            for conv, f in zip(self.diff_convs, features)
        ])

        diff = [
            conv(f[:, -1] - f[:, -2]).unsqueeze(1)
            for conv, f in zip(self.out_convs, proj_features)
        ]
        return tuple(diff)


class Corr(BlockMixin, nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            in_strides: Tuple[int, ...],
            corr_kernel: int = 4,
            corr_patch: int = 4,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self.corr_kernel = corr_kernel
        self.corr_patch = corr_patch
        self.relative_strides = [self.in_strides[-1] // s for s in self.in_strides]

        self.corr_sampler = scs.SpatialCorrelationSampler(
            kernel_size=2 * corr_kernel + 1,
            patch_size=2 * corr_patch + 1,
            stride=1,
            padding=corr_kernel,
            dilation=1,
            dilation_patch=1
        )
        self.corr_convs_1 = nn.ModuleList([
            BaseConv(c+(2*corr_patch+1)**2, c, ksize=1, stride=1)
            for c in self.in_channels
        ])
        self.corr_convs_2 = nn.ModuleList([
            BaseConv(c, c, ksize=1, stride=1)
            for c in self.in_channels
        ])

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ):
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        with torch.no_grad():
            with self.record_time('Corr2_pre'):
                weights = F.softmax(past_clip_ids[:, :-1], dim=1).reshape(B, TP - 1, 1, 1, 1).half()
                normalized_features = F.normalize(features[-1], dim=2)

            with self.record_time('Corr2_sample'):
                # B*(TP-1), Y, X, H, W
                correlation = self.corr_sampler(
                    normalized_features[:, 1:].flatten(0, 1).contiguous(),
                    normalized_features[:, :-1].flatten(0, 1).contiguous()
                )
                # B, TP-1, YX, H, W
                correlation = correlation.flatten(1, 2).unflatten(0, (B, TP - 1))
                # B, TP-1, YX, H, W
                correlation = F.softmax(correlation, dim=2)
                # B, YX, H, W
                correlation = torch.sum(correlation * weights, dim=1)

            with self.record_time('Corr2_interpolate'):
                # B, TF, YX, H, W
                correlations = [
                    F.interpolate(
                        correlation,
                        size=(f.size(-2), f.size(-1)),
                        mode='bilinear',
                        align_corners=True,
                    ).unsqueeze(1) * future_clip_ids.reshape(B, TF, 1, 1, 1)
                    for f in features
                ]

        outputs = []
        with self.record_time('Corr2_conv'):
            for f, corr, conv_1, conv_2 in zip(features, correlations, self.corr_convs_1, self.corr_convs_2):
                mix = torch.cat([f[:, -1:].expand(B, TF, -1, -1, -1), corr], dim=2)
                mix = conv_2(conv_1(mix.flatten(0, 1))).unflatten(0, (B, TF))
                outputs.append(mix)
            """
            for f, corr, conv_1 in zip(features, correlations, self.corr_convs_1):
                mix = torch.cat([f[:, -1:].expand(B, TF, -1, -1, -1), corr], dim=2)
                mix = conv_1(mix.flatten(0, 1)).unflatten(0, (B, TF))
                outputs.append(mix)
            """

        return tuple(outputs)


class CorrDiffNeck(BaseNeck):
    state_dict_replace = [
    ]
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            in_strides: Tuple[int, ...],
            corr_kernel: int = 1,
            corr_patch: int = 1,
    ):
        """High-level-feature-similarity based temporal fusion and forecasting

        :param in_channels: The channels of FPN features.
        """
        super().__init__()
        self.diff = Diff(in_channels)
        self.corr = Corr(in_channels, in_strides, corr_kernel, corr_patch)

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
        with self.record_time('neck_Diff'):
            feature_diff = self.diff(features, past_clip_ids, future_clip_ids)

        with self.record_time('neck_Corr'):
            feature_corr = self.corr(features, past_clip_ids, future_clip_ids)

        with self.record_time('neck_Combination'):
            outputs = [
                f[:, -1:] + f_d + f_c
                for f, f_d, f_c in zip(
                    features,
                    feature_diff,
                    feature_corr,
                )
            ]

        batch['intermediate']['features_f'] = tuple(outputs)
        return batch
