import functools
from typing import Tuple, Optional, Literal, List

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn
import spatial_correlation_sampler as scs


from src.models.layers.network_blocks import BaseConv
from src.primitives.batch import PYRAMID, TIME


def interpolate_features(
        features: PYRAMID,
        past_clip_ids: TIME,
        future_clip_ids: TIME,
) -> PYRAMID:
    B, TP = past_clip_ids.size()
    _, TF = future_clip_ids.size()

    with torch.no_grad():
        features = [f.detach() for f in features]
        pci = past_clip_ids.detach().int().cpu().numpy().tolist()
        fci = future_clip_ids.detach().int().cpu().numpy().tolist()
        indices = []
        for i, (p, f) in enumerate(zip(pci, fci)):
            p_pos = TP - 2
            for f_i in f:
                while p_pos >= 0:
                    if p_pos == 0:
                        indices.append(p_pos + i * TP)
                        break
                    elif f_i < -p[p_pos - 1]:
                        indices.append(p_pos + i * TP)
                        break
                    else:
                        p_pos -= 1
        outputs = [
            f.flatten(0, 1)[indices].unflatten(0, (B, TF))
            for f in features
        ]
    return tuple(outputs)


class Sim(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            in_strides: Tuple[int, ...],
            sim_stride: int = 1,
            sim_radius: int = 4,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self.sim_stride = sim_stride
        self.sim_radius = sim_radius

        self.relative_strides = [self.in_strides[-1] // s for s in self.in_strides]
        sim_channels = (2 * self.sim_radius + 1) ** 2

        self.sim_convs = nn.ModuleList([
            BaseConv(c, c, ksize=1, stride=1)
            for c in in_channels
        ])
        self.sim_mlp = nn.Linear(3 * sim_channels, 1, bias=False)

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ):
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()
        assert TP == 4 and TF == 1

        proj_features = tuple(
            [conv(f.flatten(0, 1)).unflatten(0, (B, TP)) for conv, f in zip(self.sim_convs, features)])

        sims = []
        for f, s in zip(proj_features, self.relative_strides):
            B, T, C, H, W = f.shape

            # 1, B*T-1*C, H, W
            f1 = f[:, -1:].expand(B, T - 1, C, H, W).flatten(0, 2).unsqueeze(0)
            # B*T-1*C, 1, H, W
            f2 = f[:, :-1].detach().flatten(0, 2).unsqueeze(1)
            # 1, B*T-1*C, 2*radius+1, 2*radius+1
            sim = F.conv2d(f1, f2, bias=None, stride=self.sim_stride * s, padding=self.sim_radius * self.sim_stride * s,
                           groups=B * (T - 1) * C)
            # B, T-1, C, (2*radius+1)**2
            sim = sim.reshape(B, T - 1, C, (2 * self.sim_radius + 1) ** 2)
            sim = F.softmax(sim, dim=-1)
            sim = sim * future_clip_ids.unsqueeze(-1).unsqueeze(-1) / -past_clip_ids[:, :-1].unsqueeze(-1).unsqueeze(-1)
            # B, C, T-1, (2*radius+1)**2
            sim = sim.permute(0, 2, 1, 3).contiguous().reshape(B, C, -1)
            # B, C
            sim = self.sim_mlp(sim).squeeze(-1)

            sims.append(sim.reshape(B, 1, C, 1, 1))
        return tuple(sims)


class Sim2(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            in_strides: Tuple[int, ...],
            sim_stride: int = 1,
            sim_radius: int = 4,
            sim_hidden: int = 4,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self.sim_stride = sim_stride
        self.sim_radius = sim_radius
        self.sim_hidden = sim_hidden

        self.relative_strides = [self.in_strides[-1] // s for s in self.in_strides]
        sim_channels = (2 * self.sim_radius + 1) ** 2

        self.register_buffer('sim2_kernel', torch.eye(sim_channels).reshape(sim_channels, 1, 2 * self.sim_radius + 1,
                                                                            2 * self.sim_radius + 1))
        self.sim2_conv_1 = BaseConv(3 * sim_channels, self.sim_hidden, ksize=1, stride=1)
        self.sim2_convs_2 = nn.ModuleList([
            BaseConv(c + self.sim_hidden, c, ksize=1, stride=1)
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

        _, _, C, H, W = features[-1].size()
        # B, TP-1, C, 1, H, W
        past_feature = features[-1][:, :-1].unsqueeze(3)
        # B*C, 1, H, W
        current_feature = features[-1][:, -1].reshape(B * C, 1, H, W)
        # B*C, (2R+1)**2, H, W
        shifted_feature = F.conv2d(current_feature, self.sim2_kernel, bias=None, stride=1, dilation=self.sim_stride,
                                   padding=self.sim_radius)
        # B, 1, C, (2R+1)**2, H, W
        shifted_feature = shifted_feature.reshape(B, 1, C, (2 * self.sim_radius + 1) ** 2, H, W)
        # B, TP-1, C, (2R+1)**2, H, W
        sim_feature = -torch.abs(shifted_feature - past_feature)
        # B, TP-1, (2R+1)**2, H, W
        sim_feature = F.softmax(torch.sum(sim_feature, dim=2), dim=2)
        # B, (TP-1)*(2R+1)**2, H, W
        sim_feature = sim_feature.reshape(B, (TP - 1) * ((2 * self.sim_radius + 1) ** 2), H, W)
        # B, Ch, H, W
        sim_feature = self.sim2_conv_1(sim_feature)

        sims = []
        for f, c in zip(features, self.sim2_convs_2):
            _, _, Ci, Hi, Wi = f.shape
            p = F.interpolate(sim_feature, size=(Hi, Wi), mode='bilinear', align_corners=True)
            sims.append(c(torch.cat([f[:, -1], p], dim=1)).unsqueeze(1))
        return tuple(sims)


class Sim3(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            in_strides: Tuple[int, ...],
            sim_stride: int = 1,
            sim_radius: int = 4,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self.sim_stride = sim_stride
        self.sim_radius = sim_radius

        self.relative_strides = [self.in_strides[-1] // s for s in self.in_strides]
        sim_channels = (2 * self.sim_radius + 1) ** 2

        self.sim3_convs = nn.ModuleList([
            BaseConv(c, c, ksize=1, stride=1)
            for c in in_channels
        ])
        self.sim3_mlp = nn.Linear(sim_channels, 1, bias=False)

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ):
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        proj_features = tuple([conv(f.flatten(0, 1)).unflatten(0, (B, TP)) for conv, f in zip(self.sim3_convs, features)])
        ipfs = interpolate_features(proj_features, past_clip_ids, future_clip_ids)

        sims = []
        for f, ipf, s in zip(proj_features, ipfs, self.relative_strides):
            _, _, C, H, W = f.shape

            # 1, B*TF*C, H, W
            f1 = f[:, -1:].expand(-1, TF, -1, -1, -1).flatten(0, 2).unsqueeze(0)
            # B*TF*C, 1, H, W
            f2 = ipf.flatten(0, 2).unsqueeze(1)
            # 1, B*TF*C, 2*radius+1, 2*radius+1
            sim = F.conv2d(f1, f2, bias=None, stride=self.sim_stride*s, padding=self.sim_radius*self.sim_stride*s, groups=B * TF * C)
            # B, TF, C, (2*radius+1)**2
            sim = sim.reshape(B, TF, C, (2 * self.sim_radius + 1) ** 2)
            sim = F.softmax(sim, dim=-1)
            # B, TF, C
            sim = self.sim3_mlp(sim).squeeze(-1)
            # B, TF, C, 1, 1
            sims.append(sim.unsqueeze(-1).unsqueeze(-1))
        return tuple(sims)


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
            f[:, -1:] - f[:, :-1]
            for f in proj_features
        ]
        return tuple(diff)


class Diff2(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.diff2_convs = nn.ModuleList([
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

        proj_features = tuple([conv(f.flatten(0, 1)).unflatten(0, (B, TP)) for conv, f in zip(self.diff2_convs, features)])
        ipfs = interpolate_features(proj_features, past_clip_ids, future_clip_ids)

        diff = [(
            f[:, -1:] - ipf
        ) for f, ipf in zip(proj_features, ipfs)]
        return tuple(diff)


class LS(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            ls_remap: bool = False,
            ls_residue: bool = False,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ls_remap = ls_remap
        self.ls_residue = ls_residue

        self.ls_short_convs = nn.ModuleList([
            BaseConv(f, f // 2, ksize=1, stride=1)
            for f in in_channels
        ])
        self.ls_long_convs = nn.ModuleList([
            BaseConv(f, f // 6, ksize=1, stride=1)
            for f in in_channels
        ])
        if self.ls_residue:
            self.ls_long_convs_r = nn.ModuleList([
                BaseConv(f, f - f // 2 - 2 * (f // 6), ksize=1, stride=1)
                for f in in_channels
            ])
        if self.ls_remap:
            self.ls_long_2_convs = nn.ModuleList([
                BaseConv(f - f // 2 if self.ls_residue else (f // 6) * 3, f - (f // 2), ksize=1, stride=1)
                for f in in_channels
            ])

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ):
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()
        assert TP == 4 and TF == 1

        ls = []
        for i, f in enumerate(features):
            l3, l2, l1, short = f.unbind(1)
            short = self.ls_short_convs[i](short)
            l1 = self.ls_long_convs[i](l1)
            l2 = self.ls_long_convs[i](l2)
            if self.ls_residue:
                l3 = self.ls_long_convs_r[i](l3)
            else:
                l3 = self.ls_long_convs[i](l3)
            long = torch.cat([l1, l2, l3], dim=1)
            if self.ls_remap:
                long = self.ls_long_2_convs[i](long)
            ls.append((torch.cat([short, long], dim=1)).unsqueeze(1))

        return tuple(ls)


class Corr(nn.Module):
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
        self.corr_convs = nn.ModuleList([
            BaseConv(c+2, c, ksize=1, stride=1)
            for c in self.in_channels
        ])
        # (2*corr_kernel+1)**2, 2
        self.register_buffer('corr_directions', torch.stack(torch.meshgrid(
            torch.arange(-corr_kernel, corr_kernel + 1, dtype=torch.float32),
            torch.arange(-corr_kernel, corr_kernel + 1, dtype=torch.float32),
        ), dim=-1).flatten(0, 1))

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ):
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        with torch.no_grad():
            delta_past_clip_ids = past_clip_ids[:, 1:] - past_clip_ids[:, :-1]

            normalized_features = F.normalize(features[0], dim=2)

            # B*(TP-1), Y, X, H, W
            correlation = self.corr_sampler(normalized_features[:, :-1].flatten(0, 1), normalized_features[:, 1:].flatten(0, 1))
            # B, TP-1, YX, H, W
            correlation = torch.softmax(correlation.flatten(1, 2).unflatten(0, (B, TP - 1)), dim=2)
            # B, TP-1, 2, H, W
            correlation = torch.einsum('btxhw, xc -> btchw', correlation, self.corr_directions)
            # B, TP-1, 2, H, W
            correlation = correlation / delta_past_clip_ids.reshape(B, TP - 1, 1, 1, 1)
            # B, 2, H, W
            correlation = torch.mean(correlation, dim=1)
            # B, TF, 2, H, W
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
        for f, corr, conv in zip(features, correlations, self.corr_convs):
            mix = torch.cat([f[:, -1:].expand(B, TF, -1, -1, -1), corr], dim=2)
            mix = conv(mix.flatten(0, 1)).unflatten(0, (B, TF))
            outputs.append(mix)

        return tuple(outputs)


class Corr2(nn.Module):
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

        self.corr2_sampler = scs.SpatialCorrelationSampler(
            kernel_size=2 * corr_kernel + 1,
            patch_size=2 * corr_patch + 1,
            stride=1,
            padding=corr_kernel,
            dilation=1,
            dilation_patch=1
        )
        self.corr2_convs = nn.ModuleList([
            BaseConv(c+(2*corr_kernel+1)**2, c, ksize=1, stride=1)
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
            weights = F.softmax(past_clip_ids[:, :-1], dim=1).reshape(B, TP - 1, 1, 1, 1).half()
            normalized_features = F.normalize(features[0], dim=2)

            # B*(TP-1), Y, X, H, W
            correlation = self.corr2_sampler(normalized_features[:, :-1].flatten(0, 1), normalized_features[:, 1:].flatten(0, 1))
            # B, TP-1, YX, H, W
            correlation = correlation.flatten(1, 2).unflatten(0, (B, TP - 1))
            # B, YX, H, W
            correlation = torch.sum(correlation * weights, dim=1)
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
        for f, corr, conv in zip(features, correlations, self.corr2_convs):
            mix = torch.cat([f[:, -1:].expand(B, TF, -1, -1, -1), corr], dim=2)
            mix = conv(mix.flatten(0, 1)).unflatten(0, (B, TF))
            outputs.append(mix)

        return tuple(outputs)


class Enhance(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.enhance_convs = nn.ModuleList([
            BaseConv(2*c, c, ksize=1, stride=1)
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

        enhance_features = [
            conv(f[:, -2:].flatten(1, 2)).unsqueeze(1)
            for f, conv in zip(features, self.enhance_convs)
        ]
        return tuple(enhance_features)
