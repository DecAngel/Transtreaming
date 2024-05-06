from typing import Tuple, Optional, Literal, Union

import torch
from jaxtyping import Float
from kornia.geometry import remap
from torch import nn
import torch.nn.functional as F
from kornia.utils.grid import create_meshgrid

from ..layers.network_blocks import BaseConv
from ..types import PYRAMID, BaseNeck, TIME, YOLOXLossDict

MAX_H, MAX_W = 500, 500


class DeformNeck(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs,
    ):
        super().__init__()
        # channels and strides
        base_channel = in_channels[0]
        self.max_displacement = 4
        self.hidden_channel = 64

        # coordinates (1, H, W, 2)
        self.register_buffer('grid', create_meshgrid(MAX_H, MAX_W, False, dtype=torch.float32), persistent=False)

        self.flow_estimators_1 = nn.ModuleList([
            nn.Sequential(
                BaseConv(c + (2*self.max_displacement+1)**2+2, self.hidden_channel, 3, 1, bias=True)
            )
            for c in in_channels
        ])
        """
        self.flow_estimators_2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=1, padding=1, bias=True),
                    nn.LeakyReLU(0.1),
                )
                for c_in, c_out in zip([128, 256, 352, 416], [128, 96, 64, 32])
            ] + [nn.Conv2d(448, 2, kernel_size=(3, 3), stride=1, padding=1, bias=True)]
        )
        """
        self.flow_estimators_2 = nn.Sequential(
            BaseConv(self.hidden_channel, self.hidden_channel // 2, 3, 1, bias=True),
            nn.Conv2d(self.hidden_channel // 2, 2, kernel_size=(1, 1), stride=1, padding=0, bias=True),
        )
        self.register_buffer('smooth_kernel', torch.tensor(
            [[[[0, -1.0, 0], [-1.0, 4.0, -1.0], [0, -1.0, 0]]]]
        ), persistent=False)

    def warp(
            self,
            source: Float[torch.Tensor, 'batch_size channels height width'],
            flow: Float[torch.Tensor, 'batch_size flow=2 height width'],
    ):
        B, C, H, W = source.size()
        _, _, H2, W2 = flow.size()
        if H != H2 or W != W2:
            flow = self.scale_flow(flow, (H, W))
        grid = self.grid[:, :H, :W, :] + flow.permute(0, 2, 3, 1)
        return remap(source, grid[..., 0], grid[..., 1], mode='bilinear', align_corners=True)

    def scale_flow(self, flow: Float[torch.Tensor, 'batch_size flow=2 height width'], size: Tuple[int, int]):
        H, W = size
        _, _, H2, W2 = flow.size()
        flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=False)
        flow[:, 0] *= H / H2
        flow[:, 1] *= W / W2
        return flow

    def get_normalize_feature(self, feature: Float[torch.Tensor, 'batch_size channels height width']):
        std, mean = torch.std_mean(feature)
        return (feature - mean) / std

    def get_cost_volume(
            self,
            feature_1: Float[torch.Tensor, 'batch_size channels height width'],
            feature_2: Float[torch.Tensor, 'batch_size channels height width'],
            max_displacement: Optional[int] = None,
    ):
        max_displacement = max_displacement or self.max_displacement
        feature_1 = self.get_normalize_feature(feature_1)
        feature_2 = self.get_normalize_feature(feature_2)

        B, C, H, W = feature_1.size()
        feature_2_padded = F.pad(
            feature_2,
            pad=(max_displacement, max_displacement, max_displacement, max_displacement),
            mode='constant',
            value=0.0,
        )

        volumes = []
        for x in range(2 * max_displacement + 1):
            for y in range(2 * max_displacement + 1):
                volumes.append(torch.cosine_similarity(feature_1, feature_2_padded[:, :, y:y + H, x:x + W], dim=1))

        return torch.stack(volumes, dim=1)

    def charbonnier_loss(
            self,
            difference: torch.Tensor,
    ):
        return torch.mean((difference**2+0.001**2)**0.5)

    def get_hamming_loss(
            self,
            feature_1: Float[torch.Tensor, 'batch_size channels height width'],
            feature_2: Float[torch.Tensor, 'batch_size channels height width'],
            flow: Float[torch.Tensor, 'batch_size flow=2 height width']
    ):
        feature_1_warped = self.warp(feature_1, flow)
        return self.charbonnier_loss(feature_2 - feature_1_warped)

    def get_smooth_loss(
            self,
            flow: Float[torch.Tensor, 'batch_size flow=2 height width'],
    ):
        flows = torch.split(flow, 1, dim=1)
        gradient = torch.cat([F.conv2d(f, self.smooth_kernel) for f in flows], dim=1)
        return self.charbonnier_loss(gradient)

    def wcf_block(
            self,
            feature_1: Float[torch.Tensor, 'batch_size channels height width'],
            feature_2: Float[torch.Tensor, 'batch_size channels height width'],
            level: int,
            initial_flow: Optional[Float[torch.Tensor, 'batch_size flow=2 height_2 width_2']] = None,
    ) -> torch.Tensor:
        """ Calculate flow from feature_1 to feature_2

        :param feature_1:
        :param feature_2:
        :param level:
        :param initial_flow:
        :return:
        """
        B, C, H, W = feature_1.size()

        if initial_flow is not None:
            flow = self.scale_flow(initial_flow, (H, W))
            with torch.no_grad():
                feature_1_warped = self.warp(feature_1, flow)
        else:
            flow = torch.zeros(B, 2, H, W, dtype=feature_1.dtype, device=feature_1.device)
            feature_1_warped = feature_1

        with torch.no_grad():
            volume = self.get_cost_volume(feature_1_warped, feature_2)

        x = torch.cat([feature_1, volume, flow.detach()], dim=1)
        x = self.flow_estimators_1[level](x)    # B 128 H W
        """
        for fe2 in self.flow_estimators_2[:-1]:
            x = torch.cat([x, fe2(x)], dim=1)
        x = self.flow_estimators_2[-1](x)       # B 2 H W
        """
        x = self.flow_estimators_2(x)

        return flow + x

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None,
    ) -> Tuple[PYRAMID, YOLOXLossDict]:
        B, TP0, C, _, _ = features[0].size()
        LEVELS = len(features)
        TP = TP0 - 1
        past_time_constant = past_time_constant if past_time_constant is not None else torch.range(
            -TP, 1, dtype=features[0].dtype, device=features[0].dtype
        ).unsqueeze(0).expand(B, -1)
        future_time_constant = future_time_constant if future_time_constant is not None else torch.ones(
            1, dtype=features[0].dtype, device=features[0].dtype
        ).unsqueeze(0).expand(B, -1)
        TF = future_time_constant.size(1)
        deltas = torch.cat([past_time_constant[:, 1:] - past_time_constant[:, :-1], - past_time_constant[:, -1:]], dim=1)
        deltas[torch.abs(deltas) < 1e-5] = 1

        flow = None
        vis_flows = []
        hamming_losses = []
        smooth_losses = []
        for i in [TP - 1] * 1:
            delta = deltas[:, i, None, None, None]
            flow = flow * delta if flow is not None else None
            for level in reversed(range(LEVELS)):
                feature_1 = features[level][:, i+1]
                feature_2 = features[level][:, i]
                flow = self.wcf_block(feature_1, feature_2, level, flow)
                if self.training:
                    hamming_losses.append(self.get_hamming_loss(feature_1, feature_2, flow))
                    smooth_losses.append(self.get_smooth_loss(flow))
            flow = flow / delta
            vis_flows.append(flow)
        self.vis_flows = torch.stack(vis_flows, dim=1)

        hamming_loss = sum(hamming_losses)
        smooth_loss = sum(smooth_losses)

        # flow current -> past, thus take negative to make current -> future, then multiply by ftc
        flows = - flow.unsqueeze(1) * future_time_constant[..., None, None, None]
        flows = flows.flatten(0, 1)
        features_f = []
        for f in features:
            f = f[:, -1:].expand(-1, TF, -1, -1, -1).flatten(0, 1)  # B TF C H W
            warped = self.warp(f, flows)
            features_f.append(warped.unflatten(0, (B, TF)))
        return (
            tuple(features_f),
            {'hamming_loss': hamming_loss, 'smooth_loss': smooth_loss},
        )


class DeformNeck2(BaseNeck):
    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None,
    ) -> Tuple[PYRAMID, YOLOXLossDict]:
        B, TP0, C, _, _ = features[0].size()
        LEVELS = len(features)
        TP = TP0 - 1
        past_time_constant = past_time_constant if past_time_constant is not None else torch.range(
            -TP, 1, dtype=features[0].dtype, device=features[0].dtype
        ).unsqueeze(0).expand(B, -1)
        future_time_constant = future_time_constant if future_time_constant is not None else torch.ones(
            1, dtype=features[0].dtype, device=features[0].dtype
        ).unsqueeze(0).expand(B, -1)
        TF = future_time_constant.size(1)
        deltas = torch.cat([past_time_constant[:, 1:] - past_time_constant[:, :-1], - past_time_constant[:, -1:]], dim=1)
        deltas[torch.abs(deltas) < 1e-5] = 1

        flow = None
        vis_flows = []
        hamming_losses = []
        smooth_losses = []
        for i in [TP - 1] * 1:
            delta = deltas[:, i, None, None, None]
            flow = flow * delta if flow is not None else None
            for level in reversed(range(LEVELS)):
                feature_1 = features[level][:, i+1]
                feature_2 = features[level][:, i]
                flow = self.wcf_block(feature_1, feature_2, level, flow)
                if self.training:
                    hamming_losses.append(self.get_hamming_loss(feature_1, feature_2, flow))
                    smooth_losses.append(self.get_smooth_loss(flow))
            flow = flow / delta
            vis_flows.append(flow)
        self.vis_flows = torch.stack(vis_flows, dim=1)

        hamming_loss = sum(hamming_losses)
        smooth_loss = sum(smooth_losses)

        # flow current -> past, thus take negative to make current -> future, then multiply by ftc
        flows = - flow.unsqueeze(1) * future_time_constant[..., None, None, None]
        flows = flows.flatten(0, 1)
        features_f = []
        for f in features:
            f = f[:, -1:].expand(-1, TF, -1, -1, -1).flatten(0, 1)  # B TF C H W
            warped = self.warp(f, flows)
            features_f.append(warped.unflatten(0, (B, TF)))
        return (
            tuple(features_f),
            {'hamming_loss': hamming_loss, 'smooth_loss': smooth_loss},
        )