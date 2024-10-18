from typing import Tuple, Optional, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from src.models.layers.network_blocks import BaseConv
from src.primitives.batch import PYRAMID, TIME
from src.primitives.model import BaseNeck


class SimNeck2(BaseNeck):
    input_frames: int = 2
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            sim_stride: int = 1,
            sim_radius: int = 4,
    ):
        """High-level-feature-similarity based temporal fusion and forecasting

        :param in_channels: The channels of FPN features.
        """
        super().__init__()
        self.in_channels = in_channels
        self.sim_stride = sim_stride
        self.sim_radius = sim_radius

        self.convs = nn.ModuleList([
            BaseConv(c, c, ksize=1, stride=1)
            for c in in_channels
        ])

        sim_channels = (2 * self.sim_radius + 1) ** 2
        self.sim_mlp = nn.Linear(sim_channels, 1, bias=False)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def get_sim(
            self,
            feature1: Float[torch.Tensor, '*batch_time channels height width'],
            feature2: Float[torch.Tensor, '*batch_time channels height width'],
    ):
        B, T, C, H, W = feature1.shape
        f1 = feature1.flatten(0, 2).unsqueeze(0)    # 1, B*T*C
        f2 = feature2.flatten(0, 2).unsqueeze(1)    # B*T*C, 1

        # 1, B*T*C, 2*radius+1, 2*radius+1
        sim = F.conv2d(f1, f2, bias=None, stride=self.sim_stride, padding=self.sim_radius * self.sim_stride, groups=B * T * C)

        return sim.reshape(B, T, C, 2 * self.sim_radius + 1, 2 * self.sim_radius + 1)

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ) -> PYRAMID:
        B, TP, _, _, _ = features[0].size()
        _, TF = future_clip_ids.size()

        proj_features = [conv(f.flatten(0, 1)).unflatten(0, (B, TP)) for conv, f in zip(self.convs, features)]

        # calculate the similarity of highest features along dimension T
        # B, TP-1
        sim_past_clip_ids = past_clip_ids[..., 1:] - past_clip_ids[..., :-1]
        # B, TP-1, C, 2R+1, 2R+1
        sim_features = [
            self.get_sim(f[:, 1:], f[:, :-1])
            for f in proj_features
        ]
        # B, TP-1, C, (2R+1)^2
        sim_features = [
            F.softmax(sf.flatten(3, 4), dim=-1)
            for sf in sim_features
        ]

        # need assertion to pass
        assert TP == 2 and TF == 1
        # B, 1, C
        sim_features = [
            self.sim_mlp(sf).squeeze(-1) * future_clip_ids.unsqueeze(-1) / sim_past_clip_ids.unsqueeze(-1)
            for sf in sim_features
        ]

        # B, 1, C, H, W
        diff_features = [
            f[:, -1:] - f[:, :-1]
            for f in proj_features
        ]

        # B, 1, C, H, W
        outputs = [
            f[:, -1:] + sf.unsqueeze(-1).unsqueeze(-1) + df
            for f, sf, df in zip(features, sim_features, diff_features)
        ]
        return tuple(outputs)


class SimNeck(BaseNeck):
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            use_sim: bool = True,
            sim_stride: int = 1,
            sim_radius: int = 4,
            use_diff: bool = True,
    ):
        """High-level-feature-similarity based temporal fusion and forecasting

        :param in_channels: The channels of FPN features.
        """
        super().__init__()
        self.in_channels = in_channels
        self.sim_stride = sim_stride
        self.sim_radius = sim_radius

        self.convs = nn.ModuleList([
            BaseConv(c, c, ksize=1, stride=1)
            for c in in_channels
        ])

        # sim_channels = (2 * self.sim_radius + 1) ** 2
        # self.sim_mlp = nn.Linear(sim_channels, 1, bias=False)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def get_diff(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ) -> PYRAMID:
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()
        assert TF == 1

        diff = [
            torch.sum((f[:, -1:] - f[:, :-1]) / -past_clip_ids[:, :-1], dim=1, keepdim=True)
            for f in features
        ]
        return tuple(diff)

    def get_sim(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ):
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()
        assert TF == 1

        for f in features:
            B, T, C, H, W = f.shape

            # 1, B*T-1*C, H, W
            f1 = f[:, -1:].expand(B, T-1, C, H, W).flatten(0, 2).unsqueeze(0)
            # B*T-1*C, 1, H, W
            f2 = f[:, :-1].flatten(0, 2).unsqueeze(1)
            # 1, B*T-1*C, 2*radius+1, 2*radius+1
            sim = F.conv2d(f1, f2, bias=None, stride=self.sim_stride, padding=self.sim_radius * self.sim_stride, groups=B * T * C)
            # B, T-1, C, (2*radius+1)**2
            sim.reshape(B, T, C, (2 * self.sim_radius + 1) ** 2)
            sim = F.softmax(sim, dim=-1)
            #
            self.sim_mlp(sim).squeeze(-1) * future_clip_ids.unsqueeze(-1) / sim_past_clip_ids.unsqueeze(-1)

    def get_sim(
            self,
            feature1: Float[torch.Tensor, '*batch_time channels height width'],
            feature2: Float[torch.Tensor, '*batch_time channels height width'],
            sim_stride: int = 1,
            sim_radius: int = 4,
    ):
        B, T, C, H, W = feature1.shape
        f1 = feature1.flatten(0, 2).unsqueeze(0)    # 1, B*T*C
        f2 = feature2.flatten(0, 2).unsqueeze(1)    # B*T*C, 1

        # 1, B*T*C, 2*radius+1, 2*radius+1
        sim = F.conv2d(f1, f2, bias=None, stride=sim_stride, padding=sim_radius * sim_stride, groups=B * T * C)

        return sim.reshape(B, T, C, 2 * sim_radius + 1, 2 * sim_radius + 1)

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ) -> PYRAMID:
        B, TP, _, _, _ = features[0].size()
        _, TF = future_clip_ids.size()

        proj_features = [conv(f.flatten(0, 1)).unflatten(0, (B, TP)) for conv, f in zip(self.convs, features)]

        """
        # calculate the similarity of highest features along dimension T
        # B, TP-1
        sim_past_clip_ids = past_clip_ids[..., 1:] - past_clip_ids[..., :-1]
        # B, TP-1, C, 2R+1, 2R+1
        sim_features = [
            self.get_sim(f[:, 1:], f[:, :-1])
            for f in proj_features
        ]
        # B, TP-1, C, (2R+1)^2
        sim_features = [
            F.softmax(sf.flatten(3, 4), dim=-1)
            for sf in sim_features
        ]

        # need assertion to pass
        assert TP == 2 and TF == 1
        # B, 1, C
        sim_features = [
            self.sim_mlp(sf).squeeze(-1) * future_clip_ids.unsqueeze(-1) / sim_past_clip_ids.unsqueeze(-1)
            for sf in sim_features
        ]
        """

        # B, 1, C, H, W
        diff_features = [
            f[:, -1:] - f[:, :-1]
            for f in proj_features
        ]

        """
        # B, 1, C, H, W
        outputs = [
            f[:, -1:] + sf.unsqueeze(-1).unsqueeze(-1) + df
            for f, sf, df in zip(features, sim_features, diff_features)
        ]
        """
        # B, 1, C, H, W
        outputs = [
            f[:, -1:] + df
            for f, df in zip(features, diff_features)
        ]
        return tuple(outputs)

