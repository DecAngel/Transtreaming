import math
from typing import Tuple, Literal, Optional

import torch
from torch import nn as nn
from torch.nn import functional as F

from ..layers.network_blocks import BaseConv
from ..types import BaseNeck, PYRAMID, TIME


class TA2Neck(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            neck_act_type: Literal['none', 'softmax', 'relu', 'elu', '1lu'] = 'none',
            **kwargs
    ):
        super().__init__()
        self.attn_size = (8, 8)
        self.attn_channel = in_channels[0]

        self.fc_tc = nn.Sequential(
            nn.Linear(1, self.attn_channel),
            nn.Tanh(),
        )
        self.fc_in_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, self.attn_channel),
            )
            for c in in_channels
        ])
        self.fc_out_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, c),
            ) for c in in_channels
        ])
        self.max_pool = nn.AdaptiveMaxPool2d(self.attn_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(self.attn_size)
        self.conv_in_list = nn.ModuleList([
            BaseConv(c, c, 1, 1) for c in in_channels
        ])
        self.fc_query = nn.Linear(self.attn_channel, self.attn_channel)
        self.fc_key = nn.Linear(self.attn_channel, self.attn_channel)
        if neck_act_type == 'none':
            self.attn_act = nn.Identity()
        elif neck_act_type == 'softmax':
            self.attn_act = lambda x: F.softmax(x*10, dim=-1)
        elif neck_act_type == 'relu':
            self.attn_act = lambda x: F.relu(x - 1)
        elif neck_act_type == 'elu':
            self.attn_act = lambda x: 0.333 + F.elu(x, alpha=0.333)
        elif neck_act_type == '1lu':
            self.attn_act = lambda x: 1 + F.elu(x - 1)
        else:
            raise ValueError(f'act_type {neck_act_type} not supported!')

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None,
    ) -> PYRAMID:
        if past_time_constant is None:
            TP = features[0].size(1) - 1
            past_time_constant = torch.arange(
                -TP, -1, step=1, dtype=torch.float32, device=features[0].device
            )[None, ...]
        else:
            TP = past_time_constant.size(1)
        if future_time_constant is None:
            TF = 1
            future_time_constant = torch.tensor([[1]], dtype=torch.float32, device=features[0].device)
        else:
            TF = future_time_constant.size(1)
        B = features[0].size(0)
        HWs = [(f.size(3), f.size(4)) for f in features]

        tpe_p = 0.25 * self.fc_tc(past_time_constant[..., None])[:, None, None, ...]       # B 1 1 TP CA
        tpe_f = 0.25 * self.fc_tc(future_time_constant[..., None])[:, None, None, ...]     # B 1 1 TF CA

        features_p = [
            conv_in(f.flatten(0, 1)).unflatten(0, (B, TP + 1))
            for f, conv_in in zip(features, self.conv_in_list)
        ]   # B TP_1 C H W
        feature_attn = sum([
            fc_in(F.interpolate(f.flatten(0, 1), size=self.attn_size).permute(0, 2, 3, 1).contiguous())
            for f, fc_in in zip(features_p, self.fc_in_list)
        ]).unflatten(0, (B, TP + 1)).permute(0, 2, 3, 1, 4).contiguous()  # B HA WA TP_1 CA
        """
        feature_attn = sum([
            fc_in(torch.cat(
                [self.max_pool(f.flatten(0, 1)), self.avg_pool(f.flatten(0, 1))], dim=1
            ).permute(0, 2, 3, 1).contiguous())
            for f, fc_in in zip(features_p, self.fc_in_list)
        ]).unflatten(0, (B, TP + 1)).permute(0, 2, 3, 1, 4).contiguous()  # B HA WA TP_1 CA
        """
        attn_query = self.fc_query(feature_attn[:, :, :, -1:] + tpe_f).flatten(0, 2)  # BHAWA TF CA
        attn_key = self.fc_key(feature_attn[:, :, :, -1:] + tpe_p).flatten(0, 2)  # BHAWA TP CA
        attn_values = [
            (f[:, -1:] - f[:, :-1]).permute(0, 3, 4, 1, 2).contiguous()  # B H W TP C
            for f in features_p
        ]

        attn_weight = (attn_query @ attn_key.transpose(-2, -1)) / math.sqrt(attn_query.size(1))
        attn_weight = self.attn_act(attn_weight)  # BHAWA TF TP
        if self.training:
            self.vis_attn_weight = torch.mean(attn_weight.detach(), dim=0)
        attn_weight = attn_weight.unflatten(
            0, (B, *self.attn_size, 1)
        ).permute(0, 4, 5, 3, 1, 2).contiguous().flatten(0, 2)  # BTFTP 1 HA WA

        outputs = []
        for hw, v, f, fc_out in zip(HWs, attn_values, features, self.fc_out_list):
            attn_weight = F.interpolate(attn_weight, size=hw, mode='bilinear')    # BTFTP 1 H W
            w = attn_weight.squeeze(1).unflatten(0, (B, TF, TP)).permute(0, 3, 4, 1, 2).contiguous()  # B H W TF TP
            a = w @ v                                   # B H W TF C
            a = fc_out(a).permute(0, 3, 4, 1, 2).contiguous()   # B TF C H W
            outputs.append(f[:, -1:] + a)

        return tuple(outputs)
