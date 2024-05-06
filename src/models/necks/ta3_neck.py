import math
from typing import Tuple, Literal, Optional

import torch
from positional_encodings.torch_encodings import PositionalEncoding3D
from torch import nn as nn
from torch.nn import functional as F

from ravt.systems.yolox.blocks.layers.network_blocks import BaseConv
from ravt.systems.yolox.blocks.types import BaseNeck, PYRAMID, TIME


class TA3Neck(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            neck_act_type: Literal['none', 'softmax', 'relu', 'elu', '1lu'] = 'none',
            **kwargs
    ):
        super().__init__()
        self.attn_size = (8, 8)
        self.attn_channel = in_channels[0]
        self.attn_max_T = 30

        tpe = PositionalEncoding3D(self.attn_channel)
        self.register_buffer(
            'tpe', tpe(torch.zeros(1, self.attn_max_T, *self.attn_size, self.attn_channel)), persistent=False
        )

        self.fc_in_list = nn.ModuleList([
            BaseConv(c, self.attn_channel, 1, 1) for c in in_channels
        ])
        self.fc_out_list = nn.ModuleList([
            BaseConv(c, c, 1, 1) for c in in_channels
        ])
        self.conv_in_list = nn.ModuleList([
            BaseConv(c, c, 1, 1) for c in in_channels
        ])
        self.fc_query = nn.Linear(self.attn_channel, self.attn_channel)
        self.fc_key = nn.Linear(self.attn_channel, self.attn_channel)
        if neck_act_type == 'none':
            self.attn_act = nn.Identity()
        elif neck_act_type == 'softmax':
            self.attn_act = nn.Softmax(dim=-1)
        elif neck_act_type == 'relu':
            self.attn_act = nn.ReLU()
        elif neck_act_type == 'elu':
            self.attn_act = lambda x: 1 + F.elu(x)
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

        tpe_p = []
        tpe_f = []
        for ptc, ftc in zip(past_time_constant, future_time_constant):
            tpe_p.append(-self.tpe[:, (-ptc).long()])
            tpe_f.append(self.tpe[:, ftc.long()])
        tpe_p = torch.cat(tpe_p, dim=0)     # B TP HA WA CA
        tpe_f = torch.cat(tpe_f, dim=0)     # B TF HA WA CA

        features_p = [
            F.interpolate(
                conv_in(f.flatten(0, 1)), size=self.attn_size, mode='bilinear'
            ).unflatten(0, (B, TP + 1))
            for f, conv_in in zip(features, self.conv_in_list)
        ]   # B TP_1 CA HA WA

        feature_attn = sum([
            fc_in(f.flatten(0, 1)).unflatten(0, (B, TP + 1)).permute(0, 1, 3, 4, 2)
            for f, fc_in in zip(features_p, self.fc_in_list)
        ])   # B TP_1 HA WA CA

        attn_query = self.fc_query((feature_attn[:, -1:].expand(-1, TF, -1, -1, -1) + tpe_f).flatten(1, 3)) # B TFHAWA CA
        attn_key = self.fc_key((feature_attn[:, :-1] + tpe_p).flatten(1, 3))                                # B TPHAWA CA
        attn_values = [
            (f[:, -1:] - f[:, :-1]).permute(0, 1, 3, 4, 2).flatten(1, 3)
            for f in features_p
        ]   # B TPHAWA C

        attn_weight = attn_query @ attn_key.transpose(-2, -1) / math.sqrt(attn_query.size(1))
        attn_weight = self.attn_act(attn_weight)                                                    # B TFHAWA TPHAWA

        attn_results = [
            (attn_weight @ v).unflatten(1, (TF, *self.attn_size)).permute(0, 1, 4, 2, 3)
            for v in attn_values
        ]   # B TF C HA WA

        outputs = []
        for ar, hw, f in zip(attn_results, HWs, features):
            attn_a = F.interpolate(ar.flatten(0, 1), size=hw, mode='bilinear').unflatten(0, (B, TF))    # B TF C H W
            outputs.append(f[:, -1:] + attn_a)

        return tuple(outputs)
