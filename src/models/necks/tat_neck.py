import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
import torch.nn.functional as F

from src.primitives.batch import PYRAMID, TIME, BatchDict
from src.primitives.model import BaseNeck


def pyramid2windows(
        pyramid: PYRAMID,
        window_size: Tuple[int, int],
        shift: bool = False,
) -> Tuple[Float[torch.Tensor, 'batch n length channel'], List[Tuple[int, ...]]]:
    dH, dW = window_size
    windows = []
    params = []
    for features in pyramid:
        B, T, H, W, C = features.size()
        nH, nW = math.ceil(H / dH), math.ceil(W / dW)
        pH, pW = nH * dH - H, nW * dW - W

        # pad
        features = F.pad(features, (0, 0, 0, pW, 0, pH))

        # shift
        if shift:
            features = torch.roll(features, shifts=(dH // 2, dW // 2), dims=(2, 3))

        # partition
        window = features.reshape(B, T, nH, dH, nW, dW, C)
        window = window.permute(0, 2, 4, 1, 3, 5, 6).flatten(3, 5).flatten(1, 2)  # B nHnW TdHdW C
        windows.append(window)
        params.append((nH, nW, B, T, H, W, C))
    return torch.cat(windows, dim=1).contiguous(), params


def windows2pyramid(
        windows: Float[torch.Tensor, 'batch n length channel'],
        params: List[Tuple[int, ...]],
        window_size: Tuple[int, int],
        shift: bool = False,
):
    dH, dW = window_size
    pyramid = []
    for window, (nH, nW, B, T, H, W, C) in zip(torch.split(windows, [p[0]*p[1] for p in params], dim=1), params):
        window = window.unflatten(1, (nH, nW)).unflatten(3, (T, dH, dW))
        features = window.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, T, nH * dH, nW * dW, C)

        # shift
        if shift:
            features = torch.roll(features, shifts=(- (dH // 2), - (dW // 2)), dims=(2, 3))

        # pad
        pyramid.append(features[..., :H, :W, :].contiguous())
    return tuple(pyramid)


class RelativePositionalEncoding3D(nn.Module):
    def __init__(
            self,
            window_size: Tuple[int, int],
            num_heads: int,
            rpe_coef: float = 16,
    ) -> None:
        super(RelativePositionalEncoding3D, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.rpe_coef = rpe_coef
        T = 50
        H, W = window_size
        rct_coef = 8

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_t = torch.arange(0, 2*T-1, dtype=torch.float32)  # 2*T-1
        relative_coords_h = torch.arange(-H+1, H, dtype=torch.float32)  # 2*H-1
        relative_coords_w = torch.arange(-W+1, W, dtype=torch.float32)  # 2*W-1

        relative_coords_table = torch.stack(
            torch.meshgrid([
                relative_coords_t,
                relative_coords_h,
                relative_coords_w])
        ).permute(1, 2, 3, 0).contiguous()  # 2*T-1, 2*H-1, 2*W-1, 3
        relative_coords_table[:, :, :, 0] /= 2*T-1
        relative_coords_table[:, :, :, 1] /= H-1
        relative_coords_table[:, :, :, 2] /= W-1

        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) * rct_coef + 1.0
        ) / np.log2(rct_coef)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_f = torch.stack(torch.meshgrid([torch.arange(T), torch.arange(H), torch.arange(W)]))  # 3, TF, H, W
        coords_p = torch.stack(torch.meshgrid([-torch.arange(T), torch.arange(-H+1, 1), torch.arange(-W+1, 1)]))  # 3, TP, H, W
        coords_flatten_f = torch.flatten(coords_f, 1)  # 3, TFHW
        coords_flatten_p = torch.flatten(coords_p, 1)  # 3, TPHW
        relative_coords = coords_flatten_f[:, :, None] - coords_flatten_p[:, None, :]  # 3, TFHW, TPHW
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # TFHW, TPHW, 3
        relative_coords[:, :, 0] *= (2 * H - 1)*(2 * W - 1)
        relative_coords[:, :, 1] *= 2 * W - 1
        relative_position_index = relative_coords.sum(-1).view(T, H*W, T, H*W)  # TF, HW, TP, HW
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.cached_rpt = None

    def forward(self, ptc: TIME, ftc: TIME) -> Float[torch.Tensor, 'batch head TFdHdW TPdHdW']:
        ptc = (-ptc).long()
        ftc = ftc.long()
        B, TP = ptc.size()
        _, TF = ftc.size()
        H, W = self.window_size

        if self.training:
            self.cached_rpt = None
            rpt = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        elif self.cached_rpt is None:
            rpt = self.cached_rpt = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        else:
            rpt = self.cached_rpt

        res = []
        for p, f in zip(ptc, ftc):  # permute batch
            rpi = self.relative_position_index[f]
            rpi = rpi[:, :, p]
            rpi = rpi.flatten()   # TFHWTPHW

            rpb = rpt[rpi].view(TF*H*W, TP*H*W, self.num_heads)
            rpb = self.rpe_coef * torch.sigmoid(rpb)
            res.append(rpb)
        return torch.stack(res, dim=0).permute(0, 3, 1, 2).contiguous()


class TATWindowAttention(nn.Module):
    def __init__(self, in_channel: int, hidden_channel: int, window_size: Tuple[int, int], num_heads: int):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.window_size = window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.register_buffer('logit_max', torch.log(torch.tensor(1. / 0.01)), persistent=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self,
            query: Float[torch.Tensor, 'batch n length_q channels'],
            key: Float[torch.Tensor, 'batch n length_v channels'],
            value: Float[torch.Tensor, 'batch n length_v channels'],
            position: Float[torch.Tensor, 'batch head length_q length_v']
    ) -> Tuple[
        Float[torch.Tensor, 'batch n length_q channels_v'],
        Float[torch.Tensor, 'batch n length_q length_v'],
    ]:
        B, n, Lq, C = query.size()
        _, _, Lv, _ = key.size()

        query = query.reshape(B, n, Lq, self.num_heads, C // self.num_heads).transpose(2, 3)
        key = key.reshape(B, n, Lv, self.num_heads, C // self.num_heads).transpose(2, 3)
        value = value.reshape(B, n, Lv, self.num_heads, C // self.num_heads).transpose(2, 3)

        # qkv attention
        attn = query @ key.transpose(-2, -1)    # B n heads Lq Lv
        logit_scale = torch.clamp(self.logit_scale, max=self.logit_max).exp()
        attn = attn * logit_scale

        # relative positional encoding
        attn = attn + position[:, None]

        # mask and softmax
        attn = self.softmax(attn)

        # res
        x = attn @ value    # B n heads Lq C/h
        x = x.transpose(2, 3).flatten(3, 4)

        # proj
        # x = self.proj(x)

        return x, attn


class TATLayer(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            hidden_channel: int,
            window_size: Tuple[int, int],
            num_heads: int,
            shift: bool,
            dropout: float = 0.05,
    ):
        super(TATLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.dropout = dropout

        # self.learnable_query = nn.Parameter(torch.rand(1, 1, self.window_size[0]*self.window_size[1], self.hidden_channel), requires_grad=True)

        self.fc_in = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, self.hidden_channel, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channel, self.hidden_channel, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])
        self.attn = TATWindowAttention(hidden_channel, hidden_channel, window_size, num_heads)
        self.fc_out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_channel, self.hidden_channel, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channel, c, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])
        self.norm1s = nn.ModuleList([
            nn.LayerNorm(c)
            for c in self.in_channels
        ])
        self.norm2s = nn.ModuleList([
            nn.LayerNorm(c)
            for c in self.in_channels
        ])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, c, bias=True),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(c, c, bias=True),
                nn.Dropout(self.dropout),
            )
            for c in self.in_channels
        ])

    def forward(
            self,
            features_p: PYRAMID,
            features_f: PYRAMID,
            position: Float[torch.Tensor, 'batch head length_q length_v']
    ) -> PYRAMID:
        B, TP0, _, _, _ = features_p[0].size()
        B, TF, _, _, _ = features_f[0].size()

        x = features_f

        features_p_in = tuple(fc_in(f) for f, fc_in in zip(features_p, self.fc_in))
        features_f_in = tuple(fc_in(f) for f, fc_in in zip(features_f, self.fc_in))

        query, q_p = pyramid2windows(features_f_in, self.window_size, self.shift)
        # _, N, L, C = query.size()
        # query = self.learnable_query.expand((B, N, L, C))
        key, k_p = pyramid2windows(tuple(f[:, :-1] for f in features_p_in), self.window_size, self.shift)
        value, v_p = pyramid2windows(tuple(f[:, :-1] - f[:, -1:] for f in features_p_in), self.window_size, self.shift)
        # value, v_p = pyramid2windows(tuple(f[:, :-1] for f in features_p_in), self.window_size, self.shift)

        attn = self.attn(query, key, value, position)[0]

        features_attn = windows2pyramid(attn, q_p, self.window_size, self.shift)
        features_attn = tuple(fc_out(f) for f, fc_out in zip(features_attn, self.fc_out))

        x = tuple(f + norm(fa) for f, fa, norm in zip(x, features_attn, self.norm1s))
        x = tuple(f + norm(mlp(f)) for f, mlp, norm in zip(x, self.mlps, self.norm2s))
        return x


class TATNeck(BaseNeck):
    input_frames: int = 4
    output_frames: int = 1

    def __init__(
            self,
            in_channels: Tuple[int, ...],
            num_heads: int = 4,
            window_size: Tuple[int, int] = (8, 8),
            depth: int = 1,
            hidden_channel: Optional[int] = None,
            dropout: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.depth = depth
        self.hidden_channel = hidden_channel or in_channels[0]
        self.dropout = dropout

        self.rpe = RelativePositionalEncoding3D(window_size=self.window_size, num_heads=self.num_heads)
        self.layers = nn.ModuleList([
            TATLayer(self.in_channels, self.hidden_channel, window_size, num_heads, d % 2 != 0, self.dropout)
            for d in range(depth)
        ])

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        past_clip_ids = batch['past_clip_ids'].float()
        future_clip_ids = batch['future_clip_ids'].float()
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        position = self.rpe(past_clip_ids[:, :-1], future_clip_ids)

        features_p = tuple(f.permute(0, 1, 3, 4, 2) for f in features)                  # B T H W C
        features_f = tuple(f[:, -1:].expand(-1, TF, -1, -1, -1) for f in features_p)    # B T H W C

        for layer in self.layers:
            features_f = layer(features_p, features_f, position)

        batch['intermediate']['features_f'] = tuple(f.permute(0, 1, 4, 2, 3) for f in features_f)
        return batch
