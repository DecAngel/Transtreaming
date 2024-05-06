import math
from typing import Tuple, Union, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
import torch.nn.functional as F

from ..types import PYRAMID, BaseNeck, TIME


def features2windows(
        features: Float[torch.Tensor, 'batch time channel height width'],
        window_size: Tuple[int, int],
        shift: bool = False,
) -> Float[torch.Tensor, 'batch nH nW length channel']:
    B, T, C, H, W = features.size()
    dH, dW = window_size
    nH, nW = math.ceil(H / dH), math.ceil(W / dW)
    pH, pW = nH * dH - H, nW * dW - W

    # pad
    features = F.pad(features, (0, pW, 0, pH))

    # shift
    if shift:
        features = torch.roll(features, shifts=(dH // 2, dW // 2), dims=(3, 4))

    # partition
    windows = features.reshape(B, T, C, nH, dH, nW, dW)
    windows = windows.permute(0, 3, 5, 1, 4, 6, 2).flatten(3, 5)
    return windows.contiguous()


def windows2features(
        windows: Float[torch.Tensor, 'batch nH nW length channel'],
        image_size: Tuple[int, int, int],
        window_size: Tuple[int, int],
        shift: bool = False,
) -> Float[torch.Tensor, 'batch time channel height width']:
    B, nH, nW, L, C = windows.size()
    T, H, W = image_size
    dH, dW = window_size

    # partition
    windows = windows.unflatten(3, (T, dH, dW))
    features = windows.permute(0, 3, 6, 1, 4, 2, 5).reshape(-1, T, C, nH*dH, nW*dW)

    # shift
    if shift:
        features = torch.roll(features, shifts=(- (dH // 2), - (dW // 2)), dims=(3, 4))

    # pad
    features = features[..., :H, :W]

    return features.contiguous()


class RelativePositionalEncoding3D(nn.Module):
    def __init__(self, window_size: Tuple[int, int], num_heads: int) -> None:
        super(RelativePositionalEncoding3D, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.rpe_coef = 16
        T = 20
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

    def forward(self, ptc: TIME, ftc: TIME) -> Float[torch.Tensor, 'batch head TFdHdW TPdHdW']:
        ptc = (-ptc).long()
        ftc = ftc.long()
        B, TP = ptc.size()
        _, TF = ftc.size()
        H, W = self.window_size

        rpt = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        res = []
        for p, f in zip(ptc, ftc):  # permute batch
            rpi = self.relative_position_index[f]
            rpi = rpi[:, :, p]
            rpi = rpi.flatten()   # TFHWTPHW

            rpb = rpt[rpi].view(TF*H*W, TP*H*W, self.num_heads)
            rpb = self.rpe_coef * torch.sigmoid(rpb)
            res.append(rpb)
        return torch.stack(res, dim=0).permute(0, 3, 1, 2).contiguous()


class FPCAWindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.register_buffer('logit_max', torch.log(torch.tensor(1. / 0.01)), persistent=False)
        self.fc_query = nn.Linear(dim, dim, bias=True)
        self.fc_key = nn.Linear(dim, dim, bias=False)
        self.fc_value = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self,
            query: Float[torch.Tensor, 'batch nH nW length_q channels'],
            key: Float[torch.Tensor, 'batch nH nW length_v channels'],
            value: Float[torch.Tensor, 'batch nH nW length_v channels'],
            position: Float[torch.Tensor, 'batch head length_q length_v']
    ) -> Tuple[
        Float[torch.Tensor, 'batch nH nW length_q channels_v'],
        Float[torch.Tensor, 'batch nH nW length_q length_v'],
    ]:
        B, nH, nW, Lq, C = query.size()
        _, _, _, Lv, _ = key.size()

        query = self.fc_query(query).reshape(B, nH, nW, Lq, self.num_heads, C // self.num_heads).transpose(3, 4)
        key = self.fc_key(key).reshape(B, nH, nW, Lv, self.num_heads, C // self.num_heads).transpose(3, 4)
        value = self.fc_value(value).reshape(B, nH, nW, Lv, self.num_heads, C // self.num_heads).transpose(3, 4)

        # cosine attention
        # attn = (F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1))
        attn = query @ key.transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=self.logit_max).exp()
        attn = attn * logit_scale

        # relative positional encoding
        # print(torch.std_mean(attn), torch.std_mean(position))
        attn = attn + position[:, None, None]

        # mask and softmax
        attn = self.softmax(attn)

        # res
        x = attn @ value
        x = x.transpose(3, 4).flatten(4, 5)

        # proj
        x = self.proj(x)

        return x, attn


class FPCALayer(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int):
        super(FPCALayer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = FPCAWindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim),
        )

    def forward(
            self,
            query: Float[torch.Tensor, 'batch nH nW length_q channels'],
            key: Float[torch.Tensor, 'batch nH nW length_v channels'],
            value: Float[torch.Tensor, 'batch nH nW length_v channels'],
            position: Float[torch.Tensor, 'batch head length_q length_v']
    ):
        x = query
        x = x + self.norm1(self.attn(query, key, value, position)[0])
        x = x + self.norm2(self.mlp(x))
        return x


class FPCABlock(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, depth: int):
        super(FPCABlock, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.depth = depth

        self.layers = nn.ModuleList([
            FPCALayer(dim, window_size, num_heads)
            for _ in range(depth)
        ])
        self.shifts = [d % 2 != 0 for d in range(depth)]

    def forward(
            self,
            features_f: Float[torch.Tensor, 'batch_size time_f channels height width'],
            features_p: Float[torch.Tensor, 'batch_size time_p channels height width'],
            position: Float[torch.Tensor, 'batch head length_q length_v']
    ):
        B, TF, C, H, W = features_f.size()

        res = features_f
        for layer, shift in zip(self.layers, self.shifts):
            query = features2windows(res, self.window_size, shift)
            key = features2windows(features_p, self.window_size, shift)
            value = features2windows(features_f[:, :1] - features_p, self.window_size, shift)
            # value = key
            res = layer(query, key, value, position)
            res = windows2features(res, (TF, H, W), self.window_size, shift)
        return res


class FPCANeck(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs,
    ):
        super().__init__()
        self.num_heads = 4
        self.window_size = (8, 8)
        self.depth = 1
        self.rpe = RelativePositionalEncoding3D(window_size=self.window_size, num_heads=self.num_heads)
        self.blocks = nn.ModuleList([
            FPCABlock(c, self.window_size, self.num_heads, self.depth)
            for c in in_channels
        ])

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None
    ) -> PYRAMID:
        if past_time_constant is None:
            TP = features[0].size(1) - 1
            past_time_constant = torch.arange(
                -TP, 0, step=1, dtype=torch.float32, device=features[0].device
            )[None, ...]
        if future_time_constant is None:
            future_time_constant = torch.tensor([[1]], dtype=torch.float32, device=features[0].device)
        TF = future_time_constant.size(1)

        if False:
            past_time_constant = torch.cat([past_time_constant, torch.zeros_like(past_time_constant[:, :1])], dim=1)

        outputs = []
        for f, block in zip(features, self.blocks):
            position = self.rpe(past_time_constant, future_time_constant)
            outputs.append(block(f[:, -1:].expand(-1, TF, -1, -1, -1), f[:, :-1], position))

        return tuple(outputs)


