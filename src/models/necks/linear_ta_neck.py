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


class PositionalEncoding3D(nn.Module):
    def __init__(
            self,
            channels: int,
    ):
        super().__init__()
        MAX_T, MAX_H, MAX_W = 10, 50, 50
        channels = int(np.floor(channels / 6) * 2)
        if channels % 2:
            channels -= 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

        pos_x = torch.arange(MAX_H)
        pos_y = torch.arange(MAX_W)
        pos_z = torch.arange(MAX_T)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
        emb_x = self.get_emb(sin_inp_x)[None, :, None, :]
        emb_y = self.get_emb(sin_inp_y)[None, None, :, :]
        emb_z = self.get_emb(sin_inp_z)[:, None, None, :]
        emb = torch.zeros(MAX_T, MAX_H, MAX_W, channels * 3)
        emb[:, :, :, :channels] = emb_x
        emb[:, :, :, channels:2 * channels] = emb_y
        emb[:, :, :, 2 * channels:] = emb_z
        # B T H W C
        self.register_buffer('emb', emb, persistent=False)

    @staticmethod
    def get_emb(sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(
            self,
            time_constant: TIME,
            height: int,
            width: int,
    ) -> Float[torch.Tensor, 'batch time height width channel']:
        tcs = torch.unbind(time_constant.type(torch.long), dim=0)
        return torch.stack([
            self.emb[torch.abs(tc), :height, :width, :]*torch.sign(tc)[:, None, None, None]
            for tc in tcs
        ], dim=0)


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim_q: int,
            dim_v: int,
            num_heads: int,
    ):
        super().__init__()
        assert dim_q % num_heads == 0 and dim_v % num_heads == 0
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5

        self.fc_query = nn.Linear(dim_q, dim_q)
        self.fc_key = nn.Linear(dim_q, dim_q)
        self.fc_value = nn.Linear(dim_v, dim_v)
        self.fc_out = nn.Linear(dim_v, dim_v)

    def forward(
            self,
            query: Float[torch.Tensor, 'batch nH nW length_q channels_q'],
            key: Float[torch.Tensor, 'batch nH nW length_v channels_q'],
            value: Float[torch.Tensor, 'batch nH nW length_v channels_v'],
    ) -> Tuple[
        Float[torch.Tensor, 'batch nH nW length_q channels_v'],
        Float[torch.Tensor, 'batch nH nW length_q length_v'],
    ]:
        B, nH, nW, LQ, CQ = query.size()
        B, nH, nW, LV, CQ = key.size()
        B, nH, nW, LV, CV = value.size()

        query, key, value = self.fc_query(query), self.fc_key(key), self.fc_value(value)
        # query = query.unflatten(4, (self.num_heads, -1)).permute(0, 2, 1, 3)
        # key = key.unflatten(2, (self.num_heads, -1)).permute(0, 2, 1, 3)
        # value = value.unflatten(2, (self.num_heads, -1)).permute(0, 2, 1, 3)

        query = query * self.scale
        attn = query @ key.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)

        # res = (attn @ value).permute(0, 2, 1, 3).flatten(2, 3)
        res = attn @ value
        res = self.fc_out(res)
        return res, attn


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            in_channel: int,
            num_heads: int,
            window_size: Tuple[int, int],
    ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1_1 = nn.LayerNorm(in_channel)
        self.attn1 = CrossAttention(in_channel, in_channel, self.num_heads)
        self.norm1_2 = nn.LayerNorm(in_channel)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
        )

        self.norm2_1 = nn.LayerNorm(in_channel)
        self.attn2 = CrossAttention(in_channel, in_channel, self.num_heads)
        self.norm2_2 = nn.LayerNorm(in_channel)
        self.mlp2 = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
        )

    def f2w(
            self,
            features: Float[torch.Tensor, 'batch_size time channels height width'],
            shift: bool = False,
    ):
        return features2windows(features, self.window_size, shift)

    def combined_f2w(
            self,
            features: Float[torch.Tensor, 'batch_size time channels height width']
    ):
        return torch.cat([self.f2w(features, s) for s in [False, True]], dim=0)

    def w2f(
            self,
            windows: Float[torch.Tensor, 'batch nH nW length channel'],
            image_size: Tuple[int, int, int],
            shift: bool = False,
    ):
        return windows2features(windows, image_size, self.window_size, shift)

    def combined_w2f(
            self,
            windows: Float[torch.Tensor, 'batch nH nW length channel'],
            image_size: Tuple[int, int, int],
    ):
        return sum([
            self.w2f(w, image_size, s)
            for w, s in zip(torch.chunk(windows, 2, dim=0), [False, True])
        ]) / 2

    def forward(
            self,
            features: Float[torch.Tensor, 'batch_size time_p0 channels height width'],
            past_pe: Float[torch.Tensor, 'batch_size time_p height_w width_w channels'],
            future_pe: Float[torch.Tensor, 'batch_size time_f height_w width_w channels'],
    ) -> Float[torch.Tensor, 'batch_size time_f channels height width']:
        B, _, C, H, W = features.size()
        _, TP0, _, _, _ = past_pe.size()
        _, TF, _, _, _ = future_pe.size()
        TP = TP0 - 1
        dH, dW = self.window_size

        features_all = self.combined_f2w(features)
        features_p = features_all
        features_f = features_all[:, :, :, -dH*dW:].repeat(1, 1, 1, TF, 1)

        position_p = torch.cat([past_pe.flatten(1, 3)[:, None, None, ...]] * 2, dim=0)
        position_f = torch.cat([future_pe.flatten(1, 3)[:, None, None, ...]] * 2, dim=0)
        """
        features_p = self.combined_f2w(features[:, :-1])
        features_f = self.combined_f2w(features[:, -1:].expand(-1, TF, -1, -1, -1))
        features_delta = self.combined_f2w(features[:, -1:] - features[:, :-1])
        # features_all = torch.cat([features_p, features_f], dim=3)
        position_p = torch.cat([past_pe.flatten(1, 3)[:, None, None, ...]]*2, dim=0)
        position_f = torch.cat([future_pe.flatten(1, 3)[:, None, None, ...]]*2, dim=0)
        # position_all = torch.cat([position_p, position_f], dim=3)
        """

        # 1
        # query = features_all + torch.cat([past_pe, future_pe], dim=1).flatten(1, 3)[:, None, None, ...]
        query = features_f
        key = features_p
        value = features_p

        query, key, value = self.norm1_1(query), self.norm1_1(key), self.norm1_1(value)
        res, attn = self.attn1(query, key, value)

        res = res + features_f
        res = res + self.mlp1(self.norm1_2(res))

        # 2
        query = res
        key = features_p
        value = features_p

        query, key, value = self.norm2_1(query), self.norm2_1(key), self.norm2_1(value)
        res, attn = self.attn2(query, key, value)

        res = res + features_f
        res = res + self.mlp2(self.norm2_2(res))

        # res = res[:, :, :, TP*dH*dW:]

        res = self.combined_w2f(res, (TF, H, W))

        return res


class LinearTANeck(BaseNeck):
    def __init__(
            self,
            in_channels: Tuple[int, ...],
            **kwargs,
    ):
        super().__init__()
        self.num_heads = 1
        self.window_size = (6, 6)
        self.pe = PositionalEncoding3D(min(in_channels) // 2)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(c, num_heads=self.num_heads, window_size=self.window_size)
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
                -TP, 1, step=1, dtype=torch.float32, device=features[0].device
            )[None, ...]
        else:
            past_time_constant = torch.cat([past_time_constant, torch.zeros_like(past_time_constant[:, :1])], dim=1)
        if future_time_constant is None:
            future_time_constant = torch.tensor([[1]], dtype=torch.float32, device=features[0].device)

        outputs = []
        for f, block in zip(features, self.blocks):
            B, TP0, C, H, W = f.size()
            ppe = self.pe(past_time_constant, *self.window_size)
            fpe = self.pe(future_time_constant, *self.window_size)
            if ppe.size(-1) < C:
                ppe = F.pad(ppe, (0, C - ppe.size(-1)))
            if fpe.size(-1) < C:
                fpe = F.pad(fpe, (0, C - fpe.size(-1)))
            outputs.append(block(f, ppe, fpe))

        return tuple(outputs)
