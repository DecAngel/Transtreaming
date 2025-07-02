from typing import Tuple, Dict, List

import math
import torch
import torch.nn.functional as F
from jaxtyping import Float

from src.primitives.batch import PYRAMID


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


def pyramid2windows2(
        pyramid: PYRAMID
) -> Tuple[Float[torch.Tensor, 'batch n length channel'], List[Tuple[int, ...]]]:
    # split into B H*W T C1*16+C2*4+C3*1
    # assume input stride is 4:2:1
    B, T, bH, bW, _ = pyramid[-1].size()
    # print(B, T, bH, bW)
    windows = []
    params = []
    for features, stride in zip(pyramid, [4, 2, 1]):
        _, _, H, W, C = features.size()
        pH, pW = bH * stride - H, bW * stride - W
        features = F.pad(features, (0, 0, 0, pW, 0, pH))
        window = features.reshape(B, T, bH, stride, bW, stride, C)
        # B, bH*bW, T, stride^2*C
        window = window.permute(0, 2, 4, 1, 3, 5, 6).flatten(4, 6).flatten(1, 2)
        windows.append(window)
        params.append((B, T, H, W, C, bH, bW, stride))
    return torch.cat(windows, dim=-1).contiguous(), params


def windows2pyramid2(
        windows: Float[torch.Tensor, 'batch n length channel'],
        params: List[Tuple[int, ...]],
) -> PYRAMID:
    pyramid = []
    for window, (B, T, H, W, C, bH, bW, stride) in zip(torch.split(windows, [p[4]*s*s for p, s in zip(params, [4, 2, 1])], dim=-1), params):
        window = window.unflatten(3, (stride, stride, C)).unflatten(1, (bH, bW))
        features = window.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, T, bH * stride, bW * stride, C)
        # pad
        pyramid.append(features[..., :H, :W, :].contiguous())
    return tuple(pyramid)


# def new_f2w(
#         features: Float[torch.Tensor, 'batch time channel height width'],
#         window_size: Tuple[int, int],
#         shift: bool = False,
# ) -> Tuple[Float[torch.Tensor, 'batch time window_num length channel'], Dict[str, int]]:
#     B, T, C, H, W = features.size()
#     dH, dW = window_size
#     nH, nW = math.ceil(H / dH), math.ceil(W / dW)
#     pH, pW = nH * dH - H, nW * dW - W
#
#     features = F.pad(features, (0, pW, 0, pH))
#
#     # shift
#     if shift:
#         features = torch.roll(features, shifts=(dH // 2, dW // 2), dims=(3, 4))
#
#     windows = features.reshape(B, T, C, nH, dH, nW, dW)
#     windows = windows.permute(0, 3, 5, 1, 4, 6, 2).flatten(3, 5)


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
