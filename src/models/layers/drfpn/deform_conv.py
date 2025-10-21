# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair
from torchvision.ops import DeformConv2d


class DeformConv2dPack(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    The spatial arrangement is like:

    .. code:: text

        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        offset = self.conv_offset(x)
        return super().forward(x, offset)
