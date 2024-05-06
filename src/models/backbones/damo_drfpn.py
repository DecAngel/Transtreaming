#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.
from typing import Tuple

import torch.nn as nn

from ..layers.darknet import CSPDarknet
from ..layers.giraffe_fpn_btn import GiraffeNeckV2
from ..types import IMAGE, PYRAMID, BaseBackbone


class DAMOBackbone(BaseBackbone):
    """
    use GiraffeNeckV2 as neck
    """

    def __init__(
        self,
        base_depth: int = 3,
        base_channel: int = 64,
        base_neck_depth: int = 3,
        hidden_ratio: float = 1.0,
        depthwise: bool = False,
        act='silu',
        **kwargs
    ):
        super().__init__()
        self.out_channels: Tuple[int, ...] = tuple(i * base_channel for i in (4, 8, 16))
        self.feature_names = ('dark3', 'dark4', 'dark5')

        # build network (hardcoded)
        self.backbone = CSPDarknet(
            base_depth=base_depth,
            base_channel=base_channel,
            out_features=self.feature_names,
            depthwise=depthwise, act=act
        )
        self.neck = GiraffeNeckV2(
            base_depth=base_neck_depth,
            base_channel=base_channel,
            hidden_ratio=hidden_ratio,
            act=act,
        )

    def forward(self, image: IMAGE) -> PYRAMID:
        """ Extract the FPN feature (p3, p4, p5) of an image tensor of (b, t, 3, h, w)

        """
        B, T, C, H, W = image.size()
        image = image.flatten(0, 1)

        feature = self.backbone(image)
        feature = list(feature[f_name] for f_name in self.feature_names)
        feature = self.neck(feature)

        return tuple([f.unflatten(0, (B, T)) for f in feature])
