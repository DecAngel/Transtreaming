#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.
from typing import Tuple

import torch

from src.models.layers.giraffe_fpn_btn import GiraffeNeckV2
from src.models.layers.darknet import CSPDarknet
from src.primitives.batch import IMAGE, PYRAMID
from src.primitives.model import BaseBackbone


class DRFPNBackbone(BaseBackbone):
    """
    use GiraffeNeckV2 as neck
    """
    state_dict_replace = [
        ('.lateral_conv0.', '.down_conv_5_4.'),
        ('.C3_p4.', '.down_csp_4_4.'),
        ('.reduce_conv1.', '.down_conv_4_3.'),
        ('.C3_p3.', '.down_csp_3_3.'),
        ('.bu_conv2.', '.up_conv_3_3.'),
        ('.C3_n3.', '.up_csp_3_4.'),
        ('.bu_conv1.', '.up_conv_4_4.'),
        ('.C3_n4.', '.up_csp_4_5.'),
        ('neck.', 'backbone.neck.'),
    ]

    def __init__(
        self,
        base_depth: int = 3,
        base_channel: int = 64,
        base_neck_depth: int = 3,
        hidden_ratio: float = 1.0,
        depthwise: bool = False,
        act='silu',
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

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        self.apply(init_yolo)

    def forward(self, image: IMAGE) -> PYRAMID:
        """ Extract the FPN feature (p3, p4, p5) of an image tensor of (b, t, 3, h, w)

        """
        B, T, C, H, W = image.size()
        image = image.flatten(0, 1)

        feature = self.backbone(image)
        feature = list(feature[f_name] for f_name in self.feature_names)
        feature = self.neck(feature)

        return tuple([f.unflatten(0, (B, T)) for f in feature])
