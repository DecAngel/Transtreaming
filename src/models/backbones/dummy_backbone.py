from typing import Tuple

from torch import nn

from src.primitives.batch import IMAGE, PYRAMID, BatchDict
from src.primitives.model import BaseBackbone
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class DummyBackbone(BaseBackbone):
    def __init__(self, out_channels: Tuple[int, ...] = (128, 256, 512)):
        super().__init__()
        self.linear = nn.ModuleList()
        self.pool = nn.ModuleList()
        for c1, c2 in zip((3, ) + out_channels[:-1], out_channels):
            self.linear.append(nn.Conv2d(c1, c2, 1, 1))
            self.pool.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))

    def forward(self, batch: BatchDict) -> BatchDict:
        image = batch['image']['image']
        logger.debug(f'Input shape: {tuple(image.shape)}')
        B, T, C, H, W = image.shape
        features = []
        x = image.flatten(0, 1)
        for l, p in zip(self.linear, self.pool):
            x = p(l(x))
            features.append(x.unflatten(0, (B, T)))
        batch['intermediate']['features_p'] = tuple(features)
        return batch
