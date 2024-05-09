from typing import Optional, Union

import torch

from src.primitives.batch import PYRAMID, COORDINATE, LABEL, SIZE, BBoxDict, LossDict
from src.primitives.model import BaseHead
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class DummyHead(BaseHead):
    require_prev_frame = False

    def __init__(self):
        super().__init__()

    def forward(
            self,
            features: PYRAMID,
            gt_coordinate: Optional[COORDINATE] = None,
            gt_label: Optional[LABEL] = None,
            shape: Optional[SIZE] = None,
    ) -> Union[BBoxDict, LossDict]:
        logger.debug(f'Input shape: {[tuple(f.shape) for f in features]}')
        if self.training:
            logger.debug(f'Input gt_coordinate shape: {tuple(gt_coordinate.shape)}')
            logger.debug(f'Input gt_label shape: {tuple(gt_label.shape)}')
            logger.debug(f'Input shape value: {tuple(shape.cpu().tolist())}')
            return {'loss': torch.rand(1, requires_grad=True).mean()}
        else:
            return {
                'coordinate': torch.randint(0, 100, size=(50, 4), dtype=torch.float32, device=features[0].device),
                'label': torch.randint(0, 10, size=(50, ), dtype=torch.long, device=features[0].device),
                'probability': torch.ones(50, dtype=torch.float32, device=features[0].device),
            }
