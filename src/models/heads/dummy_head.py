from typing import Optional, Union

import torch

from src.primitives.batch import PYRAMID, COORDINATE, LABEL, SIZE, BBoxDict, LossDict, TIME, BatchDict
from src.primitives.model import BaseHead
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class DummyHead(BaseHead):
    require_prev_frame = False

    def __init__(self):
        super().__init__()

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_f']
        gt_coordinates = batch['bbox']['coordinate'] if 'bbox' in batch else None
        gt_labels = batch['bbox']['label'] if 'bbox' in batch else None
        shape = tuple(batch['meta']['current_size'][0].cpu().tolist())

        logger.debug(f'Input shape: {[tuple(f.shape) for f in features]}')
        if self.training:
            logger.debug(f'Input gt_coordinate shape: {tuple(gt_coordinates.shape)}')
            logger.debug(f'Input gt_label shape: {tuple(gt_labels.shape)}')
            logger.debug(f'Input shape value: {shape}')
            batch['loss'] = {'loss': torch.rand(1, requires_grad=True).mean()}
            return batch
        else:
            batch['bbox_pred'] = {
                'coordinate': torch.randint(0, 100, size=(50, 4), dtype=torch.float32, device=features[0].device),
                'label': torch.randint(0, 10, size=(50, ), dtype=torch.long, device=features[0].device),
                'probability': torch.ones(50, dtype=torch.float32, device=features[0].device),
            }
            return batch
