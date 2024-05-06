import itertools
from typing import Sequence, List

import torch

from src.primitives.datamodule import BaseDataSource

from src.utils import RankedLogger
from src.primitives.batch import MetaDict, ImageDict, BBoxDict

logger = RankedLogger(__name__, rank_zero_only=False)


class DummyDataSource(BaseDataSource):
    def __init__(self):
        super().__init__([-3, -2, 0], [0, 1, 3])

    def __post_init__(self):
        self.current_size = torch.tensor([200, 300], dtype=torch.int32)
        self.image = torch.ones(3, 200, 300)
        self.coordinates = torch.zeros(50, 4, dtype=torch.float32)
        self.coordinates[:10, 2:] += 1
        self.labels = torch.randint(0, 10, size=(50, ), dtype=torch.long)
        self.labels[10:] = 0
        self.probabilities = torch.ones(50, dtype=torch.float32)
        self.probabilities[10:] = 0
        self.seq_lens = [10, 20, 30]
        self.seq_lens_acc = [0] + list(itertools.accumulate(self.seq_lens[:-1]))

    def get_meta(self, seq_id: int, frame_id: int) -> MetaDict:
        return {
            'seq_id': torch.tensor(seq_id, dtype=torch.int32),
            'frame_id': torch.tensor(frame_id, dtype=torch.int32),
            'image_id': torch.tensor(self.seq_lens_acc[seq_id] + frame_id, dtype=torch.int32),
            'current_size': self.current_size,
            'original_size': self.current_size * 2,
        }

    def get_image(self, seq_id: int, frame_id: int) -> ImageDict:
        return {
            'image': self.image * ((seq_id + frame_id) % 256)
        }

    def get_bbox(self, seq_id: int, frame_id: int) -> BBoxDict:
        return {
            'coordinate': self.coordinates * ((seq_id + frame_id) % 100),
            'label': self.labels,
            'probability': self.probabilities,
        }

    def get_length(self) -> List[int]:
        return self.seq_lens
