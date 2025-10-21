from typing import Optional, Tuple, List

import kornia.augmentation as ka
import kornia.filters as kf
import torch

from src.primitives.batch import BatchDict
from src.primitives.model import BaseTransform


def random_resize(*sizes: Tuple[int, int]):
    return ka.VideoSequential(
        *[ka.Resize((h, w)) for h, w in sizes],
        random_apply=1,
    )


class KorniaTransform2(BaseTransform):
    def __init__(
            self,
            train: Optional[List[ka.AugmentationBase2D]] = None,
            val: Optional[List[ka.AugmentationBase2D]] = None,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.register_buffer('c255', tensor=torch.tensor(255.0, dtype=torch.float32), persistent=False)

    def preprocess(self, batch: BatchDict) -> BatchDict:
        self.train