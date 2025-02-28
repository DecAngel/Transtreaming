import math
from typing import Optional, List

import numpy as np
import torch

from src.primitives.batch import PYRAMID
from src.primitives.model import BaseModel
from src.primitives.sap import BaseSAPStrategy, SAPClient

from src.utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=False)


def break_pyramid(pyramid: PYRAMID) -> List[PYRAMID]:
    t = pyramid[0].size(1)
    return [tuple(p[:, i:i+1] for p in pyramid) for i in range(t)]


#TODO: change
class MGPUStrategy(BaseSAPStrategy):
    def __init__(
            self,
            past_length: int,
            future_length: int,
            window_size: int = 5,
    ):
        super().__init__(past_length, future_length)
        self.past_length = past_length
        self.future_length = future_length
        self.window_size = window_size
        self.num_gpus = torch.cuda.device_count()
        self.models = []
    
    def infer_prepare(
            self,
            model: BaseModel,
            client: SAPClient,
            sap_factor: float,
            dataset_resize_ratio: int,
            demo_input: List[np.ndarray],
    ) -> None:
        super().infer_prepare(model, client, sap_factor, dataset_resize_ratio, demo_input)
        self.models = []
        