from typing import Optional

from src.primitives.batch import PYRAMID, TIME
from src.primitives.model import BaseNeck


class IdentityNeck(BaseNeck):
    input_frames: int = 2
    output_frames: int = 2

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None
    ) -> PYRAMID:
        TF = future_time_constant.size(-1)
        return tuple(f[:, [-1]*TF] for f in features)
