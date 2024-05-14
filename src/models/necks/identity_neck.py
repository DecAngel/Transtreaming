from typing import Optional

from src.primitives.batch import PYRAMID, TIME
from src.primitives.model import BaseNeck


class IdentityNeck(BaseNeck):
    input_frames: int = 2
    output_frames: int = 2

    def __init__(self):
        super().__init__()

    def forward(
            self,
            features: PYRAMID,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
    ) -> PYRAMID:
        B, TP, _, _, _ = features[0].size()
        _, TF = future_clip_ids.size()
        return tuple(f[:, [-1]*TF] for f in features)
