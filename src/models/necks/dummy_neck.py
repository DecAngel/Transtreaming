from typing import Optional

from src.primitives.batch import PYRAMID, TIME
from src.primitives.model import BaseNeck
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class DummyNeck(BaseNeck):
    input_frames: int = 4
    output_frames: int = 4

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
            self,
            features: PYRAMID,
            past_time_constant: Optional[TIME] = None,
            future_time_constant: Optional[TIME] = None
    ) -> PYRAMID:
        logger.debug(f'Input shape: {[tuple(f.shape) for f in features]}')
        logger.debug(f'Input time: {past_time_constant[0].cpu().tolist()}, {future_time_constant[0].cpu().tolist()}')
        TF = future_time_constant.size(-1)
        return tuple(f[:, [-1]*TF] for f in features)
