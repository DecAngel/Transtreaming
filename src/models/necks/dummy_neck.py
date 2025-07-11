from src.primitives.batch import PYRAMID, TIME, BatchDict
from src.primitives.model import BaseNeck
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class DummyNeck(BaseNeck):
    input_frames: int = 4
    output_frames: int = 4

    def __init__(self):
        super().__init__()

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        past_clip_ids = batch['past_clip_ids']
        future_clip_ids = batch['future_clip_ids']
        B, TP = past_clip_ids.size()
        _, TF = future_clip_ids.size()

        logger.debug(f'Input shape: {[tuple(f.shape) for f in features]}')
        logger.debug(f'Input time: {past_clip_ids[0].cpu().tolist()}, {future_clip_ids[0].cpu().tolist()}')
        TF = future_clip_ids.size(-1)

        batch['intermediate']['features_f'] = tuple(f[:, [-1]*TF] for f in features)
        return batch
