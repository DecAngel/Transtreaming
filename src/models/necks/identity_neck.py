from src.primitives.batch import PYRAMID, TIME, BatchDict
from src.primitives.model import BaseNeck


class IdentityNeck(BaseNeck):
    input_frames: int = 2
    output_frames: int = 2

    def __init__(self, use_replicate: bool = False):
        super().__init__()
        self.use_replicate = use_replicate

    def forward(self, batch: BatchDict) -> BatchDict:
        features = batch['intermediate']['features_p']
        B, TP = batch['past_clip_ids'].size()
        _, TF = batch['future_clip_ids'].size()

        if self.use_replicate:
            batch['intermediate']['features_f'] = tuple(f[:, [-1]*TF] for f in features)
        else:
            batch['intermediate']['features_f'] = features
        return batch
