from src.primitives.model import BaseModel
from src.primitives.sap_strategy import BaseSAPStrategy, SAPClient

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DynamicStrategy(BaseSAPStrategy):
    def __init__(self, past_length: int, future_length: int, threshold: float = 0.8):
        super().__init__(past_length, future_length)
        self.threshold = threshold

    def infer_step(
            self,
            model: BaseModel,
            client: SAPClient,
    ) -> None:
        buffer = None

        while True:
            inp = self.recv_fn(client)
            if inp is None:
                # end of sequence
                break

            frame_id, delta, image = inp

            if buffer is not None:
                self.past_clip_ids[:, :-1] = self.past_clip_ids[:, 1:] - delta
                self.past_clip_ids.clamp_min_(-9)
                self.past_clip_ids[:, -1].zero_()

            start_fid = self.time_fn()
            runtime_remainder = start_fid - frame_id
            if runtime_remainder > self.threshold:
                continue

            res, buffer = self.proc_fn(image, self.past_clip_ids, self.future_clip_ids, buffer, model)
            self.send_fn(res, client)
