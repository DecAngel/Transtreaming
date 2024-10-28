from src.primitives.model import BaseModel
from src.primitives.sap import BaseSAPStrategy, SAPClient

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
        past_clip_ids = self.past_clip_ids
        future_clip_ids = self.future_clip_ids
        prev_id = 0

        while True:
            inp = self.recv_fn(client)
            if inp is None:
                # end of sequence
                break

            current_id = inp['meta']['frame_id'].item()
            delta = current_id - prev_id
            prev_id = current_id
            if buffer is not None:
                past_clip_ids[:, :-1] = past_clip_ids[:, 1:] - delta
                past_clip_ids.clamp_min_(-9)
                past_clip_ids[:, -1].zero_()
                inp['buffer'] = buffer

            inp['past_clip_ids'] = past_clip_ids
            inp['future_clip_ids'] = future_clip_ids

            start_fid = self.time_fn()
            runtime_remainder = start_fid - current_id
            if runtime_remainder > self.threshold:
                continue

            res = self.proc_fn(model, inp)
            buffer = res['buffer']
            self.send_fn(res, client)
