import torch

from src.primitives.model import BaseModel
from src.primitives.sap_strategy import BaseSAPStrategy, SAPClient

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class NoBufStrategy(BaseSAPStrategy):
    def __init__(self, past_length: int, future_length: int):
        super().__init__(past_length, future_length)

    def infer_step(
            self,
            model: BaseModel,
            client: SAPClient,
    ) -> None:
        input_buffer = []

        while True:
            inp = self.recv_fn(client)
            if inp is None:
                # end of sequence
                break

            frame_id, delta, image = inp
            input_buffer.append(image)
            input_buffer = input_buffer[:self.past_length]
            image = torch.cat(input_buffer, dim=1)

            res, _ = self.proc_fn(image, self.past_clip_ids, self.future_clip_ids, None, model)
            self.send_fn(res, client)
