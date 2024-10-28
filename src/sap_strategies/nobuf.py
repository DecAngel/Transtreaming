import torch

from src.primitives.model import BaseModel
from src.primitives.sap import BaseSAPStrategy, SAPClient

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

            inp['past_clip_ids'] = self.past_clip_ids
            inp['future_clip_ids'] = self.future_clip_ids

            input_buffer.append(inp['image']['image'])
            input_buffer = input_buffer[:self.past_length]
            inp['image']['image'] = torch.cat(input_buffer, dim=1)

            res = self.proc_fn(model, inp)
            self.send_fn(res, client)
