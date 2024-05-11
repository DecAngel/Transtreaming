from typing import Callable, Tuple, Dict, Optional

import torch

from src.primitives.batch import IMAGE, BBoxDict, TIME, PYRAMID
from src.primitives.sap_strategy import BaseSAPStrategy

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DadeStrategy(BaseSAPStrategy):
    def __init__(self, past_length: int, threshold: float = 0.8):
        super().__init__()
        self.past_length = past_length
        self.threshold = threshold
        self.past_clip_ids = torch.tensor([list(range(-past_length+1, 1))], dtype=torch.long)
        self.future_clip_ids = torch.tensor([[1]], dtype=torch.long)

    def _infer_sequence_impl(
            self,
            input_fn: Callable[[], Optional[Tuple[int, IMAGE]]],
            process_fn: Callable[[IMAGE, TIME, TIME, Optional[PYRAMID]], Tuple[BBoxDict, PYRAMID]],
            output_fn: Callable[[BBoxDict], None],
            time_fn: Callable[[], float],
    ) -> None:
        """TODO: Finish"""
        current_fid = 0
        last_runtime = 1
        buffer = None

        while True:
            inp = input_fn()
            if inp is None:
                # end of sequence
                break

            delta, image = inp
            current_fid += delta

            if buffer is None:
                # first iteration
                self.past_clip_ids = self.past_clip_ids.to(device=image.device)
                self.future_clip_ids = self.future_clip_ids.to(device=image.device)
            else:
                self.past_clip_ids[:, :-1] = self.past_clip_ids[:, 1:] - delta
                self.past_clip_ids[:, -1].zero_()

            start_fid = time_fn()

            runtime_remainder = start_fid - current_fid
            if runtime_remainder > self.threshold:
                continue

            res, buffer = process_fn(image, self.past_clip_ids, self.future_clip_ids, buffer)
            output_fn(res)
