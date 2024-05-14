import time
from typing import Callable, Tuple, Dict, Optional

import torch
import torch.multiprocessing as mp

from kornia.geometry import resize

from src.primitives.batch import IMAGE, BBoxDict, TIME, PYRAMID
from src.primitives.sap_strategy import BaseSAPStrategy

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class AdaptiveQueryStrategy(BaseSAPStrategy):
    def __init__(self, past_length: int, threshold: float = 0.8):
        super().__init__()
        self.past_length = past_length
        self.threshold = threshold
        self.past_clip_ids = torch.tensor([list(range(-past_length+1, 1))], dtype=torch.long)
        self.future_clip_ids = torch.tensor([[1]], dtype=torch.long)

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.input_event = mp.Event()
        self.output_event = mp.Event()

        self.end_event = mp.Event()

    def input_worker(self, recv_fn):
        while not self.end_event.is_set():
            self.input_event.wait()
            while True:
                res = recv_fn()
                if res is None:
                    self.input_event.clear()
                    break
                else:
                    self.input_queue.put(res)

    def process_worker(self, process_fn, time_fn):
        pass

    def output_worker(self, send_fn, time_fn):
        while not self.end_event.is_set():
            self.input_event.wait()
            while True:
                pass

    def infer_sequence(
            self,
            input_fn: Callable[[], Optional[Tuple[int, IMAGE]]],
            process_fn: Callable[[IMAGE, TIME, TIME, Optional[PYRAMID]], Tuple[BBoxDict, PYRAMID]],
            output_fn: Callable[[BBoxDict], None],
            time_fn: Callable[[], float],
    ) -> None:
        current_fid = 0
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
