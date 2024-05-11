from typing import Callable, Tuple, Optional, List

import matplotlib.pyplot as plt
import torch

from src.primitives.batch import IMAGE, BBoxDict, PYRAMID, TIME
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def concat_pyramids(pyramids: List[PYRAMID], dim: int = 1) -> PYRAMID:
    return tuple(
        torch.cat([
            p[i]
            for p in pyramids
        ], dim=dim)
        for i in range(len(pyramids[0]))
    )


class BaseSAPStrategy:
    def __init__(self, exp_tag: str = 'sAP'):
        super().__init__()
        self.exp_tag = exp_tag
        self.process_time = []
        self.first_time = True
        self.max_time = 100

    def infer_sequence(
            self,
            input_fn: Callable[[], Optional[Tuple[int, IMAGE]]],
            process_fn: Callable[[IMAGE, TIME, TIME, Optional[PYRAMID]], Tuple[BBoxDict, PYRAMID]],
            output_fn: Callable[[BBoxDict], None],
            time_fn: Callable[[], float],
    ) -> None:
        """

        """
        if self.first_time:
            self.first_time = False

            def process_fn_record(*args, **kwargs):
                t1 = time_fn()
                res = process_fn(*args, **kwargs)
                t2 = time_fn()
                d = int(t1) - len(self.process_time)
                self.process_time.extend([0]*d)
                self.process_time.append(t2-t1)
                return res

            def output_fn_record(*args, **kwargs):
                return output_fn(*args, **kwargs)

            return self._infer_sequence_impl(
                input_fn=input_fn,
                process_fn=process_fn_record,
                output_fn=output_fn_record,
                time_fn=time_fn,
            )
        else:
            return self._infer_sequence_impl(
                input_fn=input_fn,
                process_fn=process_fn,
                output_fn=output_fn,
                time_fn=time_fn,
            )

    def _infer_sequence_impl(
            self,
            input_fn: Callable[[], Optional[Tuple[int, IMAGE]]],
            process_fn: Callable[[IMAGE, TIME, TIME, Optional[PYRAMID]], Tuple[BBoxDict, PYRAMID]],
            output_fn: Callable[[BBoxDict], None],
            time_fn: Callable[[], float],
    ) -> None: raise NotImplementedError()

    def plot_process_time(self):
        plt.figure(dpi=600)
        plt.bar(list(range(len(self.process_time)))[:self.max_time], self.process_time[:self.max_time], width=0.5)
        plt.plot([-2, self.max_time+2], [1, 1], 'k--')
        plt.xlabel('Frame index')
        plt.ylabel('Process time / frame interval')
        plt.title(self.exp_tag)
        plt.show()
        log.info(f'frame nums {len(self.process_time)}')
