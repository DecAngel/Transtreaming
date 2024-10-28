from typing import Callable, Tuple, Dict, Optional

import torch

from src.primitives.batch import IMAGE, BBoxDict, TIME, PYRAMID
from src.primitives.sap import BaseSAPStrategy

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DadeStrategy(BaseSAPStrategy):
    def infer_step(
            self,
            recv_fn: Callable[[], Optional[Tuple[int, int, IMAGE]]],
            proc_fn: Callable[[IMAGE, TIME, TIME, Optional[PYRAMID]], Tuple[BBoxDict, PYRAMID]],
            send_fn: Callable[[BBoxDict], None],
            time_fn: Callable[[], float]
    ) -> None:
        """TODO: Finish"""
        pass
