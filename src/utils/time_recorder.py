import sys
import os
import time
from typing import Optional, TextIO
from collections import defaultdict

import numpy as np


class TimeRecorder:
    def __init__(self, description: str = 'TimeRecorder', mode: str = 'sum', file: Optional[TextIO] = sys.stdout):
        self.description = description
        assert mode in ['sum', 'avg']
        self.reduce_fn = np.sum if mode == 'sum' else np.mean
        self.file = file if file is not None else open(os.devnull, 'w')
        self.restart()

    def __enter__(self):
        self.restart()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record(f'{self.description} exit')
        self.print()

    def __str__(self):
        return f'{self.description} total time: {round(self.last_time - self.start_time, 4)}s\n\t' + \
               '\t'.join([f'{k}: {np.round(self.reduce_fn(v), 4).item()}s, {len(v)} times\n' for k, v in self.t.items()])

    def restart(self):
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.t = defaultdict(list)

    def record(self, tag: Optional[str] = None) -> float:
        """Record and return the time elapsed from last call with `tag`.
        If `tag` is None, the time is not recorded, just returned.
        Recorded time with the same `tag` are summed or averaged according to `mode`.

        :param tag: the string tag
        :return: the duration in seconds
        """
        duration = time.perf_counter() - self.last_time

        if tag is not None:
            self.t[tag].append(duration)

        self.last_time = time.perf_counter()
        return duration

    def get_res_dict(self):
        return self.t

    def print(self):
        print(self.__str__(), file=self.file)
