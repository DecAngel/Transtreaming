import contextlib
import time
from collections import defaultdict

import numpy as np

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TimeRecorder:
    def __init__(self, description: str = 'TimeRecorder'):
        self.description = description
        self.time_start = {}
        self.time_record = defaultdict(list)

    def __enter__(self):
        self.record_start(f'{self.description} total time')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record_end(f'{self.description} total time')
        self.print()

    def __str__(self):
        return f'{self.description}\n\t' + '\t'.join([
            f'{k: <20s}: '
            f'total={np.sum(v).item():6.4f}s, '
            f'mean={np.mean(v).item():6.4f}s, '
            f'std={np.std(v).item():6.4f}s, '
            f'times={len(v)}\n'
            for k, v in self.time_record.items()
        ])

    def record_start(self, tag: str):
        self.time_start[tag] = time.perf_counter()

    def record_end(self, tag: str):
        self.time_record[tag].append(time.perf_counter() - self.time_start[tag])

    def restart(self):
        self.time_start = {}
        self.time_record = defaultdict(list)

    @contextlib.contextmanager
    def record(self, tag: str):
        """Record and return the time elapsed from last call with `tag`.
        If `tag` is None, the time is not recorded, just returned.
        Recorded time with the same `tag` are summed or averaged according to `mode`.

        :param tag: the string tag
        :return: the duration in seconds
        """
        self.record_start(tag)
        try:
            yield None
        finally:
            self.record_end(tag)

    def get_res_dict(self):
        return self.time_record

    def print(self):
        log.info(self.__str__())
