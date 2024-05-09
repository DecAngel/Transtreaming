from typing import Callable, Tuple, Dict, Optional

import matplotlib.pyplot as plt

from src.primitives.batch import IMAGE, BufferDict, BBoxDict


class BaseSAPStrategy:
    def __init__(self, exp_tag: str = 'sAP'):
        super().__init__()
        self.exp_tag = exp_tag
        self.process_time = []
        self.first_time = True
        self.max_time = 30

    def infer_sequence(
            self,
            input_fn: Callable[[], Optional[IMAGE]],
            process_fn: Callable[[IMAGE, BufferDict], Tuple[BBoxDict, BufferDict]],
            output_fn: Callable[[BBoxDict], None],
            time_fn: Callable[[], float],
    ) -> None:
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

            return self.infer_sequence_impl(
                input_fn=input_fn,
                process_fn=process_fn_record,
                output_fn=output_fn_record,
                time_fn=time_fn,
            )
        else:
            return self.infer_sequence_impl(
                input_fn=input_fn,
                process_fn=process_fn,
                output_fn=output_fn,
                time_fn=time_fn,
            )

    def infer_sequence_impl(
            self,
            input_fn: Callable[[], Optional[IMAGE]],
            process_fn: Callable[[IMAGE, BufferDict], Tuple[BBoxDict, BufferDict]],
            output_fn: Callable[[BBoxDict], None],
            time_fn: Callable[[], float],
    ) -> None: raise NotImplementedError()

    def plot_process_time(self):
        plt.figure(dpi=300)
        plt.bar(list(range(len(self.process_time)))[:self.max_time], self.process_time[:self.max_time], width=0.5)
        plt.plot([-2, self.max_time+2], [1, 1], 'k--')
        plt.xlabel('Frame index')
        plt.ylabel('Process time / 33ms')
        plt.title(self.exp_tag)
        plt.show()
