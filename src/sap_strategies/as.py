import math
from typing import Optional, List

import numpy as np
import torch

from src.primitives.batch import PYRAMID
from src.primitives.model import BaseModel
from src.primitives.sap_strategy import BaseSAPStrategy, SAPClient

from src.utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=False)


def break_pyramid(pyramid: PYRAMID) -> List[PYRAMID]:
    t = pyramid[0].size(1)
    return [tuple(p[:, i:i+1] for p in pyramid) for i in range(t)]


class AdaptiveStrategy(BaseSAPStrategy):
    def __init__(
            self,
            past_length: int,
            future_length: int,
            window_size: int = 5,
    ):
        super().__init__(past_length, future_length)
        self.exponential = np.array(0.5) ** (list(range(1, window_size)) + [window_size-1])
        self.time_other = np.ones((window_size,)) * 1
        self.time_bn = np.ones((window_size,)) * 0.5
        self.time_h = np.ones((window_size,)) * 0.5

    """
    def transform_fn(self, frame: np.ndarray):
        frame = frame[::self._resize_ratio, ::self._resize_ratio, [2, 1, 0]]
        frame = torch.from_numpy(frame.transpose((2, 0, 1))).unsqueeze(0).unsqueeze(0)
        frame = frame.to(dtype=torch.float16, device=self._device)
        return frame
    """

    def update_t(self, time_array: np.ndarray, new_t: Optional[float] = None):
        if new_t is not None:
            time_array[1:] = time_array[:-1]
            time_array[0] = new_t
        return np.dot(time_array, self.exponential)

    def infer_prepare(
            self,
            model: BaseModel,
            client: SAPClient,
            sap_factor: float,
            dataset_resize_ratio: int,
            demo_input: List[np.ndarray],
    ) -> None:
        super().infer_prepare(model, client, sap_factor, dataset_resize_ratio, demo_input)
        buffer = None
        for frame in demo_input:
            frame = self.transform_fn(frame)
            shape = torch.tensor([frame.shape[-2:]], dtype=torch.long, device=model.device)
            ff, buffer = model.inference_backbone_neck(frame, self.past_clip_ids, self.future_clip_ids, buffer)
            _ = model.inference_head(break_pyramid(ff)[0], shape=shape)

    def infer_step(
            self,
            model: BaseModel,
            client: SAPClient,
    ) -> None:
        buffer: Optional[PYRAMID] = None

        t_other: float = self.update_t(self.time_other)
        t_bn: float = self.update_t(self.time_bn)
        t_h: float = self.update_t(self.time_h)
        t_last: Optional[float] = None

        while True:
            inp = self.recv_fn(client)
            if inp is None:
                # end of sequence
                break

            frame_id, delta, image = inp

            if buffer is not None:
                self.past_clip_ids[:, :-1] = self.past_clip_ids[:, 1:] - delta
                self.past_clip_ids.clamp_min_(-29)
                self.past_clip_ids[:, -1].zero_()

            # update t_other
            t_1 = self.time_fn()
            if t_last is None:
                t_other = self.update_t(self.time_other, None)
            else:
                t_other = self.update_t(self.time_other, t_1 - t_last)

            # calculate TF
            t_start_delta = t_1 - frame_id
            t_delta = ((self.future_length+1)*t_h+t_bn+t_other)/self.future_length
            t_first = t_start_delta+t_bn+t_h
            TF = list(dict.fromkeys([min(math.ceil(t_first+i*t_delta), 19) for i in range(self.future_length)]))
            TP = self.past_clip_ids.cpu().tolist()[0]
            future_clip_ids = torch.tensor([TF], dtype=torch.long)

            # log.info(f'TF={TF}, TP={TP}, t_start_delta={t_start_delta:.3f}, t_first={t_first:.3f}, t_delta={t_delta:.3f}, t_other={t_other:.3f}, t_bn={t_bn:.3f}, t_h={t_h:.3f}')

            # bn
            shape = torch.tensor([image.shape[-2:]], dtype=torch.long, device=model.device)
            features_f, buffer = self.proc_backbone_neck_fn(image, self.past_clip_ids, future_clip_ids, buffer, model)
            t_2 = self.time_fn()
            t_bn = self.update_t(self.time_bn, t_2-t_1)

            # h
            features_f_list = break_pyramid(features_f)
            # quick first
            res = self.proc_head_fn(features_f_list[0], shape, model)
            self.send_fn(res, client)
            t_3 = self.time_fn()
            t_h = self.update_t(self.time_h, t_3 - t_2)

            for tf, ff in zip(TF[1:], features_f_list[1:]):
                t_4 = self.time_fn()
                if t_4 > frame_id + tf + t_h:
                    # too late, skip
                    continue
                res = self.proc_head_fn(ff, shape, model)
                t_5 = self.time_fn()
                self.send_fn(res, client, max(0.0, (frame_id + tf - t_5) / 30 / self._sap_factor))
                t_6 = self.time_fn()
                t_h = self.update_t(self.time_h, t_6 - t_4)

            t_last = self.time_fn()
