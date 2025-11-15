import math
import queue
from typing import Optional, List, Callable, Tuple
from ctypes import c_int, c_double, c_float, c_uint8

import numpy as np
import torch
import torch.multiprocessing as mp

from src.primitives.batch import PYRAMID, BatchDict
from src.primitives.model import BaseModel
from src.primitives.sap import BaseSAPStrategy, SAPClient
from src.utils.array_operations import remove_pad_along

from src.utils.pylogger import RankedLogger
from src.utils.array_operations import slice_along
from src.utils.collection_operations import to_device


log = RankedLogger(__name__, rank_zero_only=False)
mp.set_sharing_strategy('file_system')
#
#
# def hash_ndarray(array: np.ndarray, ) -> float:
#     coef = np.zeros_like(array)
#     for i in range(coef.ndim):
#         slice_along(coef, i)


def deep_clone(obj):
    if isinstance(obj, dict):
        return type(obj)({k: deep_clone(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)([deep_clone(v) for v in obj])
    elif isinstance(obj, torch.Tensor):
        return torch.clone(obj)
    else:
        return obj


def submit_worker(
        input_array: mp.RawArray,
        h_w_r: tuple[int, int, int],
        past_time_initial: list[int],
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        model: BaseModel,
) -> None:
    buffer = None
    h, w, r = h_w_r
    image = torch.from_numpy(np.frombuffer(input_array, dtype=np.uint8).reshape((3, h, w)))
    device = model.device
    past_time = past_time_initial
    prev_frame = None
    while True:
        res = input_queue.get()
        while True:
            # get latest input
            try:
                res = input_queue.get_nowait()
            except queue.Empty:
                break

        if res is None:
            # end
            output_queue.put(None)
            output_queue.close()
            break
        else:
            # new input frame
            # transform
            frame_index, future_time = res
            if prev_frame is not None:
                delta = frame_index - prev_frame
                past_time = [max(p - delta, -60) for p in past_time[1:]] + [0] if delta > 0 else past_time
            prev_frame = frame_index
            video = image.to(device=device, dtype=torch.float16).unsqueeze(0).unsqueeze(0)
            print(f'recv: {frame_index, past_time, future_time, video.mean().item()}')
            batch: BatchDict = {
                'meta': {
                    'image_id': torch.tensor([frame_index], dtype=torch.long, device=device),
                    'seq_id': torch.tensor([0], dtype=torch.long, device=device),
                    'frame_id': torch.tensor([frame_index], dtype=torch.long, device=device),
                    'current_size': torch.tensor(
                        [[h // r, w // r]],
                        dtype=torch.long, device=device),
                    'original_size': torch.tensor([h, w], dtype=torch.long, device=device)
                },
                'image': {'image': video},
                'image_clip_ids': torch.tensor([[0]], dtype=torch.long, device=device),
                'past_clip_ids': torch.tensor([past_time], dtype=torch.long, device=device),
                'future_clip_ids': torch.tensor([future_time], dtype=torch.long, device=device),
            }
            if buffer is not None:
                batch['buffer'] = buffer
            batch = model.inference(batch)
            buffer = batch['buffer']
            for i, f in enumerate(batch['future_clip_ids'][0].tolist()):
                output_queue.put((frame_index, frame_index+f, {
                    'coordinate': batch['bbox_pred']['coordinate'][0, i].cpu().numpy(),
                    'probability': batch['bbox_pred']['probability'][0, i].cpu().numpy(),
                    'label': batch['bbox_pred']['label'][0, i].cpu().numpy(),
                }))


class LASStrategy(BaseSAPStrategy):
    def __init__(
            self,
            past_length: int,
            future_length: int,
    ):
        super().__init__(past_length, future_length)
        self.output_buffer = {}
        self._trans_time = {}
        self._proc_input_time = {}
        # self.manager = mp.Manager()
        # self.end_event: mp.Event = self.manager.Event()
        # self.input_queue: mp.Queue = self.manager.Queue()
        # self.output_queue: mp.Queue = self.manager.Queue()
        self.input_array: mp.RawArray = None
        self.input_ndarray: np.ndarray | None = None
        self.input_queue: mp.Queue | None = None
        self.output_queue: mp.Queue | None = None
        self.process: mp.Process | None = None

    def transform_fn(self, frame: np.ndarray):
        frame = frame[::self._resize_ratio, ::self._resize_ratio, [2, 1, 0]]
        return np.transpose(frame, [2, 0, 1])

    def recv_simple_fn(self, client: SAPClient) -> Tuple | int | None:
        current_frame_id, frame = client.get_frame()
        if current_frame_id is not None:
            if self._prev_frame_id is None:
                frame_id_delta = current_frame_id
                self._prev_frame_id = current_frame_id
            elif current_frame_id != self._prev_frame_id:
                frame_id_delta = current_frame_id - self._prev_frame_id
                self._prev_frame_id = current_frame_id
            else:
                # not yet coming
                return 0

            t1 = self.time_fn()
            # h, w = frame.shape[:2]
            inp = current_frame_id, frame_id_delta, self.transform_fn(frame)
            t2 = self.time_fn()
            self._recv_time.append((t1, t2 - t1))
            return inp
        else:
            return None

    def send_simple_fn(self, bbox: dict, client: SAPClient, delay: float = 0.0) -> None:
        t1 = self.time_fn()
        coordinate = remove_pad_along(bbox['coordinate'], axis=0) * self._resize_ratio
        probability = bbox['probability'][:coordinate.shape[0]]
        label = bbox['label'][:coordinate.shape[0]]
        client.send_result_to_server(coordinate, probability, label, delay=delay)
        t2 = self.time_fn()
        self._send_time.append((t1, t2 - t1))


    def proc_input_fn(
            self,
            timestamp: int,
            image: np.ndarray,
            future_time: list[int],
    ):
        t1 = self.time_fn()
        self.input_ndarray[:] = image.flatten()
        self.input_queue.put((timestamp, future_time))
        # self.input_queue.put((timestamp, batch))
        self._proc_input_time[timestamp] = t1

    def proc_output_fn(
            self,
    ) -> list[tuple[int, dict]] | None:
        t2 = self.time_fn()
        try:
            res = self.output_queue.get_nowait()
        except queue.Empty:
            return None
        timestamp = res[0]

        outputs = [res]
        while True:
            # get all output
            try:
                res = self.output_queue.get_nowait()
                outputs.append(res)
            except queue.Empty:
                break

        self._proc_time.append((self._proc_input_time[timestamp], t2 - self._proc_input_time[timestamp]))
        return outputs


    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=10.0)

    def infer_prepare(
            self,
            model: BaseModel,
            client: SAPClient,
            sap_factor: float,
            dataset_resize_ratio: int,
            demo_input: List[np.ndarray],
    ) -> None:
        self._device = torch.device("cpu")
        self.past_clip_ids = self.past_clip_ids.to(self._device)
        self.future_clip_ids = self.future_clip_ids.to(self._device)
        self._sap_factor = sap_factor
        self._resize_ratio = dataset_resize_ratio
        self._prev_frame_id = None
        self._recv_time.clear()
        self._proc_time.clear()
        self._send_time.clear()

        self.output_buffer.clear()
        self._proc_input_time.clear()

        frame = self.transform_fn(demo_input[0])
        h, w = frame.shape[1:]

        self.input_array = mp.RawArray(c_uint8, 3 * h * w)
        self.input_ndarray = np.frombuffer(self.input_array, dtype=np.uint8)
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.process = mp.Process(target=submit_worker, args=(
            self.input_array,
            (h, w, self._resize_ratio),
            self.past_clip_ids[0].tolist(),
            self.input_queue,
            self.output_queue,
            model,
        ))
        self.process.start()

        for i, frame in enumerate(demo_input):
            image = self.transform_fn(frame)
            self.input_ndarray[:] = image.flatten()
            self.input_queue.put((i-len(demo_input), self.future_clip_ids[0].tolist()))
            self.output_queue.get()
            while not self.output_queue.empty():
                self.output_queue.get()

    def infer_step(
            self,
            model: BaseModel,
            client: SAPClient,
    ) -> None:
        delta_t = 1.0
        prev_submit_time = 0
        past_time = self.past_clip_ids[0].tolist()
        future_time = self.future_clip_ids[0].tolist()
        while True:
            # input
            start_time = self.time_fn()
            inp = self.recv_simple_fn(client)
            # inp = self.recv_fn(client)
            if inp is None:
                # end of sequence
                self.input_queue.put(None)
                self.input_queue.close()
                while True:
                    res = self.output_queue.get()
                    if res is None:
                        break
                self.process.join()
                assert not self.process.is_alive()
                self.process = None
                break
            elif inp == 0:
                continue

            frame_id, delta, frame = inp

            future_time = list(range(max(math.floor(start_time-frame_id), 0), min(60, math.ceil(start_time-frame_id+3*delta_t)+1)))

            log.info(f'processing {frame_id}')
            self.proc_input_fn(frame_id, frame, future_time)
            end_time = self.time_fn()
            self._trans_time[frame_id] = end_time - start_time

            # check output queue and emit output
            while True:
                # check output first
                res = self.proc_output_fn()
                if res is not None:
                    # get all output
                    index = None
                    for i, future_time, bbox in res:
                        index = i
                        self.output_buffer[future_time] = bbox

                    cur_time = self.time_fn()
                    delta_t_alg = cur_time - self._proc_input_time[index]
                    delta_t_trans = self._trans_time[index]
                    delta_t = 0.5*delta_t + 0.5*(delta_t_alg+delta_t_trans)

                # submit result
                cur_time = self.time_fn()
                if cur_time > prev_submit_time:
                    # submit
                    for t in range(math.ceil(cur_time), prev_submit_time, -1):
                        if t in self.output_buffer:
                            print(f'submitting {t}')
                            bbox = self.output_buffer[t]
                            del self.output_buffer[t]
                            self.send_simple_fn(bbox, client)
                            break
                    prev_submit_time = math.ceil(cur_time)

                if res is not None:
                    break
