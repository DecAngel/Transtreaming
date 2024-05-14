from typing import Callable, Tuple, Optional, List

import matplotlib.pyplot as plt
import torch

from src.primitives.batch import IMAGE, BBoxDict, PYRAMID, TIME
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

import contextlib
import functools
import io
import json
import signal
import sys
import os
import platform
import socket
import subprocess
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from pycocotools.coco import COCO
from sap_toolkit.client import EvalClient
from sap_toolkit.generated import eval_server_pb2
from tqdm import tqdm
from kornia.geometry.transform import resize

from src.primitives.model import BaseModel
from src.primitives.batch import IMAGE, BBoxDict
from src.utils.array_operations import remove_pad_along
from src.utils.time_recorder import TimeRecorder


coco_eval_metric_names = [
    ('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ', 'AP5095'),
    ('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ', 'AP50'),
    ('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ', 'AP75'),
    ('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ', 'APsmall'),
    ('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ', 'APmedium'),
    ('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ', 'APlarge'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ', 'AR5095_1'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ', 'AR5095_10'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ', 'AR5095_100'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ', 'ARsmall'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ', 'ARmedium'),
    ('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ', 'ARlarge'),
]


class SAPServer:
    def __init__(self, data_dir: str, ann_file: str, output_dir: str, sap_factor: float = 1.0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.ann_file = Path(ann_file)
        self.output_dir = Path(output_dir)
        self.sap_factor = sap_factor
        self.config_path: Optional[Path] = None
        self.proc: Optional[subprocess.Popen] = None

    @staticmethod
    def find_2_unused_ports() -> Tuple[int, int]:
        # create a temporary socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                # bind the sockets to random addresses
                s1.bind(('', 0))
                s2.bind(('', 0))
                # retrieve the port numbers that was allocated
                port1 = s1.getsockname()[1]
                port2 = s2.getsockname()[1]
        return port1, port2

    def __enter__(self):
        # create temporary configs
        p1, p2 = self.find_2_unused_ports()
        config = {
            "image_service_port": p1,
            "result_service_port": p2,
            "loopback_ip": "127.0.0.1"
        }
        self.config_path = self.output_dir.joinpath(f'{p1}_{p2}.json')
        self.config_path.write_text(json.dumps(config))

        # start server
        self.proc = subprocess.Popen(
            ' '.join([
                sys.executable, '-m', 'sap_toolkit.server',
                '--data-root', f'{str(self.data_dir.resolve())}',
                '--annot-path', f'{str(self.ann_file.resolve())}',
                '--overwrite',
                '--eval-config', f'{str(self.config_path.resolve())}',
                '--out-dir', f'{str(self.output_dir.resolve())}',
                '--perf-factor', str(self.sap_factor),
            ]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=str(self.output_dir.resolve()),
            shell=True,
            text=True,
        )

        keyword = 'Welcome'
        while True:
            result = self.proc.stdout.readline()
            if keyword in result:
                break

        return self

    def get_result(self, results_file: str = 'results.json') -> Dict[str, Any]:
        self.proc.stdin.write(f'evaluate {results_file}\n')
        self.proc.stdin.flush()

        output_dict = {}
        pos = 0
        while True:
            output = self.proc.stdout.readline()
            if coco_eval_metric_names[pos][0] in output:
                output_dict[coco_eval_metric_names[pos][1]] = float(output[-5:])
                pos += 1
                if pos == len(coco_eval_metric_names):
                    break
        return output_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.communicate('close\n')
        os.remove(self.config_path)
        self.proc = None
        self.config_path = None


class SAPClient(EvalClient):
    def __init__(self, server: SAPServer, resize_ratio: int, device_id: int):
        super().__init__(
            json.loads(server.config_path.read_text()),
            (
                mp.Value('i', -1, lock=True),
                mp.Event(),
                mp.Manager().dict(),
            ),
            verbose=True
        )
        self.stream_start_time = mp.Value('d', 0.0, lock=True)
        self.resize_ratio = resize_ratio
        self.device_id = device_id
        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    def generate(self, results_file='results.json'):
        self.result_stub.GenResults(eval_server_pb2.String(value=results_file))

    def close(self):
        self.result_channel.close()
        self.existing_shm.close()
        self.results_shm.close()
        log.warning('SAP Shutdown')


def concat_pyramids(pyramids: List[PYRAMID], dim: int = 1) -> PYRAMID:
    return tuple(
        torch.cat([
            p[i]
            for p in pyramids
        ], dim=dim)
        for i in range(len(pyramids[0]))
    )


class BaseSAPStrategy:
    def __init__(self):
        super().__init__()
        self.exp_tag = self.__class__.__name__
        
        self.tr = TimeRecorder(description=f'{self.exp_tag}', mode='avg')
        self._frame_id_prev = None
        self._seq_start_time = None
        self._max_frames = None
        self._min_time = 0.1
        self._recv_times = []
        self._process_times = []
        self._send_times = []

        self.model: Optional[BaseModel] = None
        self.client: Optional[SAPClient] = None
        self.server: Optional[SAPServer] = None

    def _reset(self):
        self.tr.restart()
        self._frame_id_prev = None
        self._seq_start_time = time.perf_counter()
        self._recv_times.clear()
        self._process_times.clear()
        self._send_times.clear()

    def recv_fn(self):
        self.tr.record()
        t1 = self.time_fn()
        res = None
        while True:
            frame_id_next, frame = self.client.get_frame()
            if frame_id_next is not None:
                if self._frame_id_prev is None:
                    frame_id_delta = frame_id_next
                    self._frame_id_prev = frame_id_next
                elif frame_id_next != self._frame_id_prev:
                    frame_id_delta = frame_id_next - self._frame_id_prev
                    self._frame_id_prev = frame_id_next
                else:
                    continue
                self.tr.record('recv_fn_wait')

                frame = torch.from_numpy(frame).permute(2, 0, 1)[None, [2, 1, 0]]
                self.tr.record('recv_fn_tensor')

                frame = frame.to(dtype=torch.float16, device=f'cuda:{self.client.device_id}')
                self.tr.record('recv_fn_cuda_half')

                frame = resize(
                    frame,
                    (frame.size(-2) // self.client.resize_ratio, frame.size(-1) // self.client.resize_ratio),
                )[None]
                self.tr.record('recv_fn_resize')
                
                res = frame_id_delta, frame
                break
            else:
                break
        t2 = self.time_fn()
        self._recv_times.append((t1, max(t2-t1, self._min_time)))
        return res

    def send_fn(self, bbox: BBoxDict):
        self.tr.record()
        t1 = self.time_fn()
        coordinate = remove_pad_along(bbox['coordinate'][0, 0].cpu().numpy(), axis=0) * self.client.resize_ratio
        probability = bbox['probability'][0, 0, :coordinate.shape[0]].cpu().numpy()
        label = bbox['label'][0, 0, :coordinate.shape[0]].cpu().numpy()
        self.tr.record('send_fn_convert')
        self.client.send_result_to_server(coordinate, probability, label)
        self.tr.record('send_fn_send')
        t2 = self.time_fn()
        self._send_times.append((t1, max(t2-t1, self._min_time)))

    def process_fn(self, *args, **kwargs):
        self.tr.record()
        t1 = self.time_fn()
        res = self.model.inference(*args, **kwargs)
        self.tr.record('process_fn')
        t2 = self.time_fn()
        self._process_times.append((t1, max(t2-t1, self._min_time)))
        return res

    def time_fn(self):
        return (time.perf_counter() - self._seq_start_time) * 30 * self.server.sap_factor

    def infer_all(
            self,
            model: BaseModel,
            server: SAPServer,
            client: SAPClient,
            visualize_plot: bool = False,
            visualize_print: bool = False,
            output_dir: Optional[str] = None,
    ):
        self.model = model
        self.server = server
        self.client = client
        self.exp_tag = f'{model.neck.__class__.__name__} under {self.__class__.__name__}'
        self._max_frames = int(20 * server.sap_factor)

        # load coco dataset
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(server.ann_file)
        seqs = coco.dataset['sequences']

        # warm_up
        with torch.inference_mode():
            buffer = None
            example_input = model.example_input_array[0]
            example_input = {
                'image': example_input['image']['image'][:1].to(dtype=torch.float16, device=model.device),
                'past_clip_ids': example_input['image_clip_ids'][:1].to(dtype=torch.long, device=model.device),
                'future_clip_ids': example_input['bbox_clip_ids'][:1].to(dtype=torch.long, device=model.device),
            }
            for i in range(3):
                _, buffer = model.inference(
                    **example_input,
                    buffer=buffer,
                )

        torch.cuda.synchronize(device=client.device_id)

        with torch.inference_mode():
            for i, seq_id in enumerate(tqdm(seqs)):
                client.request_stream(seq_id)
                client.get_stream_start_time()  # wait
                self._reset()

                self.infer_sequence(
                    input_fn=self.recv_fn,
                    process_fn=self.process_fn,
                    output_fn=self.send_fn,
                    time_fn=self.time_fn,
                )
                client.stop_stream()
                self.tr.record('stop_stream')
                if visualize_plot:
                    self.plot_time(output_dir)
                    self.tr.record('plot')
                if visualize_print:
                    self.tr.print()

        filename = f'{int(time.time()) % 1000000000}.json'
        client.generate(filename)
        return server.get_result(filename)

    def infer_sequence(
            self,
            input_fn: Callable[[], Optional[Tuple[int, IMAGE]]],
            process_fn: Callable[[IMAGE, TIME, TIME, Optional[PYRAMID]], Tuple[BBoxDict, PYRAMID]],
            output_fn: Callable[[BBoxDict], None],
            time_fn: Callable[[], float],
    ) -> None: raise NotImplementedError()

    def plot_time(self, save_dir: Optional[str] = None):
        fig, ax = plt.subplots(dpi=600)
        ax.broken_barh(self._send_times, (6, 8), facecolors=(51 / 255, 57 / 255, 91 / 255))
        ax.broken_barh(self._process_times, (16, 8), facecolors=(93 / 255, 116 / 255, 162 / 255))
        ax.broken_barh(self._recv_times, (26, 8), facecolors=(142 / 255, 45 / 255, 48 / 255))
        ax.set_ylim(0, 40)
        ax.set_xlim(0, self._max_frames)
        ax.set_xlabel('Frames since start')
        ax.set_xticks(range(0, self._max_frames+1, int(self.server.sap_factor)))
        ax.set_yticks([10, 20, 30], labels=['Output', 'Infer', 'Load'])
        ax.xaxis.grid(True)
        ax.set_title(self.exp_tag)
        if save_dir:
            plt.savefig(Path(save_dir) / f'{self.exp_tag}.png')
        else:
            plt.show()
