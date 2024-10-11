import contextlib
import io
import pickle
import time
from threading import Thread
from typing import Callable, Tuple, Optional, List

import numpy as np
from PIL import Image
from kornia.geometry import resize
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from torchvision.transforms import PILToTensor
from torchvision.transforms.v2 import ToTensor
from tqdm import tqdm

from src.primitives.batch import IMAGE, BBoxDict, PYRAMID, TIME, SIZE
from src.primitives.model import BaseModel
from src.utils.array_operations import remove_pad_along
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

import json
import signal
import sys
import os
import socket
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from sap_toolkit.client import EvalClient
from sap_toolkit.generated import eval_server_pb2

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
    def __init__(self, server: SAPServer):
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
        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    def send_result_shm(self, bboxes, bbox_scores, labels, delay: float = 0.0):
        if delay > 1e-4:
            time.sleep(delay)
        timestamp = time.perf_counter()
        super().send_result_shm(bboxes, bbox_scores, labels, timestamp)

    def send_result_to_server(self, bboxes, bbox_scores, labels, delay: float = 0.0):
        if self.result_thread:
            self.result_thread.join()
        self.result_thread = Thread(target=self.send_result_shm, args=(bboxes, bbox_scores, labels, delay))
        self.result_thread.start()

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
    def __init__(self, past_length: int, future_length: int):
        self.past_length = past_length
        self.future_length = future_length
        self.past_clip_ids = torch.tensor([list(range(-past_length+1, 1))], dtype=torch.long)
        self.future_clip_ids = torch.tensor([list(range(1, future_length+1))], dtype=torch.long)

        # constant
        self._sap_factor = None
        self._resize_ratio = None
        self._prev_frame_id = None
        self._device = None
        self._start_time = None

        # time list
        self._recv_time = []
        self._proc_time = []
        self._send_time = []

    def transform_fn(self, frame: np.ndarray):
        frame = torch.from_numpy(frame).permute(2, 0, 1)[None, [2, 1, 0]]
        frame = frame.to(dtype=torch.float16, device=self._device)
        return resize(
            frame,
            (frame.size(-2) // self._resize_ratio, frame.size(-1) // self._resize_ratio),
        )[None]

    def time_fn(self):
        return (time.perf_counter() - self._start_time) * 30 * self._sap_factor

    def recv_fn(self, client: SAPClient) -> Optional[Tuple[int, int, IMAGE]]:
        # initial
        res = None
        while True:
            next_frame_id, frame = client.get_frame()

            if next_frame_id is not None:
                if self._prev_frame_id is None:
                    frame_id_delta = next_frame_id
                    self._prev_frame_id = next_frame_id
                elif next_frame_id != self._prev_frame_id:
                    frame_id_delta = next_frame_id - self._prev_frame_id
                    self._prev_frame_id = next_frame_id
                else:
                    continue

                t1 = self.time_fn()
                frame = self.transform_fn(frame)
                res = next_frame_id, frame_id_delta, frame
                t2 = self.time_fn()
                self._recv_time.append((t1, t2 - t1))
                break
            else:
                break

        return res

    def proc_fn(
            self,
            image: IMAGE,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
            buffer: Optional[PYRAMID],
            model: BaseModel,
    ) -> Tuple[BBoxDict, PYRAMID]:
        t1 = self.time_fn()
        res = model.inference(
            image=image,
            past_clip_ids=past_clip_ids,
            future_clip_ids=future_clip_ids,
            buffer=buffer,
        )
        t2 = self.time_fn()
        self._proc_time.append((t1, t2 - t1))
        return res

    def proc_backbone_neck_fn(
            self,
            image: IMAGE,
            past_clip_ids: TIME,
            future_clip_ids: TIME,
            buffer: Optional[PYRAMID],
            model: BaseModel,
    ) -> Tuple[PYRAMID, PYRAMID]:
        t1 = self.time_fn()
        res = model.inference_backbone_neck(
            image=image,
            past_clip_ids=past_clip_ids,
            future_clip_ids=future_clip_ids,
            buffer=buffer,
        )
        t2 = self.time_fn()
        self._proc_time.append((t1, t2 - t1))
        return res

    def proc_head_fn(
            self,
            features_f: PYRAMID,
            shape: Optional[SIZE],
            model: BaseModel,
    ) -> BBoxDict:
        t1 = self.time_fn()
        res = model.inference_head(
            features_f=features_f,
            shape=shape
        )
        t2 = self.time_fn()
        self._proc_time[-1] = self._proc_time[-1][0], self._proc_time[-1][1] + t2 - t1
        return res

    def send_fn(self, bbox: BBoxDict, client: SAPClient, delay: float = 0.0) -> None:
        t1 = self.time_fn()
        coordinate = remove_pad_along(
            bbox['coordinate'][0, 0].cpu().numpy(),
            axis=0
        ) * self._resize_ratio
        probability = bbox['probability'][0, 0, :coordinate.shape[0]].cpu().numpy()
        label = bbox['label'][0, 0, :coordinate.shape[0]].cpu().numpy()
        client.send_result_to_server(coordinate, probability, label, delay=delay)
        t2 = self.time_fn()
        self._send_time.append((t1, t2 - t1))

    def infer(
            self,
            model: BaseModel,
            client: SAPClient,
            start_time: float,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
        self._start_time = start_time
        self.infer_step(model, client)
        return self._recv_time, self._proc_time, self._send_time

    def infer_prepare(
            self,
            model: BaseModel,
            client: SAPClient,
            sap_factor: float,
            dataset_resize_ratio: int,
            demo_input: List[np.ndarray],
    ) -> None:
        self.past_clip_ids = self.past_clip_ids.to(model.device)
        self.future_clip_ids = self.future_clip_ids.to(model.device)
        self._sap_factor = sap_factor
        self._resize_ratio = dataset_resize_ratio
        self._prev_frame_id = None
        self._device = model.device
        self._recv_time.clear()
        self._proc_time.clear()
        self._send_time.clear()

        buffer = None
        for frame in demo_input:
            frame = self.transform_fn(frame)
            _, buffer = model.inference(frame, self.past_clip_ids, self.future_clip_ids, buffer)

    def infer_step(
            self,
            model: BaseModel,
            client: SAPClient,
    ) -> None:
        raise NotImplementedError()


class SAPRunner:
    def __init__(
            self,
            model: BaseModel,
            strategy: BaseSAPStrategy,

            data_dir: str,
            ann_file: str,
            output_dir: str,
            demo_dir: str,

            sap_tag: Optional[str] = None,
            sap_factor: float = 1.0,
            dataset_resize_ratio: int = 2,

            visualize: bool = False,
            vis_max_frame: int = 20,
    ):
        self.model = model
        self.strategy = strategy

        self.data_dir = data_dir
        self.ann_file = ann_file
        self.output_dir = output_dir
        self.demo_dir = demo_dir

        self.sap_tag = sap_tag
        self.sap_factor = sap_factor
        self.dataset_resize_ratio = dataset_resize_ratio

        self.visualize = visualize
        self.vis_max_frame = vis_max_frame
        self.vis_count = 0

    def plot_time(
            self,
            recv_times: List[Tuple[float, float]],
            proc_times: List[Tuple[float, float]],
            send_times: List[Tuple[float, float]],
    ):
        fig, ax = plt.subplots(dpi=600)
        # adjust display
        recv_times = [(i, max(j, 0.05)) for i, j in recv_times]
        proc_times = [(i, max(j, 0.05)) for i, j in proc_times]
        send_times = [(i, max(j, 0.05)) for i, j in send_times]

        ax.broken_barh(send_times, (6, 8), facecolors=(51 / 255, 57 / 255, 91 / 255))
        ax.broken_barh(proc_times, (16, 8), facecolors=(93 / 255, 116 / 255, 162 / 255))
        ax.broken_barh(recv_times, (26, 8), facecolors=(142 / 255, 45 / 255, 48 / 255))
        ax.set_ylim(0, 40)
        ax.set_xlim(0, self.vis_max_frame * int(self.sap_factor))
        ax.set_xlabel('Frames since start')
        ax.set_xticks(range(0, self.vis_max_frame * int(self.sap_factor) + 1, int(self.sap_factor)))
        ax.set_yticks([10, 20, 30], labels=['Output', 'Infer', 'Load'])
        ax.xaxis.grid(True)
        ax.set_title(self.sap_tag)
        if self.output_dir:
            path = Path(self.output_dir).joinpath('figures')
            path.mkdir(exist_ok=True)
            plt.savefig(str(path.joinpath(f'{self.sap_tag} {self.vis_count}.png')))
            path.joinpath(f'{self.sap_tag} {self.vis_count}.pkl').write_bytes(pickle.dumps(ax))
            self.vis_count += 1
        else:
            plt.show()

    def run(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        all_recv = []
        all_proc = []
        all_send = []
        with SAPServer(
                data_dir=self.data_dir,
                ann_file=self.ann_file,
                output_dir=self.output_dir,
                sap_factor=self.sap_factor
        ) as server:
            client = SAPClient(server)
            try:
                # load coco dataset
                with contextlib.redirect_stdout(io.StringIO()):
                    coco = COCO(server.ann_file)
                seqs = coco.dataset['sequences']

                # warm_up
                demo_input = []
                for p in list(Path(self.demo_dir).iterdir())[:3]:
                    demo_input.append(np.asarray(Image.open(p)).copy())

                torch.cuda.synchronize(device=self.model.device)

                with torch.inference_mode():
                    for i, seq_id in enumerate(tqdm(seqs)):
                        # init
                        client.request_stream(seq_id)   # request

                        # during server loading images, do initialization
                        # cause error if server loads too fast, but highly impossible
                        # initialize sap
                        self.strategy.infer_prepare(
                            model=self.model,
                            client=client,
                            sap_factor=self.sap_factor,
                            dataset_resize_ratio=self.dataset_resize_ratio,
                            demo_input=demo_input,
                        )

                        client.get_stream_start_time()  # wait
                        recv_time, proc_time, send_time = self.strategy.infer(self.model, client, time.perf_counter())

                        all_recv.extend([t[1] for t in recv_time])
                        all_proc.extend([t[1] for t in proc_time])
                        all_send.extend([t[1] for t in send_time])

                        for t1, t2, t3 in zip(recv_time[:40], proc_time[:40], send_time[:40]):
                            print(f'{(t1[1]+t2[1]+t3[1])*100/3:.4f}')

                        if self.visualize and i < 4:
                            self.plot_time(recv_time, proc_time, send_time)

                        client.stop_stream()

                performance_dict = {
                    'max recv_time': np.max(all_recv),
                    'max proc_time': np.max(all_proc),
                    'max send_time': np.max(all_send),
                    'min recv_time': np.min(all_recv),
                    'min proc_time': np.min(all_proc),
                    'min send_time': np.min(all_send),
                    'mean recv_time': np.mean(all_recv),
                    'mean proc_time': np.mean(all_proc),
                    'mean send_time': np.mean(all_send),
                    'std recv_time': np.std(all_recv),
                    'std proc_time': np.std(all_proc),
                    'std send_time': np.std(all_send),
                }
                filename = f'{int(time.time()) % 1000000000}.json'
                client.generate(filename)
                return server.get_result(filename), performance_dict

            except KeyboardInterrupt as e:
                log.warning('Ctrl+C detected. Shutting down sAP server & client.', exc_info=e)
                raise
            finally:
                client.close()
