from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
torch.set_float32_matmul_precision('high')

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

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
from src.primitives.sap_strategy import BaseSAPStrategy
from src.primitives.batch import IMAGE, BBoxDict
from src.utils.array_operations import remove_pad_along
from src.utils.time_recorder import TimeRecorder


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
        while True:
            output = self.proc.stdout.readline()
            log.info(output)
            if '=' in output:
                o = output.rsplit('=')
                try:
                    output_dict['='.join(o[:-1])] = float(o[-1])
                except ValueError:
                    pass
            if 'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]' in output:
                break
        return output_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.communicate('close\n')
        os.remove(self.config_path)
        self.proc = None
        self.config_path = None


class SAPClient(EvalClient):
    def generate(self, results_file='results.json'):
        self.result_stub.GenResults(eval_server_pb2.String(value=results_file))

    def close(self):
        self.result_channel.close()
        self.existing_shm.close()
        self.results_shm.close()
        print('Shutdown', flush=True)





@task_wrapper
def sap(cfg: DictConfig) -> Dict[str, float]:
    if platform.system() != 'Linux':
        raise EnvironmentError('sAP evaluation is only supported on Linux!')

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating sap_strategy <{cfg.sap_strategy._target_}>")
    strategy: BaseSAPStrategy = hydra.utils.instantiate(cfg.sap_strategy)

    # load ckpt or pth
    path = Path(cfg.ckpt_path).resolve()
    log.info(f"Loading model from {str(path)}")
    if path.suffix == '.pth':
        model.load_from_pth(str(path))
    elif path.suffix == '.ckpt':
        model.load_from_ckpt(str(path))
    else:
        raise ValueError(f"Unsupported file type {path.suffix}")

    data_dir = cfg.get('data_dir')
    ann_file = cfg.get('ann_file')
    output_dir = cfg.get('output_dir')

    sap_factor = cfg.get('sap_factor', 1.0)
    dataset_resize_ratio = cfg.get('dataset_resize_ratio', 2)
    dataset_fps = cfg.get('dataset_fps', 30)
    device_id = cfg.get('device_id', 0)
    model = model.eval().half().to(torch.device(f'cuda:{device_id}'))

    with SAPServer(
            data_dir=data_dir,
            ann_file=ann_file,
            output_dir=output_dir,
            sap_factor=sap_factor
    ) as server:
        client_state = (
            mp.Value('i', -1, lock=True),
            mp.Event(),
            mp.Manager().dict(),
        )
        client = SAPClient(json.loads(server.config_path.read_text()), client_state, verbose=True)
        client.stream_start_time = mp.Value('d', 0.0, lock=True)
        signal.signal(signal.SIGINT, client.close)
        signal.signal(signal.SIGTERM, client.close)

        try:
            # load coco dataset
            with contextlib.redirect_stdout(io.StringIO()):
                coco = COCO(ann_file)
            seqs = coco.dataset['sequences']

            frame_id_prev = None
            def recv_fn(tr: TimeRecorder) -> Optional[Tuple[int, IMAGE]]:
                nonlocal frame_id_prev
                tr.record('others')
                while True:
                    frame_id_next, frame = client.get_frame()
                    if frame_id_next is not None:
                        if frame_id_prev is None:
                            frame_id_delta = frame_id_next
                            frame_id_prev = frame_id_next
                        elif frame_id_next != frame_id_prev:
                            frame_id_delta = frame_id_next - frame_id_prev
                            frame_id_prev = frame_id_next
                        else:
                            continue
                        tr.record('recv_fn_wait')

                        frame = torch.from_numpy(frame).permute(2, 0, 1)[None, [2, 1, 0]]
                        tr.record('recv_fn_tensor')

                        frame = frame.to(dtype=torch.float16, device=f'cuda:{device_id}')
                        tr.record('recv_fn_cuda_half')

                        frame = resize(frame, (600, 960), interpolation='bilinear')[None]
                        tr.record('recv_fn_resize')

                        # frame = F.interpolate(frame, scale_factor=(1/dataset_resize_ratio, 1/dataset_resize_ratio), mode='bilinear')
                        # frame = resize_image(frame)[..., [2, 1, 0]].transpose(2, 0, 1)
                        return frame_id_delta, frame
                    else:
                        return None

            def send_fn(bbox: BBoxDict, tr: TimeRecorder) -> None:
                tr.record('others')
                coordinate = remove_pad_along(bbox['coordinate'][0, 0].cpu().numpy(), axis=0) * dataset_resize_ratio
                probability = bbox['probability'][0, 0, :coordinate.shape[0]].cpu().numpy()
                label = bbox['label'][0, 0, :coordinate.shape[0]].cpu().numpy()
                tr.record('send_fn_convert')
                client.send_result_to_server(coordinate, probability, label)
                tr.record('send_fn_send')
                return None

            def process_fn(*args, tr: TimeRecorder, **kwargs):
                tr.record('others')
                res = model.inference(*args, **kwargs)
                tr.record('process_fn')
                return res

            def time_fn(start_time: float):
                return (time.perf_counter() - start_time) * dataset_fps * sap_factor

            # warm_up

            with torch.inference_mode():
                buffer = None
                example_input = {
                    'image': torch.randint(0, 255, size=(1, 1, 3, 600, 960), dtype=torch.float16, device=model.device),
                    'past_clip_ids': torch.tensor([[-1, 0]], dtype=torch.long, device=model.device),
                    'future_clip_ids': torch.tensor([[1]], dtype=torch.long, device=model.device),
                }
                for i in range(3):
                    _, buffer = model.inference(
                        **example_input,
                        buffer=buffer,
                    )

            torch.cuda.synchronize(device=model.device)

            with torch.inference_mode():
                for i, seq_id in enumerate(tqdm(seqs)):
                    with TimeRecorder(f'seq_{i}', mode='avg') as tr:
                        frame_id_prev = None
                        client.request_stream(seq_id)
                        client.get_stream_start_time()  # wait
                        t_start = time.perf_counter()
                        tr.record('request_stream')
                        strategy.infer_sequence(
                            input_fn=functools.partial(recv_fn, tr=tr),
                            process_fn=functools.partial(process_fn, tr=tr),
                            output_fn=functools.partial(send_fn, tr=tr),
                            time_fn=functools.partial(time_fn, start_time=t_start)
                        )
                        client.stop_stream()
                        if i == 0:
                            strategy.plot_process_time()

            filename = f'{int(time.time()) % 1000000000}.json'
            client.generate(filename)
            return server.get_result(filename)

        except KeyboardInterrupt as e:
            log.warning('Ctrl+C detected. Shutting down sAP server & client.', exc_info=e)
            raise
        finally:
            client.close()


@hydra.main(version_base="1.3", config_path="../configs", config_name="sap.yaml")
def main(cfg: DictConfig) -> Dict[str, float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # run sap
    metric_dict = sap(cfg)

    return metric_dict


if __name__ == "__main__":
    main()
