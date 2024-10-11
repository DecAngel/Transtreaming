import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def workload(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """

    size = cfg.get('size', 1000)
    ratio = cfg.get('ratio', 0.5)
    device = cfg.get('device', 0)

    t = torch.rand(size, device=torch.device(f'cuda:{device}'))

    while True:
        current_time = time.perf_counter()
        t = torch.pow(torch.pow(t, 3), 1/3)
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - current_time
        time.sleep(elapsed_time * ratio)


@hydra.main(version_base="1.3", config_path="../configs", config_name="workload.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    try:
        workload(cfg)
    except KeyboardInterrupt:
        log.warning('Ctrl+C pressed. Stopping.')
    finally:
        return


if __name__ == "__main__":
    main()
