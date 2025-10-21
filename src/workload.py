from initialization import root

import time
from typing import Any, Dict, Tuple

import hydra
import torch
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def workload(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Simulate workload on gpu

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    log.info(f'Project root: {root}')

    size = cfg.get('size', 1000)
    ratio = cfg.get('ratio', 0.5)
    device = cfg.get('device', 0)

    t = torch.rand(size, device=torch.device(f'cuda:{device}'))

    while True:
        try:
            current_time = time.perf_counter()
            t = torch.pow(torch.pow(t, 3), 1/3)
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - current_time
            time.sleep(elapsed_time * ratio)
        except KeyboardInterrupt:
            break


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
        pass


if __name__ == "__main__":
    main()
