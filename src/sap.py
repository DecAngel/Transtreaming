from initialization import root

import json
import platform
from pathlib import Path
from typing import  Tuple, Dict

import torch
import hydra
import torch.multiprocessing as mp
from omegaconf import DictConfig

from src.primitives.model import BaseModel
from src.primitives.sap import BaseSAPStrategy, SAPRunner
from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def sap(cfg: DictConfig) -> Tuple[Dict[str, float], Dict[str, float]]:
    log.info(f'Project root: {root}')

    if platform.system() != 'Linux':
        raise EnvironmentError('sAP evaluation is only supported on Linux!')

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating sap_strategy <{cfg.sap_strategy._target_}>")
    strategy: BaseSAPStrategy = hydra.utils.instantiate(cfg.sap_strategy)

    log.info(f"Instantiating sap_runner <{cfg.sap_runner._target_}>")
    runner: SAPRunner = hydra.utils.instantiate(cfg.sap_runner)

    device_id = cfg.get('device_id', 0)

    # load ckpt or pth
    path = Path(cfg.ckpt_path).resolve()
    log.info(f"Loading model from {str(path)}")
    if path.suffix == '.pth':
        model.load_from_pth(str(path))
    elif path.suffix == '.ckpt':
        model.load_from_ckpt(str(path))
    else:
        raise ValueError(f"Unsupported file type {path.suffix}")

    model = model.eval().half().to(torch.device(f'cuda:{device_id}'))

    return runner.run(model, strategy)


@hydra.main(version_base="1.3", config_path="../configs", config_name="sap.yaml")
def main(cfg: DictConfig) -> float:
    """Main entry point for running sap evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # run sap
    metric_dict, performance_dict = sap(cfg)

    log.info(f'sAP metric and performance: {json.dumps(metric_dict | performance_dict, indent=2)}')

    return metric_dict['AP5095']


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
