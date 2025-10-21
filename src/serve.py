from initialization import root

import time
from typing import List

import hydra
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    instantiate_services,
    task_wrapper,
)
from src.primitives.service import BaseService

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def serve(cfg: DictConfig):
    """Run services

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    log.info(f'Project root: {root}')
    log.info("Instantiating services...")
    services: List[BaseService] = instantiate_services(cfg.get("services"))

    for s in services:
        s.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.warning("Caught Keyboard Interrupt! Shutting down...")
    finally:
        for s in services:
            s.shutdown()
        for s in services:
            s.join()
        log.info("Shutdown finished.")


@hydra.main(version_base="1.3", config_path="../configs", config_name="serve.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for serving.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    serve(cfg)


if __name__ == "__main__":
    main()
