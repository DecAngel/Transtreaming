import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
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

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_services,
    task_wrapper,
)
from src.primitives.service import BaseService

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def serve(cfg: DictConfig):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

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
