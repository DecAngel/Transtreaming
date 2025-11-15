import rootutils

root = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
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


import torch

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


import math
import functools
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda *args: type(args[0])(sum(float(a) for a in args)))
OmegaConf.register_new_resolver("multiply", lambda *args: functools.reduce(lambda a, b: type(a)(float(a) * float(b)), args))
OmegaConf.register_new_resolver("subtract", lambda a, b: type(a)(float(a) - float(b)))
OmegaConf.register_new_resolver("divide", lambda a, b: type(a)(float(a) / float(b)))
OmegaConf.register_new_resolver("round", lambda a: int(round(a)))
OmegaConf.register_new_resolver("ceil", lambda a: int(math.ceil(a)))
OmegaConf.register_new_resolver("floor", lambda a: int(math.floor(a)))
OmegaConf.register_new_resolver("join", lambda a, b: str(b).join([str(i) for i in a]))
