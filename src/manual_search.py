import functools
import random
import logging
from typing import Literal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


import optuna
import threading


exp_type: Literal['s', 'm', 'l'] = 'l'

exp_name = f'corrdiff_drfpn_{exp_type}'
speed = 's-3-2-101'
batch_size = 2 if exp_type == 'l' else 4
num_gpus = 4
num_jobs = 4

if exp_type == 'l':
    search_space = {
        'seed': ('categorical', list(range(12340, 12350))),
        'model.optim.lr': ('float', 0.001, 0.01),
        'model.optim.momentum': ('float', 0.88, 0.93),
        'model.optim.weight_decay': ('float', 6e-4, 1.2e-3),
        'model.optim.neck_coef': ('float', 5.0, 15.0),
        'model.optim.head_coef': ('float', 5.0, 15.0),
        # 'model.head.conf_thre': ('float', 0.001, 0.005),
        # 'model.head.nms_thre': ('float', 0.4, 0.7),
        # 'model.head.ignore_thr': ('float', 0.7, 0.9),
        # 'model.head.ignore_value': ('float', 1.2, 1.6),
    }
elif exp_type == 'm':
    search_space = {
        'seed': ('categorical', list(range(12340, 12350))),
        'model.optim.lr': ('float', 0.001, 0.01),
        'model.optim.momentum': ('float', 0.90, 0.95),
        'model.optim.weight_decay': ('float', 8e-4, 1.2e-3),
        'model.optim.neck_coef': ('float', 5.0, 15.0),
        'model.optim.head_coef': ('float', 5.0, 15.0),
        # 'model.head.conf_thre': ('float', 0.001, 0.005),
        # 'model.head.nms_thre': ('float', 0.4, 0.7),
        # 'model.head.ignore_thr': ('float', 0.7, 0.9),
        # 'model.head.ignore_value': ('float', 1.2, 1.6),
    }
else:
    search_space = {
        'seed': ('categorical', list(range(12340, 12350))),
        'model.optim.lr': ('float', 0.001, 0.004),
        'model.optim.momentum': ('float', 0.90, 0.95),
        'model.optim.weight_decay': ('float', 8e-4, 1.2e-3),
        'model.optim.neck_coef': ('float', 5.0, 15.0),
        'model.optim.head_coef': ('float', 5.0, 15.0),
        # 'model.head.conf_thre': ('float', 0.001, 0.005),
        # 'model.head.nms_thre': ('float', 0.4, 0.7),
        # 'model.head.ignore_thr': ('float', 0.7, 0.9),
        # 'model.head.ignore_value': ('float', 1.2, 1.6),
    }

python_prefix = f"nohup python src/train.py speed={speed} experiment={exp_name} trainer.devices=[{{device}}] datamodule.batch_size={batch_size} "
# python_suffix = " &> logs/search/key_{key}.log & \nsleep 10\nnvidia-smi -lms 10000\n"
python_suffix = " &> logs/search/key_{key}.log &\n"

res_dict = {}


def parse(d):
    return python_prefix + ' '.join([f'{k}={str(v).replace(" ", "")}' for k, v in d.items()]) + python_suffix


lock = threading.Lock()
job_num = 0
def objective(trial):
    global lock, job_num
    values = {
        k: trial.suggest_categorical(k, *args[1:]) if args[0] == 'categorical' else trial.suggest_float(k, *args[1:])
        for k, args in search_space.items()
    }
    key = f'{random.randint(0, 999999):06d}'

    with lock:
        s = parse(values)
        s = s.format(key=key, device=job_num % num_gpus)
        print(key)
        print(s)
        job_num += 1

    while True:
        if key in res_dict:
            return float(res_dict[key])


sampler = optuna.samplers.TPESampler(n_startup_trials=num_jobs*1)
study = optuna.create_study(direction='maximize', sampler=sampler)
t = threading.Thread(target=functools.partial(study.optimize, objective, n_trials=num_jobs*3, n_jobs=num_jobs))
t.start()

while True:
    s = input('input result')
    if s == 'exit':
        break
    else:
        try:
            k, v = s.split('=')
            res_dict[k] = v
        except Exception:
            log.warning('input error, retry')

t.join()
