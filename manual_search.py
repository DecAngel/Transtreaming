import functools
import random
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


import optuna
import threading


python_prefix = "python src/train.py experiment=traque_drfpn_s trainer.devices=[{device}] datamodule.batch_size=4 "
# python_suffix = " &> logs/lsf/key_{key}.log & \nsleep 10\nnvidia-smi\n"
python_suffix = "\n"
num_gpus = 2
num_jobs = 4

search_space = {
    'train': ('categorical', ['-3-2-101', '-3-2-10123', '-4-2-10124']),
    'seed': ('categorical', list(range(12340, 12350))),
    'model.optim.lr': ('float', 0.001, 0.003),
    'model.optim.momentum': ('float', 0.9, 0.93),
    'model.optim.weight_decay': ('float', 8e-4, 1.2e-3),
#    'model.neck.num_heads': ('categorical', [1, 2, 4, 8]),
#    'model.neck.window_size': ('categorical', [4, 8, 12]),
    'model.head.conf_thre': ('float', 0.001, 0.01),
    'model.head.nms_thre': ('float', 0.4, 0.6),
    'model.head.ignore_thr': ('float', 0.5, 0.9),
    'model.head.ignore_value': ('float', 1.4, 1.8),
}

res_dict = {}


def replace(d):
    if 'train' in d.keys():
        train = d.pop('train')
        if train == '-3-2-101':
            d['datamodule.train_data_source.image_clip_ids'] = [-3, -2, -1, 0]
            d['datamodule.train_data_source.bbox_clip_ids'] = [0, 1]
            d['datamodule.val_data_source.image_clip_ids'] = [-3, -2, -1, 0]
            d['datamodule.val_data_source.bbox_clip_ids'] = [1]
            d['datamodule.test_data_source.image_clip_ids'] = [-3, -2, -1, 0]
            d['datamodule.test_data_source.bbox_clip_ids'] = [1]
            d['model.metric.future_time_constant'] = [1]
            d['sap_strategy.past_length'] = 2
        elif train == '-3-2-10123':
            d['datamodule.train_data_source.image_clip_ids'] = [-3, -2, -1, 0]
            d['datamodule.train_data_source.bbox_clip_ids'] = [0, 1, 2, 3]
            d['datamodule.val_data_source.image_clip_ids'] = [-3, -2, -1, 0]
            d['datamodule.val_data_source.bbox_clip_ids'] = [1, 2, 3]
            d['datamodule.test_data_source.image_clip_ids'] = [-3, -2, -1, 0]
            d['datamodule.test_data_source.bbox_clip_ids'] = [1, 2, 3]
            d['model.metric.future_time_constant'] = [1, 2, 3]
            d['sap_strategy.past_length'] = 4
        elif train == '-4-2-10124':
            d['datamodule.train_data_source.image_clip_ids'] = [-4, -2, -1, 0]
            d['datamodule.train_data_source.bbox_clip_ids'] = [0, 1, 2, 4]
            d['datamodule.val_data_source.image_clip_ids'] = [-4, -2, -1, 0]
            d['datamodule.val_data_source.bbox_clip_ids'] = [1, 2, 4]
            d['datamodule.test_data_source.image_clip_ids'] = [-4, -2, -1, 0]
            d['datamodule.test_data_source.bbox_clip_ids'] = [1, 2, 4]
            d['model.metric.future_time_constant'] = [1, 2, 4]
            d['sap_strategy.past_length'] = 4
        else:
            raise ValueError()

    if 'model.neck.window_size' in d.keys():
        window_size = d.pop('model.neck.window_size')
        d['model.neck.window_size'] = [window_size, window_size]

    return d


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
        s = parse(replace(values))
        s = s.format(key=key, device=job_num % num_gpus)
        print(key)
        print(s)
        job_num += 1

    while True:
        if key in res_dict:
            return float(res_dict[key])


sampler = optuna.samplers.TPESampler(n_startup_trials=num_jobs*2)
study = optuna.create_study(direction='maximize', sampler=sampler)
t = threading.Thread(target=functools.partial(study.optimize, objective, n_trials=num_jobs*4, n_jobs=num_jobs))
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

