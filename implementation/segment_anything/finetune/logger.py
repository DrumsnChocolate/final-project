import time

import numpy as np
from prodict import Prodict
import json
import os.path as osp
import os


class Logger(Prodict):
    log_dir: str
    cfg: Prodict

    def __init__(self, cfg, *args, test=False, **kwargs):
        if cfg.get('timestamp') is None:
            # we try to prevent collisions if multiple experiments are run at the same time
            timestamp_in_use = True
            while timestamp_in_use:
                cfg.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
                timestamp_in_use = osp.exists(osp.join(cfg.out_dir, cfg.timestamp))
        timestamp = cfg.timestamp
        log_dir = osp.join(cfg.out_dir, timestamp)
        if cfg.get('sub_dir') is not None:
            log_dir = osp.join(log_dir, cfg.sub_dir)

        super().__init__(self, *args, cfg=cfg, log_dir=log_dir, **kwargs)
        assert self.cfg is not None
        assert self.log_dir is not None
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file = open(osp.join(self.log_dir, 'metrics.json'), 'w')
        self.text_file = open(osp.join(self.log_dir, 'logs.txt'), 'w')

    def _log_dict(self, d: dict, to_file: bool = True):
        d_json = json.dumps(d)
        print(d_json)
        if not to_file:
            return
        self.metrics_file.write(f'{d_json}\n')

    def _log_string(self, s: str, to_file: bool = True):
        print(s)
        if not to_file:
            return
        self.text_file.write(f'{s}\n')

    def log(self, s: str, to_file: bool = True):
        self._log_string(s, to_file=to_file)


class EpochLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_metrics = []

    def log_batch_metrics(self, metrics: dict):
        self.epoch_metrics.append(metrics)

    def log_epoch(self, epoch: int, split: str = 'train'):
        avg_metrics = {
            k: (np.sum([m[k] for m in self.epoch_metrics], axis=0) / len(self.epoch_metrics)).tolist() for k in
            self.epoch_metrics[0].keys()
        }
        avg_metrics['epoch'] = epoch
        avg_metrics['split'] = split
        self._log_dict(avg_metrics)
        self.epoch_metrics = []


class IterationLogger(EpochLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_metrics = []

    def _log_average_iteration_metrics(self):
        avg_metrics = {
            k: (np.sum([m[k] for m in self.iteration_metrics], axis=0) / len(self.iteration_metrics)).tolist()
            for k in self.iteration_metrics[0].keys()
        }
        avg_metrics['split'] = 'train'  # iteration logging is always train, never val.
        self._log_dict(avg_metrics)
        self.iteration_metrics = []

    def log_iteration_metrics(self, metrics: dict, iteration: int):
        self.iteration_metrics.append((metrics))
        if iteration % self.cfg.schedule.log_interval != 0:
            return
        self._log_average_iteration_metrics()
