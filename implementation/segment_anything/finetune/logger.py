from prodict import Prodict
import json
import os.path as osp
import os


class Logger(Prodict):
    log_dir: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.log_dir is not None
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file = open(osp.join(self.log_dir, 'metrics.json'), 'w')
        self.text_file = open(osp.join(self.log_dir, 'logs.txt'), 'w')

    def log_dict(self, d: dict):
        d_json = json.dumps(d)
        print(d_json)
        self.metrics_file.write(f'{d_json}\n')

    def log_string(self, s: str):
        print(s)
        self.text_file.write(f'{s}\n')

    def log(self, s: str):
        self.log_string(s)
