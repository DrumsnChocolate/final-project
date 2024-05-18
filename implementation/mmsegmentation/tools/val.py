import argparse
import os
import os.path as osp

from mmengine import DictAction, Config
from mmengine.runner import find_latest_checkpoint, Runner


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg validate a model')
    parser.add_argument('train_log_dir', help='log dir for training run we are testing')
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--save-dir',
        help='directory where painted images will be saved. '
             'If specified, it will be automatically saved '
             'to the save_dir relative from where executed')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if not args.show:
            visualization_hook['draw_gt'] = False
        if args.save_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.save_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def main():
    args = parse_args()
    print(args)

    # load config
    args.config = osp.join(args.train_log_dir, 'config.py')
    cfg = Config.fromfile(args.config)
    cfg.train_log_dir = args.train_log_dir
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join(args.train_log_dir, 'val')
    cfg.load_from = find_latest_checkpoint(cfg.train_log_dir)

    if args.show or args.save_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start validating
    runner.val()


if __name__ == '__main__':
    main()
