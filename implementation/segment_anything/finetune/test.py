
from finetune.loss import build_loss_function
from models import build_model
from datasets.loaders import build_dataloaders
from train import parse_args, get_cfg, get_logger, test_epoch
from metrics import build_metric_functions


def test(cfg):
    logger = get_logger(cfg)
    dataloaders = build_dataloaders(cfg)
    model = build_model(cfg, logger)
    loss_function = build_loss_function(cfg)
    metric_functions = build_metric_functions(cfg)
    test_epoch(cfg, model, loss_function, metric_functions, dataloaders, logger)


def main():
    args = parse_args()
    cfg = get_cfg(args)
    test(cfg)

if __name__ == '__main__':
    main()
