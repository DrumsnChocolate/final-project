
supported_losses = ['Dice', 'Focal']
supported_metrics = ['IoU', 'Dice', 'Focal']
def validate_cfg(cfg):
    # model
    assert cfg.model.name == 'sam', f'only able to train sam, not {cfg.model.name}'
    assert cfg.model.backbone in ['vit_h', 'vit_l', 'vit_b'], f'only able to train with one of vit_h, vit_l, vit_b, not {cfg.model.backbone}'
    # dataset
    permitted_datasets = ['ade20k', 'cbis-binary', 'cbis-multi']
    assert cfg.data.name in permitted_datasets, f'only able to train on one of {permitted_datasets}, not {cfg.data.name}'
    assert cfg.data.get('train') is not None, 'must specify train split'
    assert cfg.data.get('val') is not None, 'must specify val split'
    assert cfg.data.get('test') is not None, 'must specify test split'
    assert cfg.data.get('root') is not None, 'must specify data root directory'
    # finetuning
    assert cfg.model.finetuning.name == 'full', f'only able to train with full finetuning, not {cfg.model.finetuning.name}'
    # schedule
    assert cfg.schedule.get('epochs') is not None or cfg.schedule.get('iterations') is not None, 'must specify either epochs or iterations'
    assert cfg.schedule.get('epochs') is None or cfg.schedule.get('iterations') is None, 'cannot specify both epochs and iterations'
    assert cfg.schedule.get('val_interval') is not None, 'must specify val_interval'
    if cfg.schedule.get('epochs') is not None:
        assert cfg.schedule.epochs > 0, 'epochs must be positive'
        assert cfg.schedule.get('log_interval') is None, 'cannot specify log_interval with epochs'
    if cfg.schedule.get('iterations') is not None:
        assert cfg.schedule.iterations > 0, 'iterations must be positive'
        assert cfg.schedule.get('log_interval') is not None, 'must specify log_interval with iterations'
    # optimizer
    assert cfg.model.optimizer.name == 'sgd', f'only able to train with sgd, not {cfg.model.optimizer.name}'
    assert cfg.model.optimizer.get('lr') is not None, 'must specify learning rate'
    assert cfg.model.optimizer.get('wd') is not None, 'must specify weight decay'
    assert cfg.model.optimizer.get('momentum') is not None, 'must specify momentum'
    # device
    assert cfg.device in ['cpu', 'cuda'], "Only able to use cpu or cuda device"
    # loss
    assert cfg.model.get('loss') is not None, "model requires loss"
    if type(cfg.model.loss) == list:
        assert len(cfg.model.loss) > 0, "loss "
        for loss_item in cfg.model.loss:
            assert loss_item.name in supported_losses, f"loss should be one of {supported_losses}"
            assert type(loss_item.get('weight')) in [float, int], f"loss weight should be int or float"
    else:
        assert cfg.model.loss.name in supported_losses, f"loss should be one of {supported_losses}"
    # metrics
    assert type(cfg.model.metrics) == list, "metrics should be a list"
    for metric in cfg.model.metrics:
        assert metric.name in supported_metrics, f"metric should be one of {supported_metrics}"