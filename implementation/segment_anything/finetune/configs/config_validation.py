
def validate_cfg(cfg):
    # model
    assert cfg.model.name == 'sam', f'only able to train sam, not {cfg.model.name}'
    assert cfg.model.backbone in ['vit_h', 'vit_l', 'vit_b'], f'only able to train with one of vit_h, vit_l, vit_b, not {cfg.model.backbone}'
    # dataset
    assert cfg.data.name == 'ade20k', f'only able to train on ade20k, not {cfg.data.name}'
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
    # optimizer
    assert cfg.model.optimizer.name == 'sgd', f'only able to train with sgd, not {cfg.model.optimizer.name}'
    assert cfg.model.optimizer.get('lr') is not None, 'must specify learning rate'
    assert cfg.model.optimizer.get('wd') is not None, 'must specify weight decay'
    assert cfg.model.optimizer.get('momentum') is not None, 'must specify momentum'