from torchvision.transforms import InterpolationMode

supported_losses = ['IoU', 'Dice', 'Focal']
supported_loss_reductions = ['mean', 'sum']
supported_metrics = ['IoU', 'Dice']


def _is_collection_of(obj, _collection_type, _type):
    if not isinstance(obj, _collection_type):
        return False
    return all(isinstance(o, _type) for o in obj)

def _is_tuple_of(obj, _type):
    return _is_collection_of(obj, tuple, _type)


def _is_list_of(obj, _type):
    return _is_collection_of(obj, list, _type)


def _validate_model(cfg):
    permitted_models = ['sam']
    assert cfg.model.name in permitted_models, f'only able to train one of {permitted_models}, not {cfg.model.name}'
    permitted_backbones = ['vit_h', 'vit_l', 'vit_b']
    assert cfg.model.backbone in permitted_backbones, f'only able to train with one of {permitted_backbones}, not {cfg.model.backbone}'
    cfg.model.clip_grad_norm = cfg.model.get('clip_grad_norm')
    cfg.model.pixel_mean = cfg.model.get('pixel_mean')
    cfg.model.pixel_std = cfg.model.get('pixel_std')
    if cfg.model.pixel_mean is not None:
        assert _is_collection_of(cfg.model.pixel_mean, (list, tuple), (float, int)), 'pixel_mean should be a list or tuple of floats or ints'
        assert isinstance(cfg.model.pixel_mean, list), 'pixel_mean should be a list'
        assert len(cfg.model.pixel_mean) == 3, 'pixel_mean should have length 3'
    if cfg.model.pixel_std is not None:
        assert _is_collection_of(cfg.model.pixel_std, (list, tuple), (float, int)), 'pixel_std should be a list or tuple of floats or ints'
        assert len(cfg.model.pixel_std) == 3, 'pixel_std should have length 3'

def _validate_dataset_preprocess_step(step):
    supported_preprocess_steps = ['resize', 'clahe']
    assert step.name in supported_preprocess_steps, f'preprocess step should be one of {supported_preprocess_steps}'
    if step.name == 'resize':
        step.dimensions = step.get('dimensions')
        assert len(step.dimensions) == 2, 'resize dimensions should be length 2'
        assert _is_collection_of(step.dimensions, (tuple, list), (float, int)), 'resize dimensions should be list or tuple of floats or ints'
        step.mode = step.get('mode', 'bilinear')
        supported_resize_modes = [im._name_ for im in InterpolationMode]
        assert step.mode.upper() in supported_resize_modes, f'resize mode should be one of {supported_resize_modes}'
    if step.name == 'clahe':
        step.clip_limit = step.get('clip_limit', 40.0)
        assert isinstance(step.clip_limit, (float, int)), 'clahe clip_limit should be float or int'
        step.tile_grid_size = step.get('tile_grid_size', (8,8))
        assert len(step.tile_grid_size) == 2, 'clahe tile_grid_size should be length 2'
        assert _is_tuple_of(step.tile_grid_size, int), 'clahe tile_grid_size should be a tuple of ints'


def _validate_dataset_preprocess(cfg):
    cfg.data.preprocess = cfg.data.get('preprocess', [])
    assert isinstance(cfg.data.preprocess, list)
    for step in cfg.data.preprocess:
        _validate_dataset_preprocess_step(step)


def _validate_dataset(cfg):
    permitted_datasets = ['ade20k', 'cbis-binary', 'cbis-multi', 'zgt-binary']
    assert cfg.data.name in permitted_datasets, f'only able to load on one of {permitted_datasets}, not {cfg.data.name}'
    assert cfg.data.get('root') is not None, 'must specify data root directory'
    _validate_dataset_preprocess(cfg)


def _validate_loss(cfg):
    assert cfg.model.get('loss') is not None, "model requires loss"
    assert cfg.model.loss.get('parts') is not None, "loss requires parts"
    assert cfg.model.loss.get('reduction') in supported_loss_reductions, f"loss reduction should be one of {supported_loss_reductions}"
    assert type(cfg.model.loss.parts) == list, "loss parts should be a list"
    assert len(cfg.model.loss.parts) > 0, "loss parts should not be empty"
    for loss_item in cfg.model.loss.parts:
        assert loss_item.name in supported_losses, f"loss should be one of {supported_losses}"
        assert type(loss_item.get('weight')) in [float, int], f"loss weight should be int or float"


def _validate_metrics(cfg):
    assert type(cfg.model.metrics) == list, "metrics should be a list"
    for metric in cfg.model.metrics:
        assert metric.name in supported_metrics, f"metric should be one of {supported_metrics}"
        assert metric.get('invert', False) in [True, False], "metric.invert should be True or False"
        assert metric.get('per_mask') in [None, True, False], "metric.per_mask is an optional boolean, default False"
        metric.per_mask = metric.get('per_mask') or False


def _validate_cfg(cfg):
    # seed
    if cfg.get('seed') is None:
        cfg.seed = 218
    assert isinstance(cfg.seed, int), 'seed must be an integer'
    # device
    assert cfg.device in ['cpu', 'cuda'], "Only able to use cpu or cuda device"
    # model
    _validate_model(cfg)
    # dataset
    _validate_dataset(cfg)
    # loss
    _validate_loss(cfg)
    # metrics
    _validate_metrics(cfg)



def validate_cfg_train(cfg):
    _validate_cfg(cfg)
    # dataset
    assert cfg.data.get('train') is not None, 'must specify train split'
    assert cfg.data.get('val') is not None, 'must specify val split'
    # finetuning
    permitted_finetuning = ['full', 'vpt']
    assert cfg.model.finetuning.name in permitted_finetuning, f'only able to finetune with one of {permitted_finetuning}, not {cfg.model.finetuning.name}'
    if cfg.model.finetuning.name == 'vpt':
        cfg.model.finetuning.length = cfg.model.finetuning.get('length', 50)
        cfg.model.finetuning.dropout = cfg.model.finetuning.get('dropout', 0.1)
    # schedule
    cfg.schedule.epochs = cfg.schedule.get('epochs')
    cfg.schedule.iterations = cfg.schedule.get('iterations')
    assert cfg.schedule.epochs is not None or cfg.schedule.iterations is not None, 'must specify either epochs or iterations'
    assert cfg.schedule.epochs is None or cfg.schedule.iterations is None, 'cannot specify both epochs and iterations'
    cfg.schedule.val_interval = cfg.schedule.get('val_interval')
    assert cfg.schedule.val_interval is not None, 'must specify val_interval'
    if cfg.schedule.epochs is not None:
        assert cfg.schedule.epochs > 0, 'epochs must be positive'
        assert cfg.schedule.get('log_interval') is None, 'cannot specify log_interval with epochs'
    if cfg.schedule.iterations is not None:
        assert cfg.schedule.iterations >= 0, 'iterations must be non-negative'
        assert cfg.schedule.get('log_interval') is not None, 'must specify log_interval with iterations'
    cfg.schedule.stopper = cfg.schedule.get('stopper', None)
    if cfg.schedule.stopper is not None:
        assert cfg.schedule.stopper.get('name') in ['early_stopper'], 'stopper name must be early_stopper'
        assert cfg.schedule.stopper.get('patience') is not None, 'must specify early stopping patience'
        assert cfg.schedule.stopper.get('metric') is not None, 'must specify early stopping metric'
        assert cfg.schedule.stopper.get('split') in ['train', 'val'], 'early stopping split must be train or val'
        assert cfg.schedule.stopper.get('mode') in ['min', 'max'], 'early stopping mode must be min or max'
    cfg.schedule.scheduler = cfg.schedule.get('scheduler', None)
    if cfg.schedule.scheduler is not None:
        assert cfg.schedule.scheduler.get('name') in [
            'reduce_lr_on_plateau'], 'scheduler name must be reduce_lr_on_plateau'
        assert cfg.schedule.scheduler.get('mode') in ['min', 'max'], 'scheduler mode must be min or max'
        assert cfg.schedule.scheduler.get('factor') is not None, 'must specify scheduler factor'
        assert cfg.schedule.scheduler.get('patience') is not None, 'must specify scheduler patience'
        assert cfg.schedule.scheduler.get('threshold') is not None, 'must specify scheduler threshold'
        assert cfg.schedule.scheduler.get('split') in ['train', 'val'], 'scheduler split must be train or val'
        assert cfg.schedule.scheduler.get('metric') is not None, 'must specify scheduler metric'
    # optimizer
    assert cfg.model.optimizer.name in ['sgd',
                                        'adamw'], f'only able to train with sgd or adamw, not {cfg.model.optimizer.name}'
    assert cfg.model.optimizer.get('lr') is not None, 'must specify learning rate'
    if cfg.model.optimizer.name == 'sgd':
        assert cfg.model.optimizer.get('momentum') is not None, 'must specify momentum'


def validate_cfg_test(cfg):
    _validate_cfg(cfg)
    assert cfg.data.get('test') is not None, 'must specify test split'
