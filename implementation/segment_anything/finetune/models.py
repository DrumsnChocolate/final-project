from segment_anything import sam_model_registry


def build_sam(cfg):
    if cfg.model.finetuning.name == 'full':
        return sam_model_registry[cfg.model.backbone](checkpoint=cfg.model.checkpoint)
    raise NotImplementedError
    # todo: implement vpt, for this we will need to change some things about the model classes,
    # and about the registry? I think.
    # todo: use any other model properties from the config?


def build_model(cfg):
    if cfg.model.name == 'sam':
        return build_sam(cfg)
    else:
        raise NotImplementedError()  # we only support sam for now
