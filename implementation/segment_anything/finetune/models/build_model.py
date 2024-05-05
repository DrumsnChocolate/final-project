from finetune.models.build_sam import build_sam
from finetune.models.sam_wrapper import SamWrapper


def build_model(cfg, logger) -> SamWrapper:
    if cfg.model.name == 'sam':
        sam = build_sam(cfg)
        sam.to(cfg.device)
        model = sam
    else:
        raise NotImplementedError()  # we only support sam for now
    return SamWrapper(model, logger, cfg)