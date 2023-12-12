import torch

from logger import Logger
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from torchvision.transforms.functional import resize

def build_sam(cfg):
    if cfg.model.finetuning.name == 'full':
        return sam_model_registry[cfg.model.backbone](checkpoint=cfg.model.checkpoint)
    raise NotImplementedError
    # todo: implement vpt, for this we will need to change some things about the model classes,
    # and about the registry? I think.
    # todo: use any other model properties from the config?


def build_model(cfg) -> Sam:
    if cfg.model.name == 'sam':
        sam = build_sam(cfg)
        sam.to(cfg.device)
        return sam
    else:
        raise NotImplementedError()  # we only support sam for now

class ResizeLongestSide:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def __call__(self, samples):
        """
        Expects a numpy array with shape BxCxHxW in uint8 format.
        returns a numpy array with shape BxCxAxB in uint8 format, where A is the new height, and B is the new width
        """
        original_dimensions = samples.shape[-2:]
        target_dimensions = self.get_target_dimensions(original_dimensions)
        return resize(samples, target_dimensions)

    def get_target_dimensions(self, original_dimensions):
        original_height, original_width = original_dimensions
        if original_height > original_width:
            new_height = self.target_length
            new_width = int(original_width * (new_height / original_height))
        else:
            new_width = self.target_length
            new_height = int(original_height * (new_width / original_width))
        return [new_height, new_width]

class SamWrapper:
    def __init__(self, model: Sam, logger: Logger):
        self.model = model
        self.logger = logger
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

    def __call__(self, samples, multimask_output: bool = False, return_logits: bool = True):
        original_img_size = tuple(samples.shape[-2:])
        transformed_samples = self.transform(samples)
        transformed_img_size = tuple(transformed_samples.shape[-2:])
        preprocessed_samples = self.model.preprocess(transformed_samples)
        image_embeddings = self.model.image_encoder(preprocessed_samples)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks = self.model.postprocess_masks(low_res_masks, transformed_img_size, original_img_size)
        if not return_logits:
            # return_logits is True by default, because we want to be able to use the
            # logits for calculating and backpropagating loss.
            masks = masks > self.model.mask_threshold
        self.log(masks.shape)
        self.log(torch.min(masks), torch.max(masks))
        return masks, iou_predictions, low_res_masks

    def log(self, *args):
        self.logger.log(*args)


def call_model(model, samples, logger: Logger):
    wrapper = SamWrapper(model, logger)
    return wrapper(samples)
