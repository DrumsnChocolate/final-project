import torch
import torch.nn as nn

from .vit import Transformer
from ...configs import setr_configs as configs


CONFIGS = {
    "setr_pup": configs.get_pup_config(),
#     todo: fill this in, like in vit.py
}


class VisionTransformerUpHead(nn.Module):
    """Upsampling Head for vision transformer"""
    def __init__(self, in_channels, ):

class SegmentationTransformer(nn.Module):
    def __init__(self, model_type, img_size=224, num_classes=21843, vis=False):
        super(SegmentationTransformer, self).__init__()
        # todo: implement SETR specific init?
        # copy parts from vit?
        # todo: figure out what vis is for
        config = CONFIGS[model_type]
        self.transformer = Transformer(config, img_size, vis)
        self.head = None  # todo: assign PUP head
        raise NotImplementedError()

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        mask = self.head(x)  # todo: figure out dimensions of input and output
        if not vis:
            return mask
        return mask, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            pass # todo: implement
        raise NotImplementedError()


