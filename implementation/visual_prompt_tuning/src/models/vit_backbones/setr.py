import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import Transformer
from ...configs import setr_configs as configs

from mmcv.cnn import build_norm_layer


CONFIGS = {
    "setr_pup": configs.get_pup_config(),
#     todo: fill this in, like in vit.py
}


class VisionTransformerUpHead(nn.Module):
    """Upsampling Head for vision transformer, simplified from https://github.com/fudan-zvg/SETR"""
    def __init__(self, cfg, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(VisionTransformerUpHead, self).__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.channels = cfg.channels
        self.num_classes = cfg.num_classes
        self.in_index = cfg.in_index
        self.img_size = cfg.img_size
        self.norm_cfg = cfg.norm_cfg
        self.norm = norm_layer(cfg.embed_dim)
        out_channel = self.num_classes
        self.conv_0 = nn.Conv2d(cfg.embed_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(256, out_channel, kernel_size=1, stride=1)
        _, self.syncbn_fc0 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc1 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc2 = build_norm_layer(self.norm_cfg, 256)
        _, self.syncbn_fc3 = build_norm_layer(self.norm_cfg, 256)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _transform_inputs(self, x):
        # we don't consider resize_concat and all that stuff (see https://github.com/fudan-zvg/SETR)
        return x[self.in_index]  # I don't know why this is done

    def forward(self, x):
        x = self._transform_inputs(x)
        if x.dim() == 3:
            if x.shape[1] % 48 != 0:
                x = x[:, 1:]
            x = self.norm(x)
        # we assume that upsampling is bilinear
        # if self.upsampling_method == 'bilinear':
        if x.dim() == 3:
            n, hw, c = x.shape
            h = w = int(math.sqrt(hw))
            x = x.transpose(1, 2).reshape(n, c, h, w)

        # if self.num_conv == 2:
        #     if self.num_upsampe_layer == 2:
        #         x = self.conv_0(x)
        #         x = self.syncbn_fc_0(x)
        #         x = F.relu(x, inplace=True)
        #         x = F.interpolate(
        #             x, size=x.shape[-1]*4, mode='bilinear', align_corners=self.align_corners)
        #         x = self.conv_1(x)
        #         x = F.interpolate(
        #             x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        #     elif self.num_upsampe_layer == 1:
        #         x = self.conv_0(x)
        #         x = self.syncbn_fc_0(x)
        #         x = F.relu(x, inplace=True)
        #         x = self.conv_1(x)
        #         x = F.interpolate(
        #             x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        # we assume that we want 4 upsampling operations, because we restrict upsampling to 2x per operation,
        #  so we need to upsample 4 times to get from
        # elif self.num_conv == 4:
        if self.num_upsampe_layer == 4:
            x = self.conv_0(x)
            x = self.syncbn_fc_0(x)
            x = F.relu(x, inplace=True)
            x = F.interpolate(
                x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
            x = self.conv_1(x)
            x = self.syncbn_fc_1(x)
            x = F.relu(x, inplace=True)
            x = F.interpolate(
                x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
            x = self.conv_2(x)
            x = self.syncbn_fc_2(x)
            x = F.relu(x, inplace=True)
            x = F.interpolate(
                x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
            x = self.conv_3(x)
            x = self.syncbn_fc_3(x)
            x = F.relu(x, inplace=True)
            x = self.conv_4(x)
            x = F.interpolate(
                x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)

        return x


class SegmentationTransformer(nn.Module):
    def __init__(self, model_type, img_size=512, num_classes=150, vis=False):
        super(SegmentationTransformer, self).__init__()
        config = CONFIGS[model_type]
        config.head.img_size = img_size  # todo: vary this default
        config.head.num_classes = num_classes  # todo: vary this default, but how?? when using pre-trained from ade20k?
        self.transformer = Transformer(config, img_size, vis)
        self.head = VisionTransformerUpHead(config.head)

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


