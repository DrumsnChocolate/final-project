import torch
from functools import partial
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, VPTImageEncoderViT, VPTSam


# SAM's default pixel mean and std
default_pixel_mean = [123.675, 116.28, 103.53]
default_pixel_std = [58.395, 57.12, 57.375]



def build_sam(cfg):
    # todo: make the pixel mean and std configurable
    pixel_mean = default_pixel_mean
    pixel_std = default_pixel_std
    if cfg.model.pixel_mean is not None:
        pixel_mean = cfg.model.pixel_mean
    if cfg.model.pixel_std is not None:
        pixel_std = cfg.model.pixel_std

    if cfg.model.backbone == 'vit_h':
        encoder_embed_dim = 1280
        encoder_depth = 32
        encoder_num_heads = 16
        encoder_global_attn_indexes = [7, 15, 23, 31]
    elif cfg.model.backbone == 'vit_l':
        encoder_embed_dim = 1024
        encoder_depth = 24
        encoder_num_heads = 16
        encoder_global_attn_indexes = [5, 11, 17, 23]
    elif cfg.model.backbone == 'vit_b':
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        encoder_global_attn_indexes = [2, 5, 8, 11]
    else:
        raise NotImplementedError()

    if cfg.model.get('finetuning') is None or cfg.model.finetuning.name == 'full':
        return _build_sam(
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            encoder_global_attn_indexes=encoder_global_attn_indexes,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            checkpoint=cfg.model.checkpoint,
        )
    elif cfg.model.finetuning.name == 'vpt':
        return _build_vpt_sam(
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            encoder_global_attn_indexes=encoder_global_attn_indexes,
            vpt_length=cfg.model.finetuning.length,
            vpt_dropout=cfg.model.finetuning.dropout,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            checkpoint=cfg.model.checkpoint,
        )
    raise NotImplementedError()


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    pixel_mean,
    pixel_std,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def _build_vpt_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    vpt_length,
    vpt_dropout,
    pixel_mean,
    pixel_std,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = VPTSam(
        image_encoder=VPTImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            vpt_length=vpt_length,
            vpt_dropout=vpt_dropout,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=False)
    return sam