backbone_norm_cfg = dict(eps=1e-06, requires_grad=True, type='LN')
crop_size = (
    384,
    384,
)
custom_hooks = [
    dict(
        min_delta=0,
        monitor='train_loss',
        patience=40,
        rule='less',
        type='EarlyStoppingHook'),
]
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        0.0,
        0.0,
        0.0,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        384,
        384,
    ),
    std=[
        255,
        255,
        255,
    ],
    test_cfg=dict(size=(
        384,
        384,
    )),
    type='SegDataPreProcessor')
data_root = 'data/cbis/cbis-linked'
dataset_type = 'CBISMonoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=22000,
        max_keep_ckpts=2,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[],
    backbone=dict(
        drop_rate=0.0,
        embed_dims=1024,
        final_norm=True,
        img_size=(
            384,
            384,
        ),
        in_channels=3,
        init_cfg=None,
        interpolate_mode='bilinear',
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        num_heads=16,
        num_layers=24,
        out_indices=23,
        patch_bias=True,
        patch_size=16,
        type='VisionTransformer',
        with_cls_token=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            384,
            384,
        ),
        std=[
            255,
            255,
            255,
        ],
        test_cfg=dict(size=(
            384,
            384,
        )),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0,
        in_channels=1024,
        in_index=0,
        kernel_size=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=1,
        num_convs=4,
        type='SETRUPHead',
        up_scale=2),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_layers = 24
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.0001, type='Adam'),
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.0001, type='Adam')
param_scheduler = [
    dict(
        cooldown=0,
        factor=0.1,
        min_value=0,
        monitor='mCE',
        patience=25,
        rule='less',
        threshold=0.0001,
        type='ReduceOnPlateauLR'),
]
record_gpu_snapshot = False
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations_binary/test'),
        data_root='data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CLAHE'),
            dict(scale=(
                384,
                384,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CBISMonoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(iou_metrics=[
        'mIoU',
        'mAcc',
        'mFscore',
    ], type='IoUMetric'),
    dict(type='CrossEntropyMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CLAHE'),
    dict(scale=(
        384,
        384,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=220000, type='IterBasedTrainLoop', val_interval=138)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations_binary/train'),
        data_root='data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='CLAHE'),
            dict(scale=(
                384,
                384,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        type='CBISMonoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='CLAHE'),
    dict(scale=(
        384,
        384,
    ), type='Resize'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = None
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations_binary/val'),
        data_root='data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CLAHE'),
            dict(scale=(
                384,
                384,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CBISMonoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(iou_metrics=[
        'mIoU',
        'mAcc',
        'mFscore',
    ], type='IoUMetric'),
    dict(type='CrossEntropyMetric'),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CLAHE'),
    dict(scale=(
        384,
        384,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/setr_vit-l_pup-noaux_8xb2-220k_cbis-ddsm-mono-256x256'
