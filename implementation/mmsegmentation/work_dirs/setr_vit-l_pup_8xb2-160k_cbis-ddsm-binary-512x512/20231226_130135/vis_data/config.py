backbone_norm_cfg = dict(eps=1e-06, requires_grad=True, type='LN')
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/cbis/cbis-linked'
dataset_type = 'CBISBinaryDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=16000,
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
    auxiliary_head=[
        dict(
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            channels=256,
            dropout_ratio=0,
            in_channels=1024,
            in_index=0,
            kernel_size=3,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=2,
            num_convs=2,
            type='SETRUPHead'),
        dict(
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            channels=256,
            dropout_ratio=0,
            in_channels=1024,
            in_index=1,
            kernel_size=3,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=2,
            num_convs=2,
            type='SETRUPHead'),
        dict(
            act_cfg=dict(type='ReLU'),
            align_corners=False,
            channels=256,
            dropout_ratio=0,
            in_channels=1024,
            in_index=2,
            kernel_size=3,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=2,
            num_convs=2,
            type='SETRUPHead'),
    ],
    backbone=dict(
        drop_rate=0.0,
        embed_dims=1024,
        img_size=(
            512,
            512,
        ),
        in_channels=3,
        init_cfg=dict(
            checkpoint='pretrain/vit_large_p16.pth', type='Pretrained'),
        interpolate_mode='bilinear',
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        num_heads=16,
        num_layers=24,
        out_indices=(
            9,
            14,
            19,
            23,
        ),
        patch_size=16,
        type='VisionTransformer',
        with_cls_token=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0,
        in_channels=1024,
        in_index=3,
        kernel_size=3,
        loss_decode=dict(
            class_weight=[
                0.04,
                1.96,
            ],
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=4,
        type='SETRUPHead',
        up_scale=2),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 2
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.001),
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.001)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=160000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
record_gpu_snapshot = True
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations_binary/test'),
        data_root='data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CBISBinaryDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=16000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations_binary/train'),
        data_root='data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='CBISBinaryDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = None
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations_binary/val'),
        data_root='data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CBISBinaryDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
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
work_dir = './work_dirs/setr_vit-l_pup_8xb2-160k_cbis-ddsm-binary-512x512'
