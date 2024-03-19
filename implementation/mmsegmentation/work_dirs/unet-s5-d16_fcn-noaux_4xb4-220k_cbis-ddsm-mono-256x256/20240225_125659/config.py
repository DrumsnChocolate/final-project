crop_size = (
    256,
    256,
)
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
        256,
        256,
    ),
    std=[
        255,
        255,
        255,
    ],
    test_cfg=dict(size=(
        256,
        256,
    )),
    type='SegDataPreProcessor')
data_root = '/local/270180/data/cbis/cbis-linked'
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
    auxiliary_head=None,
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        base_channels=64,
        conv_cfg=None,
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        type='UNet',
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
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
            256,
            256,
        ),
        std=[
            255,
            255,
            255,
        ],
        test_cfg=dict(size=(
            256,
            256,
        )),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=64,
        in_index=4,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=1,
        num_convs=1,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.0001, type='Adam'),
    type='OptimWrapper')
optimizer = dict(lr=0.0001, type='Adam')
param_scheduler = [
    dict(monitor='mIoU', rule='greater', type='ReduceOnPlateauLR'),
]
record_gpu_snapshot = False
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations_binary/test'),
        data_root='/local/270180/data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CLAHE'),
            dict(scale=(
                256,
                256,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CBISMonoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mAcc',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CLAHE'),
    dict(scale=(
        256,
        256,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=220000, type='IterBasedTrainLoop', val_interval=275)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='images/train', seg_map_path='annotations_binary/train'),
        data_root='/local/270180/data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='CLAHE'),
            dict(scale=(
                256,
                256,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                auto_bound=True,
                degree=(
                    90,
                    90,
                ),
                prob=0.5,
                type='RandomRotate'),
            dict(
                auto_bound=True,
                degree=(
                    180,
                    180,
                ),
                prob=0.5,
                type='RandomRotate'),
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
        256,
        256,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(auto_bound=True, degree=(
        90,
        90,
    ), prob=0.5, type='RandomRotate'),
    dict(auto_bound=True, degree=(
        180,
        180,
    ), prob=0.5, type='RandomRotate'),
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
        data_root='/local/270180/data/cbis/cbis-linked',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CLAHE'),
            dict(scale=(
                256,
                256,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CBISMonoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mAcc',
        'mFscore',
    ], type='IoUMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CLAHE'),
    dict(scale=(
        256,
        256,
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
work_dir = './work_dirs/unet-s5-d16_fcn-noaux_4xb4-220k_cbis-ddsm-mono-256x256'
