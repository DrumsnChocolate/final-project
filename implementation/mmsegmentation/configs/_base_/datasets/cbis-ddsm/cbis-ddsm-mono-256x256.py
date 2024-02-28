# dataset settings
dataset_type = 'CBISMonoDataset'
data_root = 'data/cbis/cbis-linked'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='CLAHE'),
    dict(type='Resize', scale=(256, 256)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', degree=(90, 90), prob=0.5, auto_bound=True),
    dict(type='RandomRotate', degree=(180, 180), prob=0.5, auto_bound=True),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CLAHE'),
    dict(type='Resize', scale=(256, 256)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]
val_pipeline = test_pipeline
tta_pipeline = None  # we don't use test time augmentation
train_dataloader = dict(
    batch_size=16,  # 1 gpu, so total batch size is 16
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations_binary/train'),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations_binary/val'),
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/test', seg_map_path='annotations_binary/test'),
        pipeline=val_pipeline,
    ),
)
val_evaluator = [dict(type='IoUMetric', iou_metrics=['mIoU', 'mAcc', 'mFscore']), dict(type='CrossEntropyMetric')]
test_evaluator = val_evaluator
