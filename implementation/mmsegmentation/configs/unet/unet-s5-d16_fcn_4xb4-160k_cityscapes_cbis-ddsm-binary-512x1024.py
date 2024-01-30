_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/cbis-ddsm-binary.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size, test_cfg=dict(size=crop_size))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth',
            prefix='backbone',
        ),
    ),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
