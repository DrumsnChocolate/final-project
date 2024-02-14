_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/cbis-ddsm/cbis-ddsm-mono.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_220k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, test_cfg=dict(size=crop_size))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=1),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole', _delete_=True))
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
