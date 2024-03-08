_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/cbis-ddsm/cbis-ddsm-mono-256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_220k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size, test_cfg=dict(size=crop_size))
num_layers = 24
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        img_size=crop_size,
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/vit_large_p16.pth'),
        num_layers=num_layers,
        out_indices=num_layers-1,
    ),
    decode_head=dict(num_classes=1, in_index=0),
    auxiliary_head=[],
)

optim_wrapper = dict(paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
train_cfg = dict(val_interval=138)
train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = val_dataloader
