_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
num_layers = 12
crop_size = (224, 224)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        img_size=crop_size,
        embed_dims=768,
        num_heads=12,
        drop_rate=0.,
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/vit_base_p16_224.pth'),
        num_layers=num_layers,
        out_indices=num_layers-1,  # output only last layer
    ),
    decode_head=dict(num_classes=150, in_index=0, in_channels=768),
    auxiliary_head=[],  # no auxiliary head
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(100, 100)),
)

optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader