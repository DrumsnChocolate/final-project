_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/cbis-ddsm-multi.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
num_layers = 1
crop_size = (200, 200)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='PromptedVisionTransformer',
        img_size=crop_size,
        drop_rate=0.,
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/vit_base_p16_384.pth'),
        num_layers=num_layers,
        out_indices=num_layers-1,  # output only last layer
        prompt_cfg=dict(
            length=50,
            depth=num_layers,
            location='prepend',
            init='random',
            shared=False,
            dropout=0.1,
        )
    ),
    decode_head=dict(num_classes=150, in_index=0),
    auxiliary_head=[],  # no auxiliary head
    test_cfg=dict(mode='whole'),
)

optimizer = dict(lr=0.0005, weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=1.)}))
# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
