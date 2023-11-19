_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_layers = 24
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='PromptedVisionTransformer',
        img_size=crop_size,
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/vit_large_p16.pth'),
        prompt_cfg=dict(
            length=50,  # todo: hyperparameter sweep this? for [1, 5, 10, 50, 100, 200]
            depth=1,
            location='prepend',
            init='random',
            shared=False,
            dropout=0.1,  # todo: sweep for [0.0, 0.1]
        ),
        num_layers=num_layers,
        out_indices=num_layers-1,
    ),
    decode_head=dict(num_classes=150, in_index=0),
    auxiliary_head=[],
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optimizer = dict(lr=0.0005, weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=1.)}))
# num_gpus: 1 -> batch_size: 16
train_dataloader = dict(batch_size=16)
val_dataloader = dict(batch_size=4)
test_dataloader = val_dataloader