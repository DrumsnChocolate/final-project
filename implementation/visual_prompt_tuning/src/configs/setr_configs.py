import ml_collections


def get_pup_config():
    # todo: use this dict to populate the model
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})  # todo: determine
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24  # depth
    config.transformer.attention_dropout_rate = 0.
    config.transformer.dropout_rate = 0.
    config.representation_size = None
    config.head = get_pup_head_config()
    return config

def get_pup_head_config():
    config = ml_collections.ConfigDict(convert_dict=False)
    config.in_channels = 1024
    config.channels = 512
    config.embed_dim = 1024
    config.in_index = 23
    config.img_size = None  # this is determined by input parameters, not in the config already
    config.num_classes = None  # this is determined by input parameters, not in the config already
    config.num_upsampling_layers = None  # we don't use this setting because we fix it for simplicity.
    config.num_conv_layers = None  # we don't use this setting because we fix it for simplicity.
    config.norm_cfg = dict(type='SyncBN', requires_grad=True)
    return config

