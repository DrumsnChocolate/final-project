import ml_collections


def get_pup_config():
    # todo: use this dict to populate the model
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (None, None)})  # todo: determine
    config.hidden_size = None
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = None
    config.transformer.num_heads = None
    config.transformer.num_layers = None
    config.transformer.attention_dropout_rate = None
    config.transformer.dropout_rate = None
    # config.classifier = 'token'  # todo: determine params for upsampling head
    config.representation_size = None
    config.head = ml_collections.ConfigDict()
    config.head.upsampling = 'bilinear'
    config.head.num_upsampling_layers = None
    config.head.num_conv_layers = None
