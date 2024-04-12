import torch.nn as nn


def get_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def get_param_count_trainable_recursively(module: nn.Module) -> int:
    if getattr(module, 'param_count_trainable', None) is not None:
        return module.param_count_trainable
    direct_child_params = module.parameters(recurse=False)
    named_children = module.named_children()
    return sum(p.numel() for p in direct_child_params if p.requires_grad) + sum(get_param_count_trainable_recursively(child) for _, child in named_children)
