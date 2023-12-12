from typing import Callable

import torch


def call_loss(
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        outputs: torch.Tensor,
        targets: torch.Tensor
        ) -> torch.Tensor:
    return loss_function(outputs, targets)  # not sure if this requires any extra work?


def call_metrics(
        metric_functions: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        outputs: torch.Tensor,
        targets: torch.Tensor
        ) -> dict[str, torch.Tensor]:
    return {key: metric_function(outputs, targets) for key, metric_function in metric_functions.items()}
