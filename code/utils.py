from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from abstract_shape import RelationalConstraint
from networks import Normalization


def get_neg_pos_component(a: Tensor) -> Tuple[Tensor, Tensor]:
    return (
        torch.where(a < 0, a, torch.zeros_like(a)),
        torch.where(a > 0, a, torch.zeros_like(a)),
    )


def get_input_bounds(a: Tensor, eps: float, device: str) -> Tuple[Tensor, Tensor]:
    a_lb, a_ub = (a - eps).clamp(min=0), (a + eps).clamp(max=1)
    a_lb, a_ub = Normalization(device)(a_lb), Normalization(device)(a_ub)
    a_lb, a_ub = nn.Flatten()(a_lb), nn.Flatten()(a_ub)
    return a_lb.reshape(-1), a_ub.reshape(-1)


def concretize(
    rel_cstr: RelationalConstraint,
    input_lb: Tensor,
    input_ub: Tensor,
    lower_bound: bool = True,
) -> Tensor:
    neg_weight, pos_weight = get_neg_pos_component(rel_cstr.weight)
    if lower_bound:
        bound = neg_weight @ input_ub + pos_weight @ input_lb + rel_cstr.bias
    else:
        bound = neg_weight @ input_lb + pos_weight @ input_ub + rel_cstr.bias
    return bound


def consistent_masks(
    tensors: List[Tensor],
) -> bool:
    return torch.stack(tensors).sum() == len(tensors[0])
