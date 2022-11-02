from dataclasses import dataclass

from torch import Tensor


@dataclass
class RelationalConstraint:
    weight: Tensor
    bias: Tensor


@dataclass
class DeepPolyShape:
    lb: Tensor
    ub: Tensor
    rel_lb: RelationalConstraint
    rel_ub: RelationalConstraint
