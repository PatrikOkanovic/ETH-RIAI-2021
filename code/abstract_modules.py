from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from optimization import solve_bisect
from torch import Tensor

from abstract_shape import DeepPolyShape, RelationalConstraint
from networks import SPU, FullyConnected, spu, spu_grad, spu_grad2
from utils import concretize, get_neg_pos_component

EPS = 1e-9


@dataclass
class AbstractModule:
    input_bounds: Optional[Tuple[Tensor, Tensor]] = None
    initialized: bool = False

    def update_input_bounds(
        self, lb: Tensor, ub: Tensor, keep_best_bounds: bool = True
    ) -> None:
        if self.input_bounds is None or not keep_best_bounds:
            self.input_bounds = (lb, ub)
        else:
            cur_lb, cur_ub = self.input_bounds
            lb = torch.max(lb, cur_lb)
            ub = torch.min(ub, cur_ub)
            self.input_bounds = (lb, ub)

    def transform(
        self,
        shapes: List[DeepPolyShape],
        input_lb: Tensor,
        input_ub: Tensor,
    ) -> None:
        raise NotImplementedError


@dataclass
class AbstractLinearBase:
    weight: Tensor
    bias: Tensor


@dataclass
class AbstractLinear(AbstractModule, AbstractLinearBase):
    def transform(
        self,
        shapes: List[DeepPolyShape],
        input_lb: Tensor,
        input_ub: Tensor,
    ) -> None:
        rel_lb, rel_ub = self.get_cur_rel_lb_ub()
        lb, ub = self.backsubstitute(shapes, input_lb, input_ub)
        shapes.append(DeepPolyShape(lb, ub, rel_lb, rel_ub))

    def backsubstitute(
        self,
        shapes: List[DeepPolyShape],
        input_lb: Tensor,
        input_ub: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        rel_lb, rel_ub = self.get_cur_rel_lb_ub()
        for shape in reversed(shapes):
            self.update(rel_lb, shape, lower_bound=True)
            self.update(rel_ub, shape, lower_bound=False)
        lb = concretize(rel_lb, input_lb, input_ub, lower_bound=True)
        ub = concretize(rel_ub, input_lb, input_ub, lower_bound=False)
        return lb, ub

    def get_cur_rel_lb_ub(self) -> Tuple[RelationalConstraint, RelationalConstraint]:
        rel_lb = RelationalConstraint(self.weight, self.bias - EPS)
        rel_ub = RelationalConstraint(self.weight, self.bias + EPS)
        return rel_lb, rel_ub

    def update(
        self,
        rel_cstr: RelationalConstraint,
        shape: DeepPolyShape,
        lower_bound: bool = True,
    ) -> None:
        neg_weight, pos_weight = get_neg_pos_component(rel_cstr.weight)
        if lower_bound:
            rel_cstr.weight = (
                neg_weight @ shape.rel_ub.weight + pos_weight @ shape.rel_lb.weight
            )
            rel_cstr.bias = (
                rel_cstr.bias
                + neg_weight @ shape.rel_ub.bias
                + pos_weight @ shape.rel_lb.bias
            )
        else:
            rel_cstr.weight = (
                neg_weight @ shape.rel_lb.weight + pos_weight @ shape.rel_ub.weight
            )
            rel_cstr.bias = (
                rel_cstr.bias
                + neg_weight @ shape.rel_lb.bias
                + pos_weight @ shape.rel_ub.bias
            )


@dataclass
class AbstractSPU(AbstractModule):
    alpha_lb: Tensor = None
    alpha_ub: Tensor = None
    num_logits: int = -1

    def __post_init__(self) -> None:
        # assert (
        #     self.num_logits > 0
        # ), "The number of logits has to be set to initialize trainable parameters"
        self.alpha_lb = nn.Parameter(torch.zeros(self.num_logits, requires_grad=True))
        self.alpha_ub = nn.Parameter(torch.zeros(self.num_logits, requires_grad=True))

    def transform(
        self,
        shapes: List[DeepPolyShape],
        input_lb: Tensor,
        input_ub: Tensor,
    ) -> None:
        assert self.input_bounds is not None
        lb, ub = self.input_bounds
        # assert all(lb <= ub)
        spu_lb, spu_ub = spu(lb), spu(ub)
        lmbda = (spu_ub - spu_lb) / (ub - lb)
        pos_mask = lb >= 0
        neg_mask = ub <= 0
        cros_mask = (lb < 0) & (ub > 0)

        q = torch.where(
            (lmbda < 0) & cros_mask,
            torch.log(
                torch.clamp(
                    2 / (1 + torch.sqrt(torch.clamp(1 + 4 * lmbda, min=1e-15))) - 1,
                    min=1e-15,
                )
            ),
            lb,
        )

        cros_mask_ub_a = cros_mask & ((spu_lb <= spu_ub) | (lb >= q))
        cros_mask_ub_b = cros_mask & (~cros_mask_ub_a)
        # assert consistent_masks([pos_mask, neg_mask, cros_mask_ub_a, cros_mask_ub_b])

        t = torch.where(cros_mask_ub_b, self.calculate_t(q, ub, cros_mask_ub_b), q)

        if not self.initialized:
            self.initialized = True
            with torch.no_grad():
                self.alpha_lb.copy_(lb)
                self.alpha_ub.copy_((lb + ub) / 2)
                self.alpha_ub[cros_mask_ub_b] = q[cros_mask_ub_b]

        with torch.no_grad():
            self.alpha_lb[pos_mask] = torch.max(self.alpha_lb[pos_mask], lb[pos_mask])
            self.alpha_lb[cros_mask] = torch.max(
                self.alpha_lb[cros_mask], ((spu_lb + 0.5) / lb)[cros_mask]
            )
            self.alpha_ub[neg_mask] = torch.min(self.alpha_ub[neg_mask], ub[neg_mask])
            self.alpha_ub[neg_mask] = torch.max(self.alpha_ub[neg_mask], lb[neg_mask])
            self.alpha_ub[cros_mask_ub_b] = torch.min(
                self.alpha_ub[cros_mask_ub_b], t[cros_mask_ub_b]
            )
            self.alpha_ub[cros_mask_ub_b] = torch.max(
                self.alpha_ub[cros_mask_ub_b], lb[cros_mask_ub_b]
            )

        mask_lb_a = self.alpha_lb >= 0
        mask_lb_b = self.alpha_lb < 0

        rel_lb = RelationalConstraint(
            weight=torch.diag(
                spu_grad(self.alpha_lb) * (mask_lb_a & ~neg_mask)
                + self.alpha_lb * (mask_lb_b & ~neg_mask)
                + lmbda * neg_mask
            ),
            bias=(
                (spu(self.alpha_lb) - self.alpha_lb * spu_grad(self.alpha_lb))
                * (mask_lb_a & ~neg_mask)
                - 0.5 * (mask_lb_b & ~neg_mask)
                + (spu_lb - lmbda * lb) * neg_mask
                - EPS
            ),
        )
        rel_ub = RelationalConstraint(
            weight=torch.diag(
                lmbda * (pos_mask + cros_mask_ub_a)
                + spu_grad(self.alpha_ub) * (neg_mask + cros_mask_ub_b)
            ),
            bias=(
                (spu_lb - lmbda * lb) * (pos_mask + cros_mask_ub_a)
                + (spu(self.alpha_ub) - self.alpha_ub * spu_grad(self.alpha_ub))
                * (neg_mask + cros_mask_ub_b)
                + EPS
            ),
        )
        output_lb = concretize(rel_lb, lb, ub, lower_bound=True)
        output_ub = concretize(rel_ub, lb, ub, lower_bound=False)

        # assert all(output_lb <= output_ub)
        shapes.append(DeepPolyShape(output_lb, output_ub, rel_lb, rel_ub))

    def calculate_t(self, q: Tensor, b: Tensor, active_check_mask: Tensor) -> Tensor:
        def f(x: Tensor) -> Tensor:
            return spu_grad(x) * (b - x) + spu(x) - spu(b)

        def df(x: Tensor) -> Tensor:
            return spu_grad2(x) * (b - x)

        # find initial upper bound for the solution of f(x) == 0
        search_lb = q
        search_ub = q - f(q) / df(q)
        search_ub = torch.minimum(search_ub, torch.zeros_like(search_ub))
        # assert all(search_lb[active_check_mask] <= search_ub[active_check_mask])
        return solve_bisect(
            f, search_lb, search_ub, active_check_mask=active_check_mask
        )


class AbstractNetwork:
    def __init__(
        self,
        concrete_network: FullyConnected,
        lr: float,
        num_iter: int,
        verbosity: int = 0,
    ):
        self.layers: List[AbstractModule] = []
        # assert isinstance(concrete_network.layers[0], Normalization)
        # assert isinstance(concrete_network.layers[1], nn.Flatten)
        last_layer_logits = -1
        for concrete_layer in concrete_network.layers[2:]:
            if isinstance(concrete_layer, nn.Linear):
                self.layers.append(
                    AbstractLinear(concrete_layer.weight.data, concrete_layer.bias.data)
                )
                last_layer_logits = concrete_layer.bias.shape[-1]
            elif isinstance(concrete_layer, SPU):
                self.layers.append(AbstractSPU(num_logits=last_layer_logits))
            else:
                raise NotImplementedError
        self.last_layer_added = False
        self.lr = lr
        self.num_iter = num_iter
        self.verbosity = verbosity

    def reset(self) -> None:
        for layer in self.layers:
            layer.input_bounds = None
            layer.initialized = False

    def set_labels(self, true_label: int, target_label: int) -> None:
        weight = torch.zeros(1, 10)
        weight[0, true_label], weight[0, target_label] = 1, -1
        bias = torch.zeros(1)
        if self.last_layer_added:
            self.layers[-1] = AbstractLinear(weight, bias)
        else:
            self.layers.append(AbstractLinear(weight, bias))
            self.last_layer_added = True

    def parameters(self) -> List:
        params = []
        for layer in self.layers:
            if isinstance(layer, AbstractSPU):
                params.extend([layer.alpha_lb, layer.alpha_ub])
        return params

    def analyze_label(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        true_label: int,
        target_label: int,
        keep_best_bounds: bool,
    ) -> bool:
        self.reset()
        self.set_labels(true_label, target_label)
        num_iter = self.num_iter
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        while num_iter > 0:
            num_iter -= 1
            abstract_shapes: List[DeepPolyShape] = []
            for layer in self.layers:
                if len(abstract_shapes) > 0:
                    lb, ub = abstract_shapes[-1].lb, abstract_shapes[-1].ub
                    layer.update_input_bounds(lb, ub, keep_best_bounds)
                layer.transform(abstract_shapes, input_lb, input_ub)
            if self.verbosity == 2:
                print(
                    f"Verifying against target label {target_label}, output range [{abstract_shapes[-1].lb}, {abstract_shapes[-1].ub}]"
                )
            if abstract_shapes[-1].lb > 0:
                if self.verbosity == 1:
                    print(
                        f"Verifying against target label {target_label}, needed {self.num_iter-num_iter} iterations to prove property"
                    )
                return True
            optimizer.zero_grad()
            loss = -abstract_shapes[-1].lb
            loss.backward()
            optimizer.step()

        return False

    def analyze(
        self,
        input_lb: Tensor,
        input_ub: Tensor,
        true_label: int,
        keep_best_bounds: bool = False,
    ) -> bool:
        """Returns True if the network is robust for the given input"""
        for target_label in range(10):
            if target_label == true_label:
                continue
            if not self.analyze_label(
                input_lb, input_ub, true_label, target_label, keep_best_bounds
            ):
                return False
        return True
