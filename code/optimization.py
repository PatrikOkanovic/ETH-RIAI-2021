from typing import Callable, Optional

import torch

EPS = 1e-8


def solve_newton(
    f: Callable[[torch.Tensor], torch.Tensor],
    fprime: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    max_iter: int = 10,
) -> torch.Tensor:
    p = torch.tensor(x0)
    for itr in range(max_iter):
        f_val = f(p)
        f_der = fprime(p)
        dp = f_val / f_der
        p -= dp
    return p


def solve_bisect(
    f: Callable[[torch.Tensor], torch.Tensor],
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    max_iter: int = 10,
    active_check_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # assert lower_bound.shape == upper_bound.shape
    # f_lb = f(lower_bound)
    # if active_check_mask is not None:
    #     assert torch.all(f_lb[active_check_mask] >= 0), "Lower bound is not sound"

    for itr in range(max_iter):
        p = (lower_bound + upper_bound) / 2.0
        f_curr = f(p)

        mask_lb_assign = (
            torch.sgn(f_curr) > 0
        )  # f is positive if we are estimating lower than the actual solution
        mask_ub_assign = ~mask_lb_assign

        lower_bound = p * mask_lb_assign + lower_bound * (~mask_lb_assign)
        upper_bound = p * mask_ub_assign + upper_bound * (~mask_ub_assign)

    # add epsilon just in case?
    # if active_check_mask is not None:
    #     assert torch.all(torch.sgn(f(lower_bound)[active_check_mask]) >= 0)
    return lower_bound
