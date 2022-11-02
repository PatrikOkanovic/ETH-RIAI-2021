from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor


def fgsm_(
    model: nn.Module,
    x: Tensor,
    target: int,
    eps: float,
    targeted: bool = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> Tensor:
    """Internal process for all FGSM and PGD attacks."""
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits = model(input_)
    target = torch.LongTensor([target])
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    # perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


# x: input image
# target: target class
# eps: size of l-infinity ball
def fgsm_targeted(
    model: nn.Module, x: Tensor, target: int, eps: float, **kwargs: Optional[Any]
) -> Tensor:
    return fgsm_(model, x, target, eps, targeted=True, **kwargs)


# x: input image
# label: current label of x
# eps: size of l-infinity ball
def fgsm_untargeted(
    model: nn.Module, x: Tensor, label: int, eps: float, **kwargs: Optional[Any]
) -> Tensor:
    return fgsm_(model, x, label, eps, targeted=False, **kwargs)


# x: input image
# label: current label of x
# k: number of FGSM iterations
# eps: size of l-infinity ball
# eps_size: step size of FGSM iterations
def pgd_(
    model: nn.Module,
    x: Tensor,
    target: int,
    k: int,
    eps: float,
    eps_step: float,
    targeted: bool = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> Tensor:
    x_min = x - eps
    x_max = x + eps

    # Randomize the starting point x.
    x = x + eps * (2 * torch.rand_like(x) - 1)
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)

    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None)
        # as we want to apply the attack as defined
        x = fgsm_(model, x, target, eps_step, targeted)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)
    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)
    return x


def pgd(
    model: nn.Module,
    x: Tensor,
    label: int,
    k: int,
    eps: float,
    eps_step: float,
    **kwargs: Optional[Any]
) -> Tensor:
    return pgd_(model, x, label, k, eps, eps_step, targeted=False, **kwargs)
