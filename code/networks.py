from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class Normalization(nn.Module):
    def __init__(self, device: str):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.sigma


def spu(x: Tensor) -> Tensor:
    return torch.where(x > 0, x ** 2 - 0.5, torch.sigmoid(-x) - 1)


def spu_grad(x: Tensor) -> Tensor:
    return torch.where(x > 0, 2 * x, (torch.sigmoid(-x)) * (torch.sigmoid(-x) - 1))


def spu_grad2(x: Tensor) -> Tensor:
    return torch.where(
        x > 0,
        2.0,
        (torch.sigmoid(-x)) * (torch.sigmoid(-x) - 1) * (2 * torch.sigmoid(-x) - 1),
    )


class SPU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return spu(x)


class FullyConnected(nn.Module):
    def __init__(self, device: str, input_size: int, fc_layers: List[int]):
        super(FullyConnected, self).__init__()

        layers = [Normalization(device), nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [SPU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
