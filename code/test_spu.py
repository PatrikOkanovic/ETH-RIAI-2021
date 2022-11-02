from typing import Any, List

import torch

from abstract_modules import AbstractSPU
from abstract_shape import DeepPolyShape


class TestSPU:
    def make_spu_transform(self, lb: Any, ub: Any) -> DeepPolyShape:
        if isinstance(lb, (int, float)):
            lb = torch.tensor([lb], dtype=float)
            ub = torch.tensor([ub], dtype=float)
        if isinstance(ub, list):
            lb = torch.tensor(lb, dtype=float)
            ub = torch.tensor(ub, dtype=float)
        layer = AbstractSPU(num_logits=1)
        layer.update_input_bounds(lb, ub)
        shapes: List[DeepPolyShape] = []
        layer.transform(shapes, lb, ub)
        return shapes[-1]


if __name__ == "__main__":
    test_spu = TestSPU()
    print(test_spu.make_spu_transform(-4, 0.5))
