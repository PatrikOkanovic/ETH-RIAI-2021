import argparse

import torch
from torch import Tensor

from abstract_modules import AbstractNetwork
from networks import FullyConnected
from utils import get_input_bounds

DEVICE = "cpu"
INPUT_SIZE = 28


def analyze(
    net: FullyConnected,
    inputs: Tensor,
    eps: float,
    true_label: int,
    args: argparse.Namespace,
) -> bool:
    input_lb, input_ub = get_input_bounds(inputs, eps, DEVICE)
    abstract_network = AbstractNetwork(
        net, verbosity=args.verbosity, num_iter=args.num_iter, lr=args.lr
    )
    return abstract_network.analyze(input_lb, input_ub, true_label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation"
    )
    parser.add_argument(
        "--net",
        type=str,
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument(
        "--num_iter",
        type=int,
        default=10000,
        help="Maximum number of iterations for parameter optimization",
    )
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)

    with open(args.spec, "r") as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split("/")[-1].split("_")[-1])

    if args.net.endswith("fc1"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith("fc2"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith("fc3"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith("fc4"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith("fc5"):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(
        torch.load("../mnist_nets/%s.pt" % args.net, map_location=torch.device(DEVICE))
    )

    inputs = (
        torch.DoubleTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    )
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label, args):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
