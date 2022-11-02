import argparse

import torch
import torchvision
from adversarial_attack import pgd
from torch import Tensor
from tqdm import tqdm
from verifier import analyze

from networks import FullyConnected

DEVICE = "cpu"
INPUT_SIZE = 28


def get_minst_dataloader(batch_size_train: int = 1) -> torch.utils.data.DataLoader:
    batch_size_train = 1
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    return train_loader


def adversary_example(
    net: FullyConnected, true_label: int, inputs: Tensor, eps: float
) -> bool:
    """"Returns true if an adversary example has been found"""
    adv = pgd(
        net,
        inputs,
        label=true_label,
        k=10,
        eps=eps,
        eps_step=0.05,
        clip_min=0,
        clip_max=1.0,
    )
    outs = net(adv)
    pred_label = outs.max(dim=1)[1].item()
    return pred_label != true_label


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
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument(
        "--num_iter",
        type=int,
        default=10000,
        help="Maximum number of iterations for parameter optimization",
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.05)
    args = parser.parse_args()

    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)

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

    dataloader = get_minst_dataloader(1)
    statistics = {
        "verifiedAdversary": 0,
        "verifiedNOAdversary": 0,
        "NOverifiedAdversary": 0,
        "NOverifiedNOAdversary": 0,
    }
    for idx, inputs_label_list in tqdm(enumerate(dataloader)):
        inputs, true_label = inputs_label_list[0], inputs_label_list[1]
        true_label = true_label.item()
        outs = net(inputs)
        pred_label = outs.max(dim=1)[1].item()
        if pred_label != true_label:
            continue

        analyzed = analyze(net, inputs, args.eps, true_label, args)
        adv = adversary_example(net, true_label, inputs, args.eps)
        if analyzed:
            print("verified")
        else:
            print("not verified")

        if adv:
            print("Adversarial attack found adversarial example")
            if analyzed:
                statistics["verifiedAdversary"] += 1
            else:
                statistics["NOverifiedAdversary"] += 1
        else:
            print("NOT found adversarial example")
            if analyzed:
                statistics["verifiedNOAdversary"] += 1
            else:
                statistics["NOverifiedNOAdversary"] += 1

        print()
        if idx % 10 == 0:
            print(statistics)

    print(statistics)


if __name__ == "__main__":
    main()
