import torch
import torchvision


def transforms(cfg):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

def mnist_train_loader(cfg):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            cfg.dataset.root,
            transform=transforms(cfg),
            train=True,
            download=True),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
    )

def mnist_test_loader(cfg):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            cfg.dataset.root,
            transform=transforms(cfg),
            train=False,
            download=True
        ),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
    )