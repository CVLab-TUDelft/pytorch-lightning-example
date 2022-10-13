import torch
import torchvision


# def train_transform(cfg):
#     return torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(cfg.dataset.mean, cfg.dataset.std),
#     ])

def cifar10_train_loader(cfg):
    raise NotImplementedError('transforms')
    return torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(cfg.dataset.root, train=True, download=True),
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )

def cifar10_test_loader(cfg):
    raise NotImplementedError('transforms')
    return torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(cfg.dataset.root, train=False, download=True),
        batch_size=cfg.train.batch_size,
    )