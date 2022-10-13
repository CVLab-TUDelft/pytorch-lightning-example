import torch


class LeNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(cfg.dataset.channels, 6, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 120, 5),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, cfg.dataset.classes),
        )

    def forward(self, x):
        return self.network(x)