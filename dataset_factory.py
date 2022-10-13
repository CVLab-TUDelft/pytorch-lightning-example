from datasets.mnist import mnist_train_loader, mnist_test_loader


def factory(cfg):
    if cfg.dataset.name == 'MNIST':
        return {
            'train': mnist_train_loader(cfg),
            'val': mnist_test_loader(cfg),
        }
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name}")