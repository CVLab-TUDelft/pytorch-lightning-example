from datasets.mnist import mnist_train_loader, mnist_test_loader


def factory(cfg):
    """
    Returns:
        train_loader: Training dataset loader.
        val_loader: Validation dataset loader. Validation is performed after
            each training epoch. If None, no validation is performed.
        test_loader: Test dataset loader. Testing is performed after fitting is
            done. If None, no testing is performed.
    """
    if cfg.dataset.name == 'MNIST':
        return mnist_train_loader(cfg), mnist_test_loader(cfg), None
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name}")