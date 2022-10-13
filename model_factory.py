import models


def factory(cfg):
    if cfg.model.name == 'LeNet':
        return models.LeNet(cfg)
    else:
        raise NotImplementedError(f"Model {cfg.model.name}")
