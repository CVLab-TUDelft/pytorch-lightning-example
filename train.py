"""
PyTorch Lightning example code, designed for use in TU Delft CV lab.

Copyright (c) 2022 Robert-Jan Bruintjes, TU Delft.
"""
# Package imports, from conda or pip
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import torchmetrics

# Imports of own files
import model_factory
import dataset_factory


class Runner(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # Runner needs to redirect any model.forward() calls to the actual
        # network
        return self.model(x)

    def configure_optimizers(self):
        if self.cfg.optimize.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimize.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.optimizer}")
        return optimizer

    def _step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.train_accuracy(preds, batch[1])

        # Log step-level loss & accuracy
        self.log("train/loss_step", loss)
        self.log("train/acc_step", self.train_accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy(preds, batch[1])

        # Log step-level loss & accuracy
        self.log("val/loss_step", loss)
        self.log("val/acc_step", self.val_accuracy)
        return loss

    def on_train_epoch_end(self):
        # Log the epoch-level training accuracy
        self.log('train/acc', self.train_accuracy.compute())
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        # Log the epoch-level validation accuracy
        self.log('val/acc', self.val_accuracy.compute())
        self.val_accuracy.reset()


def main():
    # Load defaults and overwrite by command-line arguments
    cfg = OmegaConf.load("config.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    # Set cache dir to W&B logging directory
    os.environ["WANDB_CACHE_DIR"] = os.path.join(cfg.wandb.dir, 'cache')
    wandb_logger = WandbLogger(
        save_dir=cfg.wandb.dir,
        project=cfg.wandb.project,
        name=cfg.wandb.experiment_name,
        log_model='all' if cfg.wandb.log else None,
        offline=not cfg.wandb.log,
        # Keyword args passed to wandb.init()
        entity=cfg.wandb.entity,
        config=OmegaConf.to_object(cfg),
    )

    # Create model using factory pattern
    model = model_factory.factory(cfg)

    # Create datasets using factory pattern
    loaders = dataset_factory.factory(cfg)
    train_dataset_loader = loaders['train']
    val_dataset_loader = loaders['val']

    # Tie it all together
    runner = Runner(cfg, model)
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
    )

    # Train + validate
    trainer.fit(runner, train_dataset_loader, val_dataset_loader)



if __name__ == '__main__':
    main()