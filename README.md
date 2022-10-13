# PyTorch Lightning Example

*Robert-Jan Bruintjes*

This repository serves as a starting point for any PyTorch-based Deep Computer Vision experiments. It uses PyTorch Lightning to power the training logic (including multi-GPU training), OmegaConf to provide a flexible and reproducible way to set the parameters of experiments, and Weights & Biases to log all experimental results and outputs.

## Installation

CUDA: `conda env create -f environment-cuda11.3.yml`

CPU: `conda env create -f environment.yml`

## Extending

### Adding models

- Add the code for the model in a new file in `models`;
- Import & call the new model in `model_factory.py`

### Adding datasets

- Add the code for the dataset in a new file in `datasets`. Make sure to make methods for creating dataloaders for train and val/test.
- Import & call the new methods in `dataset_factory.py`

### Resuming training from a checkpoint

W&B saves checkpoints as "artifacts".

- Use code like below to make W&B download the `Runner` checkpoint to disk:

```python
artifact_name = f"{cfg.wandb.entity}/{project_name}/{artifact_name}"
print(artifact_name)
artifact = wandb_logger.experiment.use_artifact(artifact_name)
directory = artifact.download()
filename = os.path.join(directory, 'model.ckpt')
```

- Add flag `ckpt_path=filename` to the call to `Trainer.fit()`
- Consider generalizing this by making `artifact_name` given by a new config key `cfg.resume.artifact`
