from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
from yaml.events import StreamStartEvent
import project
import yaml

ROOT = Path(project.__file__).resolve().parent.parent


class ModelConfig(BaseModel):
    name: str
    channels: int
    n_classes: int
    dim_sizes: List[int]
    kernel_size: Optional[int] = 3
    stride: Optional[int] = 2
    padding: Optional[int] = 1
    dropout_p: Optional[float] = 0.2

    # Validation for Float Parameter

    # Validation for Dim_sizes


class TrainerConfig(BaseModel):
    n_gpus: int
    max_epochs: int
    auto_lr_find: bool


class ExperimentConfig(BaseModel):
    name: str
    train_path: str
    val_path: str
    lr: float


class Config(BaseModel):
    model_config: ModelConfig
    trainer_config: TrainerConfig
    experiment_config: ExperimentConfig


def create_config():
    config_path = Path(ROOT, "config.yml")
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config = Config(**config_dict)
    return config


config = create_config()
