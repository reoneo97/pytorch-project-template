from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel
from yaml.events import StreamStartEvent
import project
import yaml
ROOT = Path(project.__file__).resolve().parent


class ModelConfig(BaseModel):
    name: str
    channels: int
    dim_sizes: List[int]


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
    exp_config: ExperimentConfig


def create_config():
    config_path = Path(ROOT, "config.yml")
    config_dict = yaml.load(config_path, loader=yaml.Loader)

    print(config_path)

    return


config = create_config()
