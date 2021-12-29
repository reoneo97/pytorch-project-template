from loguru import logger

from project import Experiment
from project.config import config
from project.models import models

import torch
from pytorch_lightning import Trainer
test = torch.randn(1, 3, 32, 32)

model_cls = models[config.model_config.name]
model = model_cls(**config.model_config.dict())
logger.info("Loaded")

expt = Experiment(model=model, **config.experiment_config.dict())
trainer = Trainer(**config.trainer_config.dict())
trainer.fit(expt)
