from loguru import logger

from project import Experiment
from project.config import config
from project.models import models

import torch

test = torch.randn(1, 3, 32, 32)

model_cls = models[config.model_config.name]
print(config.model_config)
print(model_cls)
model = model_cls(**config.model_config.dict())
print(model)
logger.info("Loaded")
print(model(test).size())
