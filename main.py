from project import Experiment
from loguru import logger
from project.config import config
from project.models import models

model = config.model_config.name
logger.info("Loaded")
