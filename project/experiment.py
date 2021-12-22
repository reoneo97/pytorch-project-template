import pytorch_lightning as pl
from .models import conv


class Experiment(pl.LightningModule):
    """LightningModule to perform the actual training and data initialization"""

    def __init__(self, model, params):
        self.model = model
        self.lr = params.lr
