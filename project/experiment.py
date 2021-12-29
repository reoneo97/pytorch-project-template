import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from .config import config
from .datasets.dataset import cifar10_dataloader, cifar_map


class Experiment(pl.LightningModule):
    """LightningModule to perform the actual training and data initialization"""

    def __init__(self, model, train_path, val_path, lr, weight_decay,
                 batch_size, **kwargs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_path = train_path
        self.val_path = val_path
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
        img, labels = batch
        out = self.model(img)
        loss = self.loss_function(out, labels)
        return loss

    def train_dataloader(self):
        return cifar10_dataloader(self.train_path, self.batch_size)

    def val_dataloader(self):
        return cifar10_dataloader(self.val_path, self.batch_size)

    def loss_function(self, inp, tgt):
        loss_fn = CrossEntropyLoss()
        return loss_fn(inp, tgt)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_train_end(self) -> None:
        """Function to generate sample predictions and compare with ground
        truths at the end of model training
        """
        dl = self.train_dataloader()
        img, labels = next(iter(dl))
        img, labels = img[:6], labels[:6]
        gt_labels = [cifar_map[int(i)] for i in labels]
        gt_labels = "Ground Truth: " + "      ".join(gt_labels)
        if is_available():
            img = img.cuda()
        out = self.model(img)
        preds = out.argmax(axis=1)
        pred_labels = [cifar_map[int(i)] for i in preds]
        pred_labels = "Predictions: " + "      ".join(pred_labels)
        xlabel = gt_labels + "\n" + pred_labels
        grid = make_grid(img, normalize=True)
        grid = grid.permute(1, 2, 0)  # Change from (C x W x H) to (W x H x C)
        plt.imshow(grid.cpu().detach())
        plt.xlabel(xlabel)
        plt.show()
