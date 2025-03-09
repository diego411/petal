import lightning.pytorch as L
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from torch import Tensor
from typing import Tuple

class ResCNNetLightning(L.LightningModule):
    def __init__(self, n_output=1, freeze=True, lr=1e-4): # TODO: passing params through cli doesnt work
        super().__init__()

        print("NOUTPUT: ", n_output)
        
        self.save_hyperparameters()
        self.lr = lr
        resnet18 = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])  # Keep deeper features
        
        if freeze:
            self.freeze()
        
        self.n_output = n_output
        self.l1 = nn.Linear(512, n_output)
        self.criterion = nn.BCEWithLogitsLoss() if n_output == 1 else nn.CrossEntropyLoss()
        
        # Metrics
        self.accuracy = Accuracy(task="binary" if n_output == 1 else "multiclass", num_classes=n_output)
        self.precision = Precision(task="binary" if n_output == 1 else "multiclass", num_classes=n_output)
        self.recall = Recall(task="binary" if n_output == 1 else "multiclass", num_classes=n_output)
        self.f1 = F1Score(task="binary" if n_output == 1 else "multiclass", num_classes=n_output, average="weighted")
        self.auroc = AUROC(task="binary" if n_output == 1 else "multiclass", num_classes=n_output)
        
    def forward(self, x) -> Tensor:
        x = self.resnet(x).squeeze()
        return self.l1(x).squeeze()
    
    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def single_step(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        features: Tensor; targets: Tensor
        features, targets = batch
        targets: Tensor = targets.float() if self.n_output == 1 else targets  # Ensure correct dtype
        predictions: Tensor = self(features)
        loss: Tensor = self.criterion(predictions, targets)
        
        return targets, predictions, loss 
    
    def _update_metrics(self, targets: Tensor, predictions: Tensor):
        self.accuracy.update(predictions, targets)
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
        self.f1.update(predictions, targets)
        self.auroc.update(predictions, targets)
    
    def _log_metrics(self, stage: str):
        self.log(f"{stage}_accuracy", self.accuracy.compute())
        self.log(f"{stage}_precision", self.precision.compute())
        self.log(f"{stage}_recall", self.recall.compute())
        self.log(f"{stage}_f1", self.f1.compute())
        self.log(f"{stage}_auroc", self.auroc.compute())

    def _reset_metrics(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()

    def training_step(self, batch, batch_idx):
        _, _, loss = self.single_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        targets, predictions, loss = self.single_step(batch)
        self.log("validation_loss", loss)
        self._update_metrics(targets, predictions)
        return loss

    def on_validation_epoch_end(self):
        self._log_metrics("validation")
        self._reset_metrics()

    def test_step(self, batch, batch_idx):
        targets, predictions, loss = self.single_step(batch)
        self.log("test_loss", loss)
        self._update_metrics(targets, predictions)
        return loss 
    
    def on_test_epoch_end(self):
        self._log_metrics("test")
        self._reset_metrics()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
