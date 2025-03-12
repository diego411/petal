import lightning.pytorch as L
from lightning.pytorch.loggers import CSVLogger, Logger
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix, PrecisionRecallCurve, ROC
from torch import Tensor
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Optional
from pathlib import Path

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
        self.confusion_matrix = ConfusionMatrix(task="binary" if n_output == 1 else "multiclass", num_classes=n_output) 
        self.precision_recall_curve = PrecisionRecallCurve(task="binary" if n_output == 1 else "multiclass", num_classes=n_output)
        self.roc_curve = ROC(task="binary" if n_output == 1 else "multiclass", num_classes=n_output)
        
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
        self.confusion_matrix.update(predictions, targets)
        self.precision_recall_curve.update(predictions, targets)
        self.roc_curve.update(predictions, targets)
    
    def _log_metrics(self, stage: str):
        self.log(f"{stage}_accuracy", self.accuracy.compute())
        self.log(f"{stage}_precision", self.precision.compute())
        self.log(f"{stage}_recall", self.recall.compute())
        self.log(f"{stage}_f1", self.f1.compute())
        self.log(f"{stage}_auroc", self.auroc.compute())

        csv_logger = self.get_csv_logger()
        if csv_logger is None:
            return
        if csv_logger.log_dir is None:
            return
        log_dir = Path(csv_logger.log_dir) / 'files'
        os.makedirs(log_dir, mode=0o777, exist_ok=True)

        current_epoch = self.trainer.current_epoch
        self.log_confusion_matrix(stage, log_dir, current_epoch)
        self.log_precision_recall_curve(stage, log_dir, current_epoch)
        self.log_roc_curve(stage, log_dir, current_epoch)

    def _reset_metrics(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()
        self.confusion_matrix.reset()
        self.precision_recall_curve.reset()
        self.roc_curve.reset()
    
    def log_confusion_matrix(self, stage: str, log_dir: Path, current_epoch: int):
        conf_matrix = self.confusion_matrix.compute()

        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix {stage} Epoch {current_epoch}')

        self.log(
            f'{stage}_true_negatives',
            conf_matrix[0, 0].item(),
            sync_dist=True
        )
        self.log(
            f'{stage}_false_positives',
            conf_matrix[0, 1].item(),
            sync_dist=True
        )
        self.log(
            f'{stage}_false_negatives',
            conf_matrix[1, 0].item(),
            sync_dist=True
        )
        self.log(
            f'{stage}_true_positives',
            conf_matrix[1, 1].item(),
            sync_dist=True
        )
        
        confusion_matrices_dir = log_dir / 'confusion_matrices'
        os.makedirs(confusion_matrices_dir, mode=0o777, exist_ok=True)

        fig.savefig(confusion_matrices_dir / f'{stage}_confustion_matrix_epoch_{current_epoch}.png')
        plt.close(fig)
    
    def log_precision_recall_curve(self, stage: str, log_dir: Path, current_epoch: int):
        precision_recall_curves_dir = log_dir / 'precision_recall_curves'
        os.makedirs(precision_recall_curves_dir, mode=0o777, exist_ok=True)

        fig, ax = self.precision_recall_curve.plot(score=True)
        fig.savefig(precision_recall_curves_dir / f"{stage}_precision_recall_curve_epoch_{current_epoch}.png", dpi=300)
        plt.close(fig)

    def log_roc_curve(self, stage: str, log_dir: Path, current_epoch: int):
        roc_curves_dir = log_dir / 'roc_curves'
        os.makedirs(roc_curves_dir, mode=0o777, exist_ok=True)

        fig, ax = self.roc_curve.plot(score=True)
        fig.savefig(roc_curves_dir / f"{stage}_roc_curve_epoch_{current_epoch}.png", dpi=300)
        plt.close(fig)

    def get_csv_logger(self) -> Optional[Logger]:
        csv_loggers = list(filter(lambda logger: isinstance(logger, CSVLogger), self.trainer.loggers))

        if len(csv_loggers) == 0:
            return None
        
        return csv_loggers[0]

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
