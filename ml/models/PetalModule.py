import lightning.pytorch as L
from lightning.pytorch.loggers import CSVLogger, Logger
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix, PrecisionRecallCurve, ROC
from torch import Tensor
from typing import Tuple
from ml.utils.metric_loggers import log_confusion_matrix, log_precision_recall_curve, log_roc_curve
import os
from typing import Optional
from pathlib import Path
from collections import Counter
from torch import Tensor

class PetalModule(L.LightningModule):
    def __init__(
        self,
        n_output:int=1,
        weigh_loss:bool=False
    ):
        super().__init__()

        self.save_hyperparameters()
        self.n_output = n_output
        self.weight_loss = weigh_loss

        task = "binary" if n_output == 1 else "multiclass" 
        self.accuracy = Accuracy(task=task, num_classes=n_output)
        self.precision = Precision(task=task, num_classes=n_output, average="macro")
        self.recall = Recall(task=task, num_classes=n_output, average="macro")
        self.weighted_f1 = F1Score(task=task, num_classes=n_output, average="weighted")
        self.macro_f1 = F1Score(task=task, num_classes=n_output, average="macro")
        self.auroc = AUROC(task=task, num_classes=n_output, average="macro")
        self.confusion_matrix = ConfusionMatrix(task=task, num_classes=n_output) 
        self.precision_recall_curve = PrecisionRecallCurve(task=task, num_classes=n_output)
        self.roc_curve = ROC(task=task, num_classes=n_output)

    def setup(self, stage=None):
        if not self.weight_loss:
            self.criterion = nn.BCEWithLogitsLoss() if self.n_output == 1 else nn.CrossEntropyLoss()
            return

        train_class_counts: Counter = self.trainer.datamodule.train_class_counts
        assert train_class_counts is not None, "No train_class_counts in datamodule"

        if self.n_output == 1:
            assert len(train_class_counts) == 2, "Unexpected number of classes in train_class_counts"
            pos_weight = train_class_counts[1] / train_class_counts[0]
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=Tensor([pos_weight]))
        else:
            weights = [None] * len(train_class_counts)

            number_of_train_samples = sum(train_class_counts.values())
            for cls, count in train_class_counts.items():
                weights[cls] = number_of_train_samples / (self.n_output * count)
            self.criterion = nn.CrossEntropyLoss(weight=Tensor(weights))
        
    def forward(self, x) -> Tensor:
        raise NotImplementedError 
    
    def single_step(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        features: Tensor; targets: Tensor
        features, targets = batch
        targets: Tensor = targets.float() if self.n_output == 1 else targets  # Ensure correct dtype
        predictions: Tensor = self(features)
        loss: Tensor = self.criterion(predictions, targets)
        
        return targets, predictions, loss 
    
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
    
    def _update_metrics(self, targets: Tensor, predictions: Tensor):
        targets = targets.long() if self.n_output == 1 else targets
        self.accuracy.update(predictions, targets)
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
        self.weighted_f1.update(predictions, targets)
        self.macro_f1.update(predictions, targets)
        self.auroc.update(predictions, targets)
        self.confusion_matrix.update(predictions, targets)
        self.precision_recall_curve.update(predictions, targets)
        self.roc_curve.update(predictions, targets)
    
    def _log_metrics(self, stage: str):
        self.log(f"{stage}_accuracy", self.accuracy.compute())
        self.log(f"{stage}_precision", self.precision.compute())
        self.log(f"{stage}_recall", self.recall.compute())
        self.log(f"{stage}_weighted_f1", self.weighted_f1.compute())
        self.log(f"{stage}_macro_f1", self.macro_f1.compute())
        self.log(f"{stage}_auroc", self.auroc.compute())

        csv_logger = self._get_csv_logger()
        if csv_logger is None:
            return
        if csv_logger.log_dir is None:
            return
        log_dir = Path(csv_logger.log_dir) / 'files'
        os.makedirs(log_dir, mode=0o777, exist_ok=True)

        current_epoch = self.trainer.current_epoch
        idx_to_class = self.trainer.datamodule.idx_to_class # type: ignore
        
        minority_class_ratio = None
        if stage == 'validation':
            minority_class_ratio = self.trainer.datamodule.validation_minority_class_ratio
        elif stage == 'test':
            minority_class_ratio = self.trainer.datamodule.test_minority_class_ratio

        log_confusion_matrix(self.log, self.confusion_matrix, stage, log_dir, current_epoch, idx_to_class)
        log_precision_recall_curve(
            self.precision_recall_curve,
            stage,
            log_dir,
            current_epoch,
            idx_to_class,
            minority_class_ratio
        )
        log_roc_curve(self.roc_curve, stage, log_dir, current_epoch, idx_to_class)

    def _reset_metrics(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.weighted_f1.reset()
        self.macro_f1.reset()
        self.auroc.reset()
        self.confusion_matrix.reset()
        self.precision_recall_curve.reset()
        self.roc_curve.reset()
    
    def _get_csv_logger(self) -> Optional[Logger]:
        csv_loggers = list(filter(lambda logger: isinstance(logger, CSVLogger), self.trainer.loggers))

        if len(csv_loggers) == 0:
            return None
        
        return csv_loggers[0]