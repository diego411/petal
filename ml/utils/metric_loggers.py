from torchmetrics import Metric
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Callable


def log_confusion_matrix(
    log: Callable,
    confusion_matrix: Metric,
    stage: str,
    log_dir: Path,
    current_epoch: int,
    idx_to_class: dict
):
    conf_matrix = confusion_matrix.compute()

    fig, ax = plt.subplots()
    sns.heatmap(
        conf_matrix.cpu().numpy(),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[idx_to_class[i] for i in range(len(idx_to_class))],
        yticklabels=[idx_to_class[i] for i in range(len(idx_to_class))],
        ax=ax
    )
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix {stage} Epoch {current_epoch}')

    log(
        f'{stage}_true_negatives',
        conf_matrix[0, 0].item(),
        sync_dist=True
    )
    log(
        f'{stage}_false_positives',
        conf_matrix[0, 1].item(),
        sync_dist=True
    )
    log(
        f'{stage}_false_negatives',
        conf_matrix[1, 0].item(),
        sync_dist=True
    )
    log(
        f'{stage}_true_positives',
        conf_matrix[1, 1].item(),
        sync_dist=True
    )
    
    confusion_matrices_dir = log_dir / 'confusion_matrices'
    os.makedirs(confusion_matrices_dir, mode=0o777, exist_ok=True)

    fig.savefig(confusion_matrices_dir / f'{stage}_confusion_matrix_epoch_{current_epoch}.png')
    plt.close(fig)

def log_precision_recall_curve(
    precision_recall_curve: Metric,
    stage: str,
    log_dir: Path,
    current_epoch: int,
    idx_to_class: dict
):
    precision_recall_curves_dir = log_dir / 'precision_recall_curves'
    os.makedirs(precision_recall_curves_dir, mode=0o777, exist_ok=True)

    fig, ax = precision_recall_curve.plot(score=True)

    if hasattr(ax, 'legend_') and ax.legend_:
        class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
        ax.legend(class_labels, title="Classes")

    ax.set_title(f'Precision Recall Curve {stage} Epoch {current_epoch}')

    fig.savefig(precision_recall_curves_dir / f"{stage}_precision_recall_curve_epoch_{current_epoch}.png", dpi=300)
    plt.close(fig)

def log_roc_curve(
    roc_curve: Metric,
    stage: str,
    log_dir: Path,
    current_epoch: int,
    idx_to_class: dict
):
    roc_curves_dir = log_dir / 'roc_curves'
    os.makedirs(roc_curves_dir, mode=0o777, exist_ok=True)

    fig, ax = roc_curve.plot(score=True)

    if hasattr(ax, 'legend_') and ax.legend_:
        class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
        ax.legend(class_labels, title="Classes")

    ax.set_title(f'ROC curve {stage} Epoch {current_epoch}')

    fig.savefig(roc_curves_dir / f"{stage}_roc_curve_epoch_{current_epoch}.png", dpi=300)
    plt.close(fig)