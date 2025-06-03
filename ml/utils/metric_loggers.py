from torchmetrics import Metric
from torchmetrics.classification.confusion_matrix import BinaryConfusionMatrix
from torchmetrics.classification.precision_recall_curve import BinaryPrecisionRecallCurve
from torchmetrics.classification.roc import BinaryROC
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Callable, Optional


def log_confusion_matrix(
    log: Callable,
    confusion_matrix: Metric,
    stage: str,
    log_dir: Path,
    current_epoch: int,
    idx_to_class: dict
):
    conf_matrix = confusion_matrix.compute()

    # For binary confusion matrices, reorder so minority class comes first
    if isinstance(confusion_matrix, BinaryConfusionMatrix):
        # Calculate class totals (true positives + false negatives for each class)
        class_0_total = conf_matrix[0, 0] + conf_matrix[1, 0]  # TN + FN
        class_1_total = conf_matrix[0, 1] + conf_matrix[1, 1]  # FP + TP
        
        # Determine which class is minority
        minority_is_class_1 = class_1_total < class_0_total
        
        if minority_is_class_1:
            # Swap rows and columns to put minority class (class 1) first
            conf_matrix_display = conf_matrix[[1, 0]][:, [1, 0]]
            labels_display = [idx_to_class[1], idx_to_class[0]]
        else:
            # Keep original order (class 0 is already minority)
            conf_matrix_display = conf_matrix
            labels_display = [idx_to_class[0], idx_to_class[1]]
    else:
        # For multi-class, keep original order
        conf_matrix_display = conf_matrix
        labels_display = [idx_to_class[i] for i in range(len(idx_to_class))]

    fig, ax = plt.subplots()
    sns.heatmap(
        conf_matrix_display.cpu().numpy(),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_display,
        yticklabels=labels_display,
        ax=ax
    )
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    if stage == 'validation':
        ax.set_title(f'Confusion Matrix {stage} Epoch {current_epoch}')
    elif stage == 'test':
        ax.set_title('Confusion Matrix on test data')

    if isinstance(confusion_matrix, BinaryConfusionMatrix):
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
    idx_to_class: dict,
    minority_class_ratio: Optional[float],
):
    precision_recall_curves_dir = log_dir / 'precision_recall_curves'
    os.makedirs(precision_recall_curves_dir, mode=0o777, exist_ok=True)

    fig, ax = precision_recall_curve.plot(score=True)

    ax.set_ylabel("True positive rate (Recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if (not isinstance(precision_recall_curve, BinaryPrecisionRecallCurve)) and hasattr(ax, 'legend_') and ax.legend_:
        class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
        ax.legend(class_labels, title="Classes")

    if minority_class_ratio is not None:
        baseline = minority_class_ratio
    
        ax.axhline(
            y=baseline,
            color='red',
            linestyle='--', 
            alpha=0.7, 
            label=f'Random Classifier (AP={baseline:.3f})'
        )

        ax.text(
            0.5,
            baseline + 0.02,
            'Random classifier', 
            horizontalalignment='center', 
            verticalalignment='bottom',
            color='red', 
            fontsize=10,
            alpha=0.8
        )

    if stage == 'validation': 
        ax.set_title(f'Precision Recall Curve validation Epoch {current_epoch}')
    elif stage == 'test':
        ax.set_title('Precision Recall Curve on test data')

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

    if (not isinstance(roc_curve, BinaryROC)) and hasattr(ax, 'legend_') and ax.legend_:
        class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
        ax.legend(class_labels, title="Classes")
    
    if isinstance(roc_curve, BinaryROC):
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.7)
        
        ax.text(
            0.5,
            0.55,
            'Random classifier', 
            horizontalalignment='center', 
            verticalalignment='top',
            color='red', 
            fontsize=10,
            alpha=0.8,
            rotation=40
        )

    if stage == 'validation':
        ax.set_title(f'ROC curve validation Epoch {current_epoch}')
    elif stage == 'test':
        ax.set_title('ROC curve for test data')

    fig.savefig(roc_curves_dir / f"{stage}_roc_curve_epoch_{current_epoch}.png", dpi=300)
    plt.close(fig)