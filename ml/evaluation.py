import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


def build_confusion_matrix_with_metrics(model, dataloader, class_names):
    model.eval()  # Set the model to evaluation mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Iterate through the dataset and collect predictions
    y_prediction_list = []
    y_true_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, y_prediction = torch.max(outputs, 1)
            y_prediction_list.extend(y_prediction.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())

    confusion_mat = confusion_matrix(y_true_list, y_prediction_list)
    report = classification_report(y_true_list, y_prediction_list, target_names=class_names)

    res = []
    for l in range(0, len(class_names)):
        precision, recall, _, _ = precision_recall_fscore_support(
            np.array(y_true_list) == l,
            np.array(y_prediction_list) == l,
            pos_label=True,
            average=None
        )
        res.append([class_names[l], recall[0], recall[1]])
    return confusion_mat, res


def save_confusion_matrix(cm, classes, path):
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
