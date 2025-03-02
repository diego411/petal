import os.path

from ml.dataset import get_data_loaders, download_data
from ml.model import ResCNNet
from ml.functions import train, test
from ml.evaluation import build_confusion_matrix_with_metrics, save_confusion_matrix
import torch
import time


def train_model(model, train_dataloader, validation_dataloader):
    epochs = 50
    learning_rate = 0.0001  # best_trial.config["lr"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cost = torch.nn.CrossEntropyLoss()

    date = time.ctime(time.time())
    # best_accuracy = 0

    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        train(train_dataloader, model, optimizer, cost)
        test(validation_dataloader, model, cost, verbose=1)

        # if accuracy > best_accuracy:
        directory = '../out/models'
        if not os.path.isdir(directory):
            os.makedirs(directory, mode=0o777, exist_ok=True)

        torch.save(model.state_dict(), f'{directory}/{epochs}_{date}.pth')
        # best_accuracy = accuracy

    # print(best_accuracy)
    print('Done!')


def evaluate_model(model, test_dataloader):
    classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    cm, _ = build_confusion_matrix_with_metrics(
        model=model,
        dataloader=test_dataloader,
        class_names=classes
    )

    confusion_matrix_directory = '../out/confusion_matrix/'
    if not os.path.isdir(confusion_matrix_directory):
        os.makedirs(confusion_matrix_directory, mode=0o777, exist_ok=True)

    save_confusion_matrix(
        cm=cm,
        classes=classes,
        path=f'{confusion_matrix_directory}/confusion_matrix_{time.ctime(time.time())}.png'
    )


if __name__ == '__main__':
    # download_data()
    train_dataloader, test_dataloader, validation_dataloader = get_data_loaders()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResCNNet(n_output=7).to(device)
    # model.load_state_dict(torch.load(f'../out/models/50_Fri Feb 14 16:01:54 2025.pth'))
    print("Number of training samples:", len(train_dataloader))
    train_model(model, train_dataloader, validation_dataloader)
    evaluate_model(model, test_dataloader)
