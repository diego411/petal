import os.path

from ml.dataset import get_data_loaders, download_data
from ml.models.ResCNNet import ResCNNet
from ml.models.CNNet import CNNet
#from ml.RandomClassifier import RandomClassifier
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

        print('-------------------Evaluation on validation set-------------------')
        test(validation_dataloader, model, cost, verbose=1)

        print('-------------------Evaluation on training set-------------------')
        test(train_dataloader, model, cost, verbose=1)

        # if accuracy > best_accuracy:
        directory = '../out/models'
        if not os.path.isdir(directory):
            os.makedirs(directory, mode=0o777, exist_ok=True)

        torch.save(model.state_dict(), f'{directory}/{epochs}_{date}.pth')
        # best_accuracy = accuracy

    # print(best_accuracy)
    print('Done!')


def evaluate_model(model, test_dataloader):
    cm, _ = build_confusion_matrix_with_metrics(
        model=model,
        dataloader=test_dataloader,
        class_names=classes
    )

    confusion_matrix_directory = 'out/confusion_matrix/'
    if not os.path.isdir(confusion_matrix_directory):
        os.makedirs(confusion_matrix_directory, mode=0o777, exist_ok=True)

    save_confusion_matrix(
        cm=cm,
        classes=classes,
        path=f'{confusion_matrix_directory}/confusion_matrix_{time.ctime(time.time())}.png'
    )

def get_class_counts(dataloader):
    from collections import Counter

    # Assuming your DataLoader is named 'dataloader'
    class_counts = Counter()

    for _, labels in dataloader:
        class_counts.update(labels.numpy())  # Convert tensor to NumPy for counting

    print(class_counts)

if __name__ == '__main__':
    dataset_type = 'post-labeled'
    spectrogram_type = 'mel-spectrogram'
    
    classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
    n_output = 6

    if dataset_type == 'pre-labeled':
        classes.append('neutral')
        n_output = 7

    #download_data()
    train_dataloader, test_dataloader, validation_dataloader = get_data_loaders(dataset_type, spectrogram_type)
    print(get_class_counts(test_dataloader))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResCNNet(n_output=n_output).to(device)
    # model.load_state_dict(torch.load(f'../out/models/50_Fri Feb 14 16:01:54 2025.pth'))
    train_model(model, train_dataloader, validation_dataloader)
    evaluate_model(model, test_dataloader)
