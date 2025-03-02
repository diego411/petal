import torch
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


def train(dataloader, model, optimizer, cost):
    model.train()
    size = len(dataloader.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch, (X, Y) in tqdm(enumerate(dataloader), total=len(dataloader)):

        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        prediction = model(X)
        loss = cost(prediction, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def test(dataloader, model, cost, verbose=0):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch, (X, Y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            X, Y = X.to(device), Y.to(device)
            prediction = model(X)

            test_loss += cost(prediction, Y).item()
            correct += (prediction.argmax(1) == Y).type(torch.float).sum().item()

            # Store predictions and labels for metric computation
            all_preds.extend(prediction.argmax(1).cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    test_loss /= size
    correct /= size

    # Compute precision, recall, and F1-score
    precision = precision_score(all_labels, all_preds, average='macro')  # Change to 'weighted' if needed
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    if verbose != 0:
        print(f'\nTest Error:\nacc: {(100 * correct):>0.1f}%, avg loss: {test_loss:>8f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n')

    return test_loss, correct, precision, recall, f1
