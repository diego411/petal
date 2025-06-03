from torch import nn
import torch.nn.functional as F
import torch


class RandomClassifier(nn.Module):
    def __init__(self, n_output=1):
        super().__init__()
        self.n_output = n_output

    def forward(self, x):
        batch_size = x.shape[0]
        random_logits = torch.rand(batch_size, self.n_output)
        return F.log_softmax(random_logits, dim=1)