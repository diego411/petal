from torch import nn
from torchvision import models


class ResCNNet(nn.Module):
    def __init__(self, n_output=1, freeze=True):
        super().__init__()

        resnet18 = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])  # Keep deeper features

        if freeze:
            self.freeze()

        self.l1 = nn.Linear(512, n_output)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        return self.l1(x) 

    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
