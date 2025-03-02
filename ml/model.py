from torch import nn
from torchvision import models


class ResCNNet(nn.Module):
    def __init__(self, n_output=1, freeze=True, dropout_rate=0.3):
        super().__init__()

        resnet18 = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-2])  # Keep deeper features

        if freeze:
            self.freeze()

        self.pool = nn.AdaptiveAvgPool2d(1)  # More flexible pooling
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, n_output)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
