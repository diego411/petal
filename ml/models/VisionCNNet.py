from ml.models.PetalModule import PetalModule
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch import Tensor


class VisionCNN(PetalModule):
    def __init__(self, n_output=1, freeze=True, lr=1e-4): 
        super().__init__(n_output=n_output)

        self.lr = lr
        resnet18 = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])  # Keep deeper features
        
        if freeze:
            self.freeze()
        
        self.l1 = nn.Linear(512, n_output)

    def forward(self, x) -> Tensor:
        x = self.resnet(x).squeeze()
        return self.l1(x).squeeze()
    
    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
