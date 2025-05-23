from ml.models.PetalModule import PetalModule
import torch
import torch.nn as nn
from torch import Tensor

class CNNet(PetalModule):
    def __init__(
        self,
        n_output:int=1,
        weigh_loss:bool=False,
        lr:float=1e-3,
        kernel_size:int=3,
        pool_kernel_size:int=2,
        stride:int=1,
        pool_stride:int=2,
        padding:int=1,
        dropout_rate:float=0.1
    ):
        super().__init__(n_output=n_output, weigh_loss=weigh_loss)
        print("[Model] Using CNNet")

        self.lr = lr

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            
            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            
            nn.Conv2d(256, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            
            # Global pooling - this ensures fixed output size regardless of input
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # With adaptive pooling, the output will always be 512 × 1 × 1
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, n_output)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)

        if self.n_output == 1:
            x = x.squeeze(1)

        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
