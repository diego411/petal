from ml.models.PetalModule import PetalModule
import torchvggish
import torch
from torch import Tensor
from torch.optim import AdamW 
import torch.nn as nn

class VGGish(PetalModule):
    def __init__(self, n_output:int=1, lr=1e-4):
        super().__init__(n_output=n_output)

        self.learning_rate = lr
        # Load the pre-trained VGGish model
        self.vggish = torchvggish.vggish()
        self.vggish.eval()  # VGGish is used only for feature extraction

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_output)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        self.vggish.to(x.device)
        with torch.no_grad():
            features = self.vggish(x)  # Extract VGGish embeddings
        return self.classifier(features)
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.vggish.to(self.device)
        return self
