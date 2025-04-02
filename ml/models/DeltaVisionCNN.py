from ml.models.PetalModule import PetalModule
import torch.optim as optim
from torch import Tensor
import timm
from typing import Tuple
from torch import nn
from collections import OrderedDict 


class DeltaVisionCNN(PetalModule):
    def __init__(
        self,
        pretrained_model_name:str='resnet18',
        n_output:int=1,
        freeze=True,
        lr=1e-4
    ): 
        super().__init__(n_output=n_output)
        print("[Model] Using DeltaVisionCNN")

        self.learning_rate = lr
        self.pretrained_model = timm.create_model(
            pretrained_model_name,
            pretrained=True,
        )

        pretrained_model_out_featues = list(self.pretrained_model.children())[-1].out_features
        self.classification_head = nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout(0.1)),
            ("dense1", nn.Linear(pretrained_model_out_featues, int(pretrained_model_out_featues / 2))),
            ("dropout2", nn.Dropout(0.1)),
            ("dense2", nn.Linear(int(pretrained_model_out_featues / 2), self.n_output)),
        ]))
        
        if freeze:
            self.freeze()

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        spectrogram, delta, delta_delta = x
        spectrogram_embeddings = self.pretrained_model(spectrogram)
        delta_embeddings = self.pretrained_model(delta)
        delta_delta_embeddings = self.pretrained_model(delta_delta)

        combined_embedding = spectrogram_embeddings * delta_embeddings * delta_delta_embeddings
        return self.classification_head(combined_embedding)
    
    def predict_step(self, batch, batch_idx):
        return self(batch) 

    def freeze(self):
      for param in self.pretrained_model.parameters():
        param.requires_grad = False 

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

