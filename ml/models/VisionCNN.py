from ml.models.PetalModule import PetalModule
import torch.optim as optim
from torch import Tensor
import timm


class VisionCNN(PetalModule):
    def __init__(
        self,
        pretrained_model_name:str='resnet18',
        n_output:int=1,
        freeze=True,
        lr=1e-4
    ): 
        super().__init__(n_output=n_output)
        print("[Model] Using VisionCNN")

        self.learning_rate = lr
        self.pretrained_model = timm.create_model(
            pretrained_model_name,
            pretrained=True,
            num_classes=n_output
        )
        
        if freeze:
            self.freeze()

    def forward(self, x) -> Tensor:
        return self.pretrained_model(x).squeeze()
    
    def predict_step(self, batch, batch_idx):
        return self(batch) 

    def freeze(self):
      for param in self.pretrained_model.parameters():
        param.requires_grad = False 

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        lambda_lr = lambda epoch: 1.0 - (epoch / self.trainer.max_epochs)
        scheduler_config = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_lr),
            'monitor': 'validation_f1'
        }

        return [optimizer], scheduler_config
