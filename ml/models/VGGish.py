from ml.models.PetalModule import PetalModule
import torchvggish
import torch
from torch import Tensor
from torch.optim import AdamW 
import torch.nn as nn

class VGGish(PetalModule):
    def __init__(self, n_output:int=1, lr=1e-4):
        super().__init__(n_output=n_output)
        print("[Model] Using VGGish")

        self.learning_rate = lr
        # Load the pre-trained VGGish model
        self.vggish = torchvggish.vggish()
        self.vggish.eval()  # VGGish is used only for feature extraction

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_output)
        )
    
    def _ensure_vggish_device_sync(self, device):
        """Ensure all VGGish components are on the specified device"""
        # Move main model
        self.vggish = self.vggish.to(device)
        
        # Move postprocessor internal tensors
        if hasattr(self.vggish, 'pproc') and self.vggish.pproc:
            if hasattr(self.vggish.pproc, '_pca_matrix'):
                self.vggish.pproc._pca_matrix = self.vggish.pproc._pca_matrix.to(device)
            if hasattr(self.vggish.pproc, '_pca_means'):
                self.vggish.pproc._pca_means = self.vggish.pproc._pca_means.to(device)
    
    def forward(self, x: Tensor) -> Tensor:
        # Ensure all VGGish components are on the same device as input
        device = x.device
        
        # Check if we need to move anything
        vggish_device = next(self.vggish.parameters()).device
        pca_matrix_device = None
        if hasattr(self.vggish, 'pproc') and hasattr(self.vggish.pproc, '_pca_matrix'):
            pca_matrix_device = self.vggish.pproc._pca_matrix.device
        
        if vggish_device != device or (pca_matrix_device and pca_matrix_device != device):
            self._ensure_vggish_device_sync(device)
        
        with torch.no_grad():
            features = self.vggish(x)  # Extract VGGish embeddings
        return self.classifier(features).squeeze()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
  