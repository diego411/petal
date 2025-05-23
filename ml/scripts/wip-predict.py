import sys
from torch.utils.data import TensorDataset
import lightning.pytorch as L
from torch import Tensor
import timm
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.transforms._transforms import Spectrogram
import numpy as np
import torchaudio.transforms as T
import torch.nn.functional as F
from typing import Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from lightning.pytorch import Trainer
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from tabulate import tabulate

class VisionCNN(L.LightningModule):

    def __init__(
        self,
        pretrained_model_name:str='resnet18',
        n_output:int=1,
        freeze=True,
        lr=1e-4
    ): 
        super().__init__()

        print(pretrained_model_name)
        self.lr = lr
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
        if isinstance(batch, list):
            batch = batch[0]
        return self(batch) 


def get_number_of_fourier_transform_bins(waveform: Tensor) -> int:
    waveform_length = waveform.shape[-1]
    n_fft = 2 ** (waveform_length - 1).bit_length() // 2
    return max(n_fft, 8)  # Ensure n_fft is at least 8

def filter_spectrogram(spectrogram_tensor: Tensor) -> np.ndarray:
    spectrogram_numpy: np.ndarray = spectrogram_tensor.log2()[0, :, :].numpy().T
    
    infinity_filter = spectrogram_numpy != np.NINF
    filtered_spectrogram = np.where(
        infinity_filter,
        spectrogram_numpy,
        sys.float_info.min
    )  # replace remaining -inf with smallest float
    return filtered_spectrogram

def create_spectrogram_transform(waveform: Tensor) -> Spectrogram:
    n_fft = get_number_of_fourier_transform_bins(waveform)
    return T.Spectrogram(
        n_fft=n_fft,
        hop_length=n_fft // 4
    )

def create_spectrogram(waveform: Tensor) -> np.ndarray:
    spectrogram_transform: Spectrogram = create_spectrogram_transform(waveform)
    spectrogram_tensor: Tensor = spectrogram_transform(waveform)
    return filter_spectrogram(
        spectrogram_tensor
    ) 

def show_and_save_spectrogram_image(spectrogram: np.ndarray, n_fft: int, sample_rate: float, path: Optional[Path]):
    freqs = np.fft.rfftfreq(n_fft, d=1/sample_rate)  # Compute frequency values
    
    fig, ax = plt.subplots()
    img = ax.imshow(
        spectrogram,
        cmap='viridis',
        origin='lower',
        aspect='auto',
        extent=(
            0.0,
            spectrogram.shape[1],
            freqs[0],
            freqs[-1]
        )
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Spectrogram')
    cbar = fig.colorbar(img, ax=ax, label="Intensity")

    plt.show()

    if path is None:
        return

    # reset plot before saving image
    ax.axis('off')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.remove()

    # save spectrogram image
    fig.savefig(
        path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0,
        transparent=True
    )

    plt.close(fig)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_path>")
        sys.exit()

    ckpt_path = './ml/experiments/2052d23a216e191b702026bd2975b66a/version_0/checkpoints/epoch=43-validation_f1=0.85.ckpt'
    model = VisionCNN.load_from_checkpoint(ckpt_path)

    waveform, sample_rate = torchaudio.load(sys.argv[1])
    waveform = waveform + 1e-9

    n_fft = get_number_of_fourier_transform_bins(waveform)
    spectrogram = create_spectrogram(waveform)
    show_and_save_spectrogram_image(
        spectrogram=spectrogram,
        n_fft=n_fft,
        sample_rate=sample_rate,
        path=Path('spectrogram.png')
    )

    image = Image.open('spectrogram.png').convert('RGB')
    timm_model = timm.create_model('edgenext_small.usi_in1k', pretrained=True)
    timm_model = timm_model.eval()
    data_config = resolve_data_config(pretrained_cfg=timm_model.pretrained_cfg, model=timm_model)
    transform_image = create_transform(**data_config, is_training=False)
    image_tensor = transform_image(image).unsqueeze(0)

    dataset = TensorDataset(image_tensor)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    trainer = Trainer()
    predictions: List[Tensor] = trainer.predict(model, data_loader)
    logits = predictions[0]

    probabilities = F.softmax(logits, dim=0)
    table = []
    idx_to_class = {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'sad': 4, 'surprised': 5}
    for key, value in idx_to_class.items():
        table.append([key, f"{round(probabilities[value].item() * 100, 2)}%"])
    
    print(tabulate(table, headers=['Class', 'Probabilities']))