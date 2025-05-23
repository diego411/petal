from ml.data.modules.PetalDataModule import PetalDataModule
from ml.data.data_util import create_spectrogram_images
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import Counter
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data.dataset import Subset, ConcatDataset, Dataset
from ml.data.sets.AugmentedDataset import AugmentedDataset
from ml.data.vision import load_and_transform
from typing import List, Tuple
from torch import Tensor


class CNNetDataModule(PetalDataModule):

    def __init__(self, *args, **kwargs): # TODO: link pretrained_model_name param to model
        super().__init__(*args, **kwargs)
        print("[Datamodule] Using CNNetDataModule")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def create_dataset(self):
        spectrogram_path, mel_spectrogram_path, librosa_spectrogram_path, _ = create_spectrogram_images(
            dataset_type=self.dataset_type,
            spectrogram_type=self.spectrogram_type,
            with_deltas=False,
            binary=self.binary,
            verbose=self.verbose,
            spectrogram_backend=self.spectrogram_backend
        )

        if self.spectrogram_backend == 'torch':
            if self.spectrogram_type == 'spectrogram':
                path = spectrogram_path
            elif self.spectrogram_type == 'mel-spectrogram':
                path = mel_spectrogram_path
            else:
                raise RuntimeError("Unexpected spectrogram type")
        elif self.spectrogram_backend == 'librosa':
            path = librosa_spectrogram_path
        else:
            raise RuntimeError('Unexpected spectrogram backend')
        
        image_folder: ImageFolder = ImageFolder(
            root=str(path),
            transform=self.transform
        )
        
        self.class_to_idx = image_folder.class_to_idx
        self.idx_to_class = {v: k for k, v in image_folder.class_to_idx.items()}

        class_counts = {image_folder.classes[i]: count for i, count in Counter(image_folder.targets).items()}

        print("[Datamodule] Number of samples per class in whole dataset:", class_counts)

        return image_folder
    
    def create_augmented_dataset(self, dataset: Subset, samples: List[dict]) -> Dataset:
        key = 'spectrogram_path' if self.spectrogram_type == 'spectrogram' else 'mel_spectrogram_path' if self.spectrogram_type == 'mel-spectrogram' else None
        if key == None:
            raise RuntimeError("Unexpected spectrogram type")
        
        def _to_tensor_sample(sample: dict) -> Tuple[Tensor, Tensor]:
            image_path = sample['paths'][key]
            image_tensor = load_and_transform(image_path, self.transform)
            return image_tensor, sample['label_index']
        
        augmented_dataset = AugmentedDataset(
            list(map(_to_tensor_sample, samples))
        )

        return ConcatDataset([dataset, augmented_dataset]) 
