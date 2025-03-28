import torch
import torchaudio
import torchvggish.vggish_input as vggish_input
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch import LightningDataModule
from ml.data.data_util import get_audios
from typing import Tuple, List
from torch.utils.data.dataset import Subset
import numpy as np
from sklearn.model_selection import train_test_split


class VGGishDataset(Dataset):
    def __init__(self, audios: List[dict], max_patches: int = 1):
        self.audios = audios
        self.targets = list(
            map(lambda audio: audio['label_idx'], audios)
        )
        self.max_patches = max_patches

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = self.audios[idx]
        audio_path = audio['path']
        label_idx = audio['label_idx']

        # Load audio file
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert stereo to mono if needed

        # Convert to log-mel spectrogram (VGGish format)
        features = vggish_input.waveform_to_examples(waveform.numpy(), sr)
        features = torch.tensor(features, dtype=torch.float32)
        
        # Pad or truncate to fixed length
        num_patches = features.shape[0]

        if num_patches < self.max_patches:
            # Pad with zeros if too short
            pad_size = self.max_patches - num_patches
            padding = torch.zeros((pad_size, 1, 96, 64))
            features = torch.cat([features, padding], dim=0)
        elif num_patches > self.max_patches:
            # Truncate if too long
            features = features[:self.max_patches]

        return features.squeeze(1), torch.tensor(label_idx, dtype=torch.long)


class VGGishDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_type:str='post-labeled',
        train_ratio:float=0.7,
        validation_ratio:float=0.1,
        seed:int=42,
        batch_size:int=16,
        number_of_workers:int=1
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.number_of_workers = number_of_workers
        
    def setup(self, stage=None):
        """ Split dataset into train, validation, and test sets """
        dataset = VGGishDataset(get_audios(self.dataset_type))
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_stratified_data_split(dataset)

    def create_stratified_data_split(self, dataset: VGGishDataset) -> Tuple[Subset, Subset, Subset]:
        """
        Creates a stratified train/test/validation split from a dataset.
        
        Args:
            dataset: An ImageFolder dataset
            
        Returns:
            Tuple of (train, test, validation) subsets
        """
   
        # Get labels for all samples
        targets = np.array(dataset.targets)
        
        # Calculate the size of each split
        train_size = self.train_ratio
        validation_size = self.validation_ratio / (1 - train_size)  # Adjusted for two-step split
        
        # First split: train and (test+validation)
        train_indices, temp_indices = train_test_split(
            np.arange(len(targets)),
            test_size=(1 - train_size),
            stratify=targets,
            random_state=self.seed
        )
        
        # Get the targets for the temporary test+validation set
        temp_targets = targets[temp_indices]
        
        # Second split: test and validation from the temporary set
        test_indices, validation_indices = train_test_split(
            temp_indices,
            test_size=validation_size,
            stratify=temp_targets,
            random_state=self.seed
        )
        
        # Create subset datasets
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        validation_subset = Subset(dataset, validation_indices)
        
        return train_subset, test_subset, validation_subset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.number_of_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.number_of_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.number_of_workers)
