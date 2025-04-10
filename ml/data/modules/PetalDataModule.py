from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader 
from torch.utils.data.dataset import Subset, Dataset
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data.sets.SpectrogramDataset import SpectrogramDataset
from torchvision.datasets import ImageFolder
from typing import List, Tuple
import random
from pathlib import Path
from ml.data.augmentation import AUGMENT_TECHNIQUES
from ml.data.data_util import generate_spectrogram_for
from lightning.fabric.utilities.exceptions import MisconfigurationException
from typing import Optional


class PetalDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_type:str='post-labeled', # TODO: validate this param
        spectrogram_type:str='spectrogram', # TODO: validate this param
        spectrogram_backend:str='torch',
        binary:bool=False, 
        train_ratio:float=0.7,
        validation_ratio:float=0.1,
        augment_technique:Optional[str]=None,
        augment_ratio:float=0,
        seed:int=42,
        batch_size:int=16,
        number_of_workers:int=1,
        verbose:bool=True
    ):
        super().__init__()
        print(number_of_workers)
        self.dataset_type = dataset_type
        self.spectrogram_type = spectrogram_type
        self.spectrogram_backend = spectrogram_backend
        self.binary = binary
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio 
        self.seed = seed 
        self.batch_size = batch_size
        self.number_of_workers = number_of_workers
        self.verbose = verbose
        self.augment_ratio = augment_ratio

        if (augment_technique is not None) and (not augment_technique in AUGMENT_TECHNIQUES):
            raise MisconfigurationException(f"Supplied augment technique {augment_technique} not supported!")
        
        self.augment_technique = AUGMENT_TECHNIQUES[augment_technique] if augment_technique is not None else None

        random.seed(seed)
        
    def setup(self, stage=None):
        dataset = self.create_dataset()
        self.train_dataset, self.test_dataset, self.validation_dataset = self._create_stratified_data_split(dataset)

        augmented_samples = self._create_augment_samples(dataset)

        if augmented_samples is not None:
            self.train_dataset = self.create_augmented_dataset(
                dataset=self.train_dataset,
                samples=augmented_samples
            )

        print("[Datamodule] FInished data augmentation")
    
    def _create_augment_samples(self, dataset):
        if self.augment_technique is None:
            return

        print("[Datamodule] Starting data augmentation")

        augmented_samples: List[dict] = []
        train_sample_paths: List[Tuple[str, int]] = [dataset.samples[index] for index in self.train_dataset.indices]

        random.shuffle(train_sample_paths)

        for train_spectrogram_path, label_index in train_sample_paths[0:int(len(train_sample_paths) * self.augment_ratio)]:
            if self.augment_technique['type'] == 'audio':
                train_audio_path = Path(train_spectrogram_path.replace('spectrograms', 'audio')).with_suffix('.wav') 
                augmented_audio_path = self.augment_technique['apply'](train_audio_path) 
                spectrogram_paths = generate_spectrogram_for(
                    audio_path=augmented_audio_path,
                    dataset_type=self.dataset_type,
                    spectrogram_type=self.spectrogram_type,
                    spectrogram_backend=self.spectrogram_backend,
                    with_deltas=False
                )
                sample = {
                    'label_index': label_index,
                    'paths': spectrogram_paths
                }
                augmented_samples.append(sample)
            elif self.augment_technique['type'] == 'image':
                augmented_image_path = self.augment_technique['apply'](Path(train_spectrogram_path))
                sample = {
                    'label_index': label_index,
                    'paths': {
                        'spectrogram_path': augmented_image_path
                    }
                }
                augmented_samples.append(sample)
        
        return augmented_samples
    
    def _create_stratified_data_split(self, dataset: ImageFolder | SpectrogramDataset) -> Tuple[Subset, Subset, Subset]:
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

    def create_dataset(self) -> ImageFolder | SpectrogramDataset:
        raise NotImplementedError 
    
    def create_augmented_dataset(self, dataset: Subset, samples: List[dict]) -> Dataset:
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.number_of_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

