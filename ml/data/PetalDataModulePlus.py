from lightning.pytorch import LightningDataModule
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from ml.data.data_util import create_spectrogram_images
from typing import Tuple
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data.SpectrogramDataset import SpectrogramDataset
from pathlib import Path


class PetalDataModulePlus(LightningDataModule):

    def __init__(
        self,
        dataset_type:str='post-labeled', # TODO: validate this param
        spectrogram_type:str='spectrogram', # TODO: validate this param
        binary:bool=False, 
        pretrained_model_name:str='resnet18', # TODO: link this param to model
        train_ratio:float=0.7,
        validation_ratio:float=0.1,
        seed:int=42,
        batch_size:int=16,
        cache_file: str = 'dataset_cache.pth',
        number_of_workers:int=1,
        verbose:bool=True
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.spectrogram_type = spectrogram_type
        self.binary = binary
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio 
        self.seed = seed 
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.number_of_workers = number_of_workers
        self.verbose = verbose
        
        timm_model = timm.create_model(pretrained_model_name, pretrained=True) 
        self.transform = create_transform(**resolve_data_config(
            pretrained_cfg=timm_model.pretrained_cfg,
            model=timm_model
        ))

    def setup(self, stage=None):
        dataset = self.create_dataset()
        self.train_dataset, self.test_dataset, self.validation_dataset = self.create_stratified_data_split(dataset)

    def create_stratified_data_split(self, dataset: SpectrogramDataset) -> Tuple[Subset, Subset, Subset]:
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

    def create_dataset(self) -> SpectrogramDataset:
        _, _, dataframe = create_spectrogram_images(self.dataset_type, self.binary, self.verbose)
        dataset = SpectrogramDataset(
            dataframe=dataframe,
            transforms=self.transform,
            cache_file=self.cache_file
        )

        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = dataset.idx_to_class 

        class_counts = {dataset.classes[i]: count for i, count in Counter(dataset.targets).items()}

        print("[Datamodule] Number of samples per class in whole dataset:", class_counts)

        return dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.number_of_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

