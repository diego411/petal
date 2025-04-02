from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader 
from torch.utils.data.dataset import Subset
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class PetalDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_type:str='post-labeled', # TODO: validate this param
        spectrogram_type:str='spectrogram', # TODO: validate this param
        binary:bool=False, 
        train_ratio:float=0.7,
        validation_ratio:float=0.1,
        seed:int=42,
        batch_size:int=16,
        number_of_workers:int=1,
        verbose:bool=True
    ):
        super().__init__()
        print(number_of_workers)
        self.dataset_type = dataset_type
        self.spectrogram_type = spectrogram_type
        self.binary = binary
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio 
        self.seed = seed 
        self.batch_size = batch_size
        self.number_of_workers = number_of_workers
        self.verbose = verbose
        
    def setup(self, stage=None):
        dataset = self.create_dataset()
        self.train_dataset, self.test_dataset, self.validation_dataset = self.create_stratified_data_split(dataset)

    def create_stratified_data_split(self, dataset: Dataset) -> Tuple[Subset, Subset, Subset]:
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

    def create_dataset(self) -> Dataset:
        raise NotImplementedError 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.number_of_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

