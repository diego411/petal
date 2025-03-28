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

class PetalDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_type:str='post-labeled', # TODO: validate this param
        spectrogram_type:str='spectrogram', # TODO: validate this param
        pretrained_model_name:str='resnet18', # TODO: link this param to model
        train_ratio:float=0.7,
        validation_ratio:float=0.1,
        seed:int=42,
        batch_size:int=16,
        number_of_workers:int=1
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.spectrogram_type = spectrogram_type
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio 
        self.seed = seed 
        self.batch_size = batch_size
        self.number_of_workers = number_of_workers
        
        timm_model = timm.create_model(pretrained_model_name, pretrained=True) 
        self.transform = create_transform(**resolve_data_config(
            pretrained_cfg=timm_model.pretrained_cfg,
            model=timm_model
        ))


    def setup(self, stage=None):
        image_folder = self.create_image_folder()
        self.train_dataset, self.test_dataset, self.validation_dataset = self.create_stratified_data_split(image_folder)

    def create_stratified_data_split(self, dataset: ImageFolder) -> Tuple[Subset, Subset, Subset]:
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

    def create_image_folder(self) -> ImageFolder:
        spectrogram_path, mel_spectrogram_path = create_spectrogram_images(self.dataset_type)

        if self.spectrogram_type == 'spectrogram':
            path = spectrogram_path
        elif self.spectrogram_type == 'mel-spectrogram':
            path = mel_spectrogram_path
        else:
            raise RuntimeError("Unexpected spectrogram type")
        
        image_folder: ImageFolder = ImageFolder(
            root=str(path),
            transform=self.transform
            #transforms.Compose([
                #transforms.Resize((224, 224)),  # transforms.Resize((224,224))
                #transforms.ToTensor()
            #])
        )
        
        self.class_to_idx = image_folder.class_to_idx
        self.idx_to_class = {v: k for k, v in image_folder.class_to_idx.items()}

        labels = [label for _, label in image_folder.samples]
        class_counts = {image_folder.classes[i]: count for i, count in Counter(labels).items()}

        print("Number of samples per class in whole dataset:", class_counts)

        return image_folder

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.number_of_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

