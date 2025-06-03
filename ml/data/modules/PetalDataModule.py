from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler 
from torch.utils.data.dataset import Subset, Dataset
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data.sets.SpectrogramDataset import SpectrogramDataset
from torchvision.datasets import ImageFolder
from typing import List, Tuple
import random
from pathlib import Path
from ml.data.augmentation import AUGMENT_TECHNIQUES, get_augment_audio_path, get_augment_image_path
from ml.data.data_util import generate_spectrogram_for, get_audio_path
from lightning.fabric.utilities.exceptions import MisconfigurationException
from typing import Optional, Dict
from src.utils.file import find_file_path
from collections import Counter
import torch
from ml.data.dataset_util import remove_class_samples

class PetalDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_type:str='post-labeled',
        spectrogram_type:str='spectrogram',
        spectrogram_backend:str='torch',
        binary:bool=False, 
        train_ratio:float=0.7,
        validation_ratio:float=0.1,
        augment_technique:Optional[str]=None,
        undersample_ratios:Optional[Dict[str, float]]=None,
        oversample_ratios:Optional[Dict[str, float]]=None,
        desired_distribution:Optional[Dict[str, float]]=None,
        undersample_ratios:Optional[Dict[str, float]]=None,
        seed:int=42,
        batch_size:int=16,
        number_of_workers:int=1,
        verbose:bool=True
    ):
        super().__init__()
        
        self.dataset_type = dataset_type
        self.spectrogram_type = spectrogram_type
        self.spectrogram_backend = spectrogram_backend
        self.binary = binary
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio 
        self.desired_distribution = desired_distribution
        self.undersample_ratios = undersample_ratios
        self.seed = seed 
        self.batch_size = batch_size
        self.number_of_workers = number_of_workers
        self.verbose = verbose
        self.undersample_ratios = undersample_ratios
        self.oversample_ratios = oversample_ratios

        if dataset_type != 'pre-labeled' and dataset_type != 'post-labeled':
            raise MisconfigurationException("Invalid dataset type! Only pre-labeled and post-labeled are supported!")
        
        if spectrogram_type != 'spectrogram' and spectrogram_type != 'mel-spectrogram':
            raise MisconfigurationException("Invalid spectrogram type! Only spectrogram and mel-spectrogram are supported!")
        
        if spectrogram_backend != 'torch' and spectrogram_backend != 'librosa':
            raise MisconfigurationException("Invalid spectrogram backend! Only torch and librosa are supported!")

        if binary and dataset_type != 'pre-labeled':
            raise MisconfigurationException("Binary mode is only supported for the pre-labeled dataset!")
        
        if validation_ratio < 0 or validation_ratio > 1:
            raise MisconfigurationException("Invalid validation ratio! Please supply a value between 0 and 1!")

        if train_ratio < 0 or train_ratio > 1:
            raise MisconfigurationException("Invalid train ratio! Please supply a value between 0 and 1!")

        if (augment_technique is not None) and (not augment_technique in AUGMENT_TECHNIQUES):
            raise MisconfigurationException(f"Supplied augment technique {augment_technique} not supported!")
        
        self.augment_technique = AUGMENT_TECHNIQUES[augment_technique] if augment_technique is not None else None

        random.seed(seed)
        
    def setup(self, stage=None):
        self.dataset = self.create_dataset()
        self.train_dataset, self.test_dataset, self.validation_dataset = self._create_stratified_data_split(self.dataset)

        self.validation_minority_class_ratio = None
        self.test_minority_class_ratio = None

        if self.binary:
            targets = np.array(self.dataset.targets)
            target_counter = Counter(targets)
            minority_class_idx = min(target_counter, key=target_counter.get)

            validation_target_counter = Counter(targets[self.validation_dataset.indices])
            test_target_counter = Counter(targets[self.test_dataset.indices])

            self.validation_minority_class_ratio = validation_target_counter[minority_class_idx] / sum(validation_target_counter.values())
            self.test_minority_class_ratio = test_target_counter[minority_class_idx] / sum(test_target_counter.values())
        
        self._init_distribution_variables()

        augmented_samples = self._create_augment_samples(self.dataset)

        if augmented_samples is not None:
            self.train_dataset = self.create_augmented_dataset(
                dataset=self.train_dataset,
                samples=augmented_samples
            )

        print("[Datamodule] FInished data augmentation")

    def _init_distribution_variables(self):
        self.train_indices = self.train_dataset.indices
        self.all_targets = np.array(self.dataset.targets)
        self.train_targets = self.all_targets[self.train_indices]
        self.train_class_counts = Counter(self.train_targets)
    
    def _create_augment_samples(self, dataset):
        if self.augment_technique is None:
            return

        assert self.augment_ratios is not None, "Provided augment technique but not augment ratios!"
        try:
            augment_ratios = {self.class_to_idx[key]: augment_ratio for key, augment_ratio in self.augment_ratios.items()}
        except KeyError:
            raise MisconfigurationException("Invalid class provided in augment_ratios parameter!")
        
        print("[Datamodule] Starting data augmentation")

        augmented_samples: List[dict] = []
        train_sample_paths: List[Tuple[str, int]] = [dataset.samples[index] for index in self.train_dataset.indices]

        random.shuffle(train_sample_paths)

        if self.oversample_ratios is None:
            return None
        
        for class_name, ratio in self.oversample_ratios.items():
            class_idx = self.class_to_idx[class_name]
            augment_paths = list(filter(
                lambda sample_path: sample_path[1] == class_idx,
                train_sample_paths
            ))    
            augment_paths = augment_paths[:int(len(augment_paths)* ratio)]

            for train_spectrogram_path, label_index in augment_paths:
                train_spectrogram_path = Path(train_spectrogram_path)
                if self.augment_technique['type'] == 'audio':
                    if self.binary:
                        train_audio_path = find_file_path(
                            file_name=train_spectrogram_path.with_suffix('.wav').name,
                            directory=get_audio_path(self.dataset_type)
                        )
                        if train_audio_path is None:
                            raise RuntimeError(f"Augmentation failed. Path to spectrogram was not found for following path: {train_spectrogram_path}")
                    else:
                        train_audio_path = Path(str(train_spectrogram_path).replace('spectrograms', 'audio')).with_suffix('.wav') 

                    augmented_audio_path = get_augment_audio_path(
                        audio_path=train_audio_path,
                        technique=self.augment_technique['label'],
                        dataset_type=self.dataset_type
                    )

                    if not augmented_audio_path.exists():
                        self.augment_technique['apply'](
                            audio_path=train_audio_path,
                            target_path=augmented_audio_path
                        )

                    augmented_image_path = get_augment_image_path(
                        image_path=train_audio_path,
                        technique=self.augment_technique['label'],
                        dataset_type=self.dataset_type
                    )
                    spectrogram_paths = generate_spectrogram_for(
                        audio_path=augmented_audio_path,
                        spectrogram_type=self.spectrogram_type,
                        spectrogram_backend=self.spectrogram_backend,
                        with_deltas=False,
                        target_path=augmented_image_path.parent
                    )
                    sample = {
                        'label_index': label_index,
                        'paths': spectrogram_paths
                    }
                    augmented_samples.append(sample)
                elif self.augment_technique['type'] == 'image':
                    augmented_image_path = get_augment_image_path(
                        image_path=train_spectrogram_path,
                        technique=self.augment_technique['label'],
                        dataset_type=self.dataset_type
                    )
                    if not augmented_image_path.exists():
                        self.augment_technique['apply'](
                            image_path=train_spectrogram_path,
                            target_path=augmented_image_path
                        )
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

        if self.undersample_ratios is not None:
            for class_name, ratio in self.undersample_ratios.items():
                class_idx = self.class_to_idx[class_name]

                target_mask = np.array(dataset.targets)[train_indices] == class_idx
                target_indices = train_indices[target_mask]

                n_keep = int(len(target_indices) * ratio)
                np.random.seed(self.seed)
                indices_to_keep = np.random.choice(target_indices, n_keep, replace=False)

                train_indices = np.concatenate([train_indices[~target_mask], indices_to_keep])
        
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
        if self.desired_distribution is None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.number_of_workers)
        
        train_indices = self.train_dataset.indices
        all_targets = np.array(self.dataset.targets)
        train_targets = all_targets[train_indices]
        
        class_counts = Counter(train_targets)
        total_samples = len(train_targets)

        sample_weights = []

        assert self.desired_distribution is not None, "Illegal state: undersampling was selected but no desired distribution supplied"
        try:
            desired_distribution = {self.class_to_idx[cls]: self.desired_distribution[cls] for cls in self.desired_distribution.keys()}
        except KeyError:
            raise MisconfigurationException("Desired distribution parameter contains invalid class")

        for target in train_targets:
            current_proportion = class_counts[target] / total_samples
            desired_proportion = desired_distribution[target]
            weight = desired_proportion / current_proportion
            sample_weights.append(weight)
        
        generator = torch.Generator().manual_seed(self.seed)
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),  # Sample with replacement to match original dataset size
            replacement=True,
            generator=generator
        )
        
        dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            sampler=sampler,  # Use our custom sampler instead of shuffle=True
            num_workers=self.number_of_workers
        )

        return dataloader

    def count_classes(self, dataloader):
        counts = {}
        for _, labels in dataloader:
            for label_tensor in labels:
                label = label_tensor.item()
                counts[label] = (counts[label] if label in counts else 0) + 1
        return counts

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.number_of_workers)

