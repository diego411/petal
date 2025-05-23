from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch import Tensor, as_tensor, long, save, load
import os
from typing import Tuple
from ml.data.vision import load_and_transform

class SpectrogramDataset(Dataset):
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        transforms,
        cache_file: str 
    ) -> None:
        super().__init__()

        self.dataframe = dataframe
        self.transforms = transforms
        self.cache_file = cache_file

        self.class_to_idx = {}
        self.idx_to_class = {}

        for label_index in dataframe['label_index'].unique():
            unique_labels = dataframe[dataframe['label_index'] == label_index]['label'].unique()
            assert len(unique_labels) == 1
            self.idx_to_class[label_index] = unique_labels[0]

        for label in dataframe['label'].unique():
            unique_indices = dataframe[dataframe['label'] == label]['label_index'].unique()
            assert len(unique_indices) == 1
            self.class_to_idx[label] = unique_indices[0]

        # Load from cache if available
        # TODO: if different transform is passed the same cache is used. if dataset changes at all cache is not updated
        if os.path.exists(self.cache_file):
            print("[Datamodule] Loading precomputed tensors from cache...")
            self.items, self.targets, self.classes = load(self.cache_file, weights_only=False)
            print("\033[32m[Datamodule] Finished loading from cache\033[0m")
        else:
            print("[Datamodule] No cache found, processing dataset...")
            self.items, self.targets, self.classes = self._build_items()
            save((self.items, self.targets, self.classes), self.cache_file)
            print("\033[32m[Datamodule] Cached dataset saved for future runs\033[0m")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        return self.items[index]

    def _build_items(self):
        items = [None] * len(self)
        targets = []
        classes = self.dataframe['label'].unique()
        
        for index in range(len(self.dataframe)):
            item = self.dataframe.iloc[index].to_dict()
            target = as_tensor(item['label_index'], dtype=long)

            spectrogram_image = load_and_transform(item['spectrogram_path'], self.transforms)
            delta_spectrogram_image = load_and_transform(item['delta_spectrogram_path'], self.transforms)
            delta_delta_spectrogram_image = load_and_transform(item['delta_delta_spectrogram_path'], self.transforms)

            items[index] = (spectrogram_image, delta_spectrogram_image, delta_delta_spectrogram_image), target
            targets.append(target)

        return items, targets, classes

