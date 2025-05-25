import torch
import torchaudio
import torchvggish.vggish_input as vggish_input
from torch.utils.data import Dataset 
from ml.data.data_util import get_audios
from typing import List
from ml.data.modules.PetalDataModule import PetalDataModule

class VGGishDataset(Dataset):
    def __init__(self, items: List[dict], max_patches: int = 1):
        self.items = items
        self.targets = list(
            map(lambda item: item['label_idx'], items)
        )

        self.idx_to_class= {}
        for item in items:
            label_idx = item['label_idx']
            class_name = item['class_name']
            if label_idx in self.idx_to_class:
                assert self.idx_to_class[label_idx] == class_name, "Multiple classes supplied for same index!" 
            
            self.idx_to_class[label_idx] = class_name

        self.max_patches = max_patches

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        label_idx = item['label_idx']
        waveform = item['waveform']
        sr = item['sample_rate']

        # Convert to log-mel spectrogram (VGGish format)
        try:
            features = vggish_input.waveform_to_examples(waveform.numpy(), sr)
        except Exception as e:
            print(e)
        if isinstance(features, torch.Tensor):
            features = features.clone().detach().to(torch.float32)
        else:
            # If it's a numpy array
            features = torch.from_numpy(features).float()
        #features = torch.tensor(features, dtype=torch.float32)
        
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


class VGGishDataModule(PetalDataModule):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        print("[Datamodule] Using VGGishDataModule")
        self.idx_to_class = None
    
    def create_dataset(self):
        dataset = VGGishDataset(get_audios(self.dataset_type, self.binary))
        self.idx_to_class = dataset.idx_to_class
        return dataset
        