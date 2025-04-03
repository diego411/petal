from ml.data.data_util import create_spectrogram_images
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from collections import Counter
from ml.data.sets.SpectrogramDataset import SpectrogramDataset
from ml.data.modules.PetalDataModule import PetalDataModule


class DeltaVisionDataModule(PetalDataModule):

    def __init__(
        self,
        pretrained_model_name:str='resnet18', # TODO: link this param to model
        cache_file: str = 'dataset_cache.pth',
        *args,
        **kwargs 
    ):
        super().__init__(*args, **kwargs)
        print("[Datamodule] Using DeltaVisionDataModule")
        
        self.cache_file = cache_file
        timm_model = timm.create_model(pretrained_model_name, pretrained=True) 
        self.transform = create_transform(**resolve_data_config(
            pretrained_cfg=timm_model.pretrained_cfg,
            model=timm_model
        ))

    def create_dataset(self) -> SpectrogramDataset:
        _, _, _, dataframe = create_spectrogram_images(
            self.dataset_type,
            self.binary,
            self.verbose,
            self.spectrogram_backend
        )
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
