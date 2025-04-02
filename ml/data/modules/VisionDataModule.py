
from ml.data.modules.PetalDataModule import PetalDataModule
from ml.data.data_util import create_spectrogram_images
from torchvision.datasets import ImageFolder
from collections import Counter
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform


class VisionDataModule(PetalDataModule):

    def __init__(self, pretrained_model_name: str, *args, **kwargs): # TODO: link pretrained_model_name param to model
        super().__init__(*args, **kwargs)
        print("[Datamodule] Using VisionDataModule")

        timm_model = timm.create_model(pretrained_model_name, pretrained=True) 
        self.transform = create_transform(**resolve_data_config(
            pretrained_cfg=timm_model.pretrained_cfg,
            model=timm_model
        ))

    def create_dataset(self):
        spectrogram_path, mel_spectrogram_path, _ = create_spectrogram_images(self.dataset_type, self.binary, self.verbose)

        if self.spectrogram_type == 'spectrogram':
            path = spectrogram_path
        elif self.spectrogram_type == 'mel-spectrogram':
            path = mel_spectrogram_path
        else:
            raise RuntimeError("Unexpected spectrogram type")
        
        image_folder: ImageFolder = ImageFolder(
            root=str(path),
            transform=self.transform
        )
        
        self.class_to_idx = image_folder.class_to_idx
        self.idx_to_class = {v: k for k, v in image_folder.class_to_idx.items()}

        class_counts = {image_folder.classes[i]: count for i, count in Counter(image_folder.targets).items()}

        print("[Datamodule] Number of samples per class in whole dataset:", class_counts)

        return image_folder
