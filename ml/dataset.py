from src.controller.dropbox_controller import create_dropbox_client, download_folder
from src.AppConfig import AppConfig
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from itertools import chain
import torch
import torchaudio
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, random_split
from typing import Tuple


def download_data():
    dropbox = create_dropbox_client(
        app_key='pn2te7ax4wbcup1',#AppConfig.DROPBOX_APP_KEY,
        app_secret='8kq1tkj1pmlfu2m',#AppConfig.DROPBOX_APP_SECRET,
        refresh_token='s6KeP8r6zmgAAAAAAAAAAdeLYqqtG6fcoc51djEyt27vJat8_ikeAc8PE1BJgzUu'#AppConfig.DROPBOX_REFRESH_TOKEN
    )

    download_folder(
        dbx=dropbox,
        dropbox_folder='/EmotionExperiment/labeled/sad'
    )


def load_audio(path: str, label: str):
    dataset = []

    walker_wav = sorted(str(p) for p in Path(path).glob(f'*.wav'))
    walker_mp3 = sorted(str(p) for p in Path(path).glob(f'*.mp3'))

    for i, file_path in enumerate(list(chain(walker_wav, walker_mp3))):
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append([waveform, sample_rate, label])

    return dataset


def create_spectrogram_images() -> str:
    ekman_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
    all_emotions = ekman_emotions + ['neutral']

    spectrogram_path = '../data/spectrograms'
    if not os.path.isdir(spectrogram_path):
        os.makedirs(spectrogram_path, mode=0o777, exist_ok=True)

    for emotion in all_emotions:
        print(f"Starting to generate spectrograms for emotion: {emotion}")
        path = f'../data/audio/{emotion}'
        walker_wav = sorted(str(p) for p in Path(path).glob(f'*.wav'))
        walker_mp3 = sorted(str(p) for p in Path(path).glob(f'*.mp3'))

        emotion_path = f'{spectrogram_path}/{emotion}'

        if not os.path.isdir(emotion_path):
            os.makedirs(emotion_path, mode=0o777, exist_ok=True)
        elif os.listdir(emotion_path) != 0:
            print(f"Skipping generation of spectrograms for emotion: {emotion} since the directory is not empty!")
            continue

        for i, file_path in enumerate(list(chain(walker_wav, walker_mp3))):
            waveform, _ = torchaudio.load(file_path)
            waveform = waveform + 1e-9

            # create transformed waveforms
            waveform_length = waveform.shape[-1]
            n_fft = 2 ** (waveform_length - 1).bit_length() // 2
            n_fft = max(n_fft, 8)  # Ensure n_fft is at least 8

            try:
                spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)
                spectrogram_tensor = spectrogram_transform(waveform)
                spectrogram_numpy = spectrogram_tensor.log2()[0, :, :].numpy().T
                infinity_filter = spectrogram_numpy != np.NINF
                filtered_spectrogram = np.where(
                    infinity_filter,
                    spectrogram_numpy,
                    sys.float_info.min
                )  # replace remaining -inf with smallest float

                plt.figure()
                plt.imsave(f'{emotion_path}/spec_img{i}.png', filtered_spectrogram, cmap='viridis', origin='lower')
                plt.close()
            except Exception:
                print(f"Failed to generate spectrogram for emotion: {emotion} at index {i}")
    return spectrogram_path


def get_image_dataset() -> ImageFolder:
    spectrogram_path = create_spectrogram_images()
    image_folder: ImageFolder = ImageFolder(
        root=spectrogram_path,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # transforms.Resize((224,224))
            transforms.ToTensor()
        ])
    )
    return image_folder


def create_data_split(dataset: ImageFolder) -> Tuple[Subset, Subset, Subset]:
    # split data to test and train
    # use 80% to train
    train_size = int(0.7 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size

    generator = torch.Generator().manual_seed(42)
    train, test, validation = random_split(
        dataset,
        [train_size, test_size, validation_size],
        generator=generator
    )
    return train, test, validation


def get_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    spectrogram_dataset: ImageFolder = get_image_dataset()

    train_dataset, test_dataset, validation_dataset = create_data_split(spectrogram_dataset)

    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=2,
        shuffle=True
    )

    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=2,
        shuffle=True
    )

    validation_dataloader: DataLoader = DataLoader(
        validation_dataset,
        batch_size=16,
        num_workers=2
    )

    return train_dataloader, test_dataloader, validation_dataloader


if __name__ == '__main__':
    download_data()
