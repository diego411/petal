from src.controller.dropbox_controller import create_dropbox_client, download_folder
from src.AppConfig import AppConfig
from pathlib import Path
import numpy as np
import os
import sys
from itertools import chain
import torch
import torchaudio
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, random_split
from torchaudio.transforms._transforms import Spectrogram, MelSpectrogram
import torchaudio.transforms as T
from typing import Tuple, List
from torch import Tensor
from pydub import AudioSegment
from ml.vision import show_and_save_spectrogram_image

BASE_DATA_PATH = Path.home() / '.data/petal/'


def download_data():
    app_key = AppConfig.DROPBOX_APP_KEY
    app_secret = AppConfig.DROPBOX_APP_SECRET
    refresh_token = AppConfig.DROPBOX_REFRESH_TOKEN

    if app_key is None or app_secret is None or refresh_token is None:
        raise RuntimeError('Required dropbox environment variable is missing! Cannot download the data!')

    dropbox = create_dropbox_client(
        app_key=app_key,
        app_secret=app_secret,
        refresh_token=refresh_token
    )

    download_folder(
        dbx=dropbox,
        dropbox_folder=f'/EmotionExperiment/labeled',
        local_folder=BASE_DATA_PATH / 'pre-labeled/audio'
    )


    download_folder(
        dbx=dropbox,
        dropbox_folder=f'/EmotionExperiment/unlabeled',
        local_folder=BASE_DATA_PATH / 'post-labeled/unlabeled-audio'
    )


def load_audio(path: str, label: str):
    dataset = []

    walker_wav = sorted(str(p) for p in Path(path).glob(f'*.wav'))
    walker_mp3 = sorted(str(p) for p in Path(path).glob(f'*.mp3'))

    for i, file_path in enumerate(list(chain(walker_wav, walker_mp3))):
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append([waveform, sample_rate, label])

    return dataset

def get_number_of_fourier_transform_bins(waveform: Tensor) -> int:
    waveform_length = waveform.shape[-1]
    n_fft = 2 ** (waveform_length - 1).bit_length() // 2
    return max(n_fft, 8)  # Ensure n_fft is at least 8

def filter_spectrogram(spectrogram_tensor: Tensor) -> np.ndarray:
    spectrogram_numpy: np.ndarray = spectrogram_tensor.log2()[0, :, :].numpy().T
    
    infinity_filter = spectrogram_numpy != np.NINF
    filtered_spectrogram = np.where(
        infinity_filter,
        spectrogram_numpy,
        sys.float_info.min
    )  # replace remaining -inf with smallest float
    return filtered_spectrogram

def create_spectrogram_transform(waveform: Tensor) -> Spectrogram:
    n_fft = get_number_of_fourier_transform_bins(waveform)
    return T.Spectrogram(
        n_fft=n_fft,
        hop_length=n_fft // 4
    )

def create_mel_spectrogram_transform(waveform: Tensor, sample_rate: int) -> MelSpectrogram:
    n_fft = get_number_of_fourier_transform_bins(waveform)
    n_freqs = n_fft // 2 + 1
    n_mels = min(64, n_freqs)

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,         # Number of FFT bins
        hop_length=n_fft // 4,     # Step size between FFTs
        n_mels=n_mels          # Number of mel bands
    )

    return mel_spectrogram

def create_spectrogram(waveform: Tensor) -> np.ndarray:
    spectrogram_transform: Spectrogram = create_spectrogram_transform(waveform)
    spectrogram_tensor: Tensor = spectrogram_transform(waveform)
    return filter_spectrogram(
        spectrogram_tensor
    ) 


def create_mel_spectrogram(waveform, sample_rate):
    mel_spectrogram_transform = create_mel_spectrogram_transform(waveform, sample_rate)
    mel_spectrogram_tensor: Tensor = mel_spectrogram_transform(waveform)
    return filter_spectrogram(
        mel_spectrogram_tensor
    )

# Emotions occuring in the video
# key is the timestamp where the specific video clip ENDS
# value is the emotion of the clip
VIDEO_LABELS = {
    12000: 'happy', # puppies
    20000: 'happy', # kid with avocado
    59000: 'angry', # kid throwing a tantrum: this has multiple emotions!
    82000: 'happy', # runners supporting competitor over finish line
    120000: 'disgusted', # man eating a maggot
    155000: 'sad', # soldiers in battle: this has multiple emotions!
    207000: 'angry', # trump talking: this has multiple emotions imo!
    237000: 'surprised', # mountain biker riding down rock bridge: is this really surprise?
    265000: 'fearful', # person biking on top of a skyscraper
    283000: 'surprised', # runner almost falling off skyscraper
    298000: 'angry', # man beating raccoon: is this really anger?
    363000: 'sad', # social worker feeding starved toddler
    394000: 'sad', # residents collecting electronic waste in the slums of Accra
    407000: 'sad', # dog on the grave of his owner
    562000: 'fearful' # man discovering monster through his camera
}

def label_by_video_emotions():
    path = BASE_DATA_PATH / 'post-labeled' / 'unlabeled-audio'
    walker_wav: List[Path]  = sorted(p for p in Path(path).glob(f'*.wav'))

    for file_path in walker_wav:
        base_name = file_path.stem 
        try:
            recording = AudioSegment.from_wav(str(file_path))
        except Exception as e:
            print(f'Loading recording: {file_path} failed with following error: {e}')

        cursor: int = 0
        i: int = 0
        for time_stamp, emotion in VIDEO_LABELS.items():
            try:
                snippet = recording[cursor:time_stamp]
                directory = BASE_DATA_PATH / 'post-labeled' / 'audio' / emotion 
              
                if not os.path.exists(directory):
                   os.makedirs(directory)
                
                snippet.export(directory / f'{base_name}_{i}.wav', format='wav')
            except Exception as e:
                print(f'Failed to create or save snippet for recording: {file_path} with following error: {e}. Start of snippet: {cursor}. End of snippet: {time_stamp}')
            
            cursor = time_stamp
            i += 1


def create_spectrogram_images(dataset_type: str) -> Tuple[Path, Path]:
    ekman_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']

    all_emotions = ekman_emotions
    if dataset_type != 'post-labeled':
        all_emotions = ekman_emotions + ['neutral']

    if dataset_type == 'post-labeled':
        label_by_video_emotions()

    spectrogram_path = BASE_DATA_PATH / dataset_type / 'spectrograms'
    mel_spectrogram_path = BASE_DATA_PATH / dataset_type / 'mel-spectrograms'
    
    if not os.path.isdir(spectrogram_path):
        os.makedirs(spectrogram_path, mode=0o777, exist_ok=True)

    if not os.path.isdir(mel_spectrogram_path):
        os.makedirs(mel_spectrogram_path, mode=0o777, exist_ok=True)

    for emotion in all_emotions:
        print(f"Starting to generate spectrograms for emotion: {emotion}")
        path = BASE_DATA_PATH / dataset_type / 'audio' / emotion
        walker_wav  = sorted(str(p) for p in Path(path).glob(f'*.wav'))
        walker_mp3 = sorted(str(p) for p in Path(path).glob(f'*.mp3'))

        spectrogram_emotion_path = spectrogram_path / emotion
        mel_spectrogram_emotion_path = mel_spectrogram_path / emotion

        skip_spectrogram_generation = False
        skip_mel_spectrogram_generation = False

        if not os.path.isdir(spectrogram_emotion_path):
            os.makedirs(spectrogram_emotion_path, mode=0o777, exist_ok=True)
        elif os.listdir(spectrogram_emotion_path) != 0:
            print(f"Skipping generation of spectrograms for emotion: {emotion} since the directory is not empty!")
            skip_spectrogram_generation = True

        if not os.path.isdir(mel_spectrogram_emotion_path):
            os.makedirs(mel_spectrogram_emotion_path, mode=0o777, exist_ok=True)
        elif os.listdir(mel_spectrogram_emotion_path) != 0:
            print(f"Skipping generation of mel spectrograms for emotion: {emotion} since the directory is not empty!")
            skip_mel_spectrogram_generation = True

        if skip_spectrogram_generation and skip_mel_spectrogram_generation:
            continue

        for i, file_path in enumerate(list(chain(walker_wav, walker_mp3))):
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = waveform + 1e-9

            try:
                n_fft = get_number_of_fourier_transform_bins(waveform)

                if not skip_spectrogram_generation:
                    spectrogram: np.ndarray = create_spectrogram(waveform)
                    show_and_save_spectrogram_image(
                        spectrogram=spectrogram,
                        n_fft=n_fft,
                        sample_rate=sample_rate,
                        path=spectrogram_emotion_path / f'spec_img{i}.png'
                    )

                if not skip_mel_spectrogram_generation:
                    mel_spectrogram: np.ndarray = create_mel_spectrogram(waveform, sample_rate)
                    show_and_save_spectrogram_image(
                        spectrogram=mel_spectrogram,
                        n_fft=n_fft,
                        sample_rate=sample_rate,
                        path=mel_spectrogram_emotion_path / f'spec_img{i}.png'
                    )
            except Exception:
                print(f"Failed to generate spectrogram for emotion: {emotion} at index {i}")
    return spectrogram_path, mel_spectrogram_path


def get_image_dataset(dataset_type: str, spectrogram_type: str) -> ImageFolder:
    spectrogram_path, mel_spectrogram_path = create_spectrogram_images(dataset_type)

    if spectrogram_type == 'spectrogram':
        path = spectrogram_path
    elif spectrogram_type == 'mel-spectrogram':
        path = mel_spectrogram_path
    else:
        raise RuntimeError("Unexpected spectrogram type")
    
    image_folder: ImageFolder = ImageFolder(
        root=str(path),
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # transforms.Resize((224,224))
            transforms.ToTensor()
        ])
    )
    print(image_folder.class_to_idx)
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


def get_data_loaders(dataset_type: str, spectrogram_type: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    spectrogram_dataset: ImageFolder = get_image_dataset(dataset_type, spectrogram_type)

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
    label_by_video_emotions()
