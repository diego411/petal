from src.controller.dropbox_controller import create_dropbox_client, download_folder
from src.AppConfig import AppConfig
from pathlib import Path
import numpy as np
import os
import sys
from itertools import chain
import torchaudio
from torchaudio.transforms._transforms import Spectrogram, MelSpectrogram
import torchaudio.transforms as T
from typing import Tuple, List
from torch import Tensor
from pydub import AudioSegment
from ml.data.vision import show_and_save_spectrogram_image

BASE_DATA_PATH = Path.home() / '.data/petal/'


def download_data(verbose: bool):
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
        dropbox_folder=Path('/EmotionExperiment/labeled'),
        local_folder=BASE_DATA_PATH / 'pre-labeled/audio',
        verbose=verbose
    )

    download_folder(
        dbx=dropbox,
        dropbox_folder=Path('/EmotionExperiment/unlabeled'),
        local_folder=BASE_DATA_PATH / 'post-labeled/unlabeled-audio',
        verbose=verbose
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
    12000: 'happy', # puppies - 12sec.
    20000: 'happy', # kid with avocado - 8sec.
    59000: 'angry', # kid throwing a tantrum: this has multiple emotions! - 39sec.
    82000: 'happy', # runners supporting competitor over finish line - 23sec.
    120000: 'disgusted', # man eating a maggot - 38sec.
    155000: 'sad', # soldiers in battle: this has multiple emotions! - 35sec.
    207000: 'angry', # trump talking: this has multiple emotions imo! - 52sec.
    237000: 'surprised', # mountain biker riding down rock bridge: is this really surprise? - 30sec.
    265000: 'fearful', # person biking on top of a skyscraper - 28sec.
    283000: 'surprised', # runner almost falling off skyscraper - 18sec.
    298000: 'angry', # man beating raccoon: is this really anger? - 15sec.
    363000: 'sad', # social worker feeding starved toddler - 65sec.
    394000: 'sad', # residents collecting electronic waste in the slums of Accra - 31sec.
    407000: 'sad', # dog on the grave of his owner - 13sec.
    562000: 'fearful' # man discovering monster through his camera - 155sec.
}

def label_by_video_emotions(verbose: bool):
    path = BASE_DATA_PATH / 'post-labeled' / 'unlabeled-audio'
    walker_wav: List[Path]  = sorted(p for p in Path(path).glob(f'*.wav'))

    for file_path in walker_wav:
        base_name = file_path.stem 
        try:
            recording = AudioSegment.from_wav(str(file_path))
        except Exception as e:
            print(f'\033[31m[Datamodule] Loading recording: {file_path} failed with following error: {e}\033[0m')

        cursor: int = 0
        i: int = 0
        for time_stamp, emotion in VIDEO_LABELS.items():
            try:
                directory = BASE_DATA_PATH / 'post-labeled' / 'audio' / emotion 
              
                if not os.path.exists(directory):
                   os.makedirs(directory)
                
                video_length = time_stamp - cursor
                num_snippets = video_length // 10000
                if num_snippets == 0:
                    num_snippets = 1
                    snippet_length = video_length
                else:
                    snippet_length = video_length / num_snippets

                start = cursor
                for snippet_index in range(0, num_snippets):
                    snippet_path = directory / f'{base_name}_{i}_{snippet_index}.wav' 
                    if snippet_path.exists():
                        if verbose:
                            print(f'\033[33m[Datamodule] Skipping snippet with emotion {emotion} for recording: {base_name}. Video index: {i}, snippet index: {snippet_index}\033[0m')
                        continue

                    end = start + snippet_length
                    snippet = recording[start:end]
                    snippet.export(snippet_path, format='wav')
                    if verbose:
                        print(f'\033[32m[Datamodule] Exporting snippet with emotion {emotion} for recording: {base_name}. Video index: {i}, snippet index: {snippet_index}\033[0m')
                    start = end
            except Exception as e:
                print(f'\033[31m[Datamodule] Failed to create or save snippet for recording: {file_path} with following error: {e}. Start of snippet: {cursor}. End of snippet: {time_stamp}\033[0m')
            
            cursor = time_stamp
            i += 1


def create_spectrogram_images(dataset_type: str, binary: bool, verbose: bool) -> Tuple[Path, Path]:
    download_data(verbose)
    ekman_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']

    all_emotions = ekman_emotions
    if dataset_type != 'post-labeled':
        all_emotions = ekman_emotions + ['neutral']

    if dataset_type == 'post-labeled':
        label_by_video_emotions(verbose)

    type_path = Path(dataset_type) / 'binary' if binary else dataset_type
    spectrogram_path = BASE_DATA_PATH / type_path / 'spectrograms'
    mel_spectrogram_path = BASE_DATA_PATH / type_path / 'mel-spectrograms'
    
    if not os.path.isdir(spectrogram_path):
        os.makedirs(spectrogram_path, mode=0o777, exist_ok=True)

    if not os.path.isdir(mel_spectrogram_path):
        os.makedirs(mel_spectrogram_path, mode=0o777, exist_ok=True)

    for emotion in all_emotions:
        print(f"[Datamodule] Starting to generate spectrograms for emotion: {emotion}")
        path = BASE_DATA_PATH / dataset_type / 'audio' / emotion
        walker_wav  = sorted(p for p in Path(path).glob(f'*.wav'))
        walker_mp3 = sorted(p for p in Path(path).glob(f'*.mp3'))

        class_path = emotion
        if binary:
            if emotion != 'neutral':
                class_path = 'not-neutral'
    
        spectrogram_emotion_path = spectrogram_path / class_path 
        mel_spectrogram_emotion_path = mel_spectrogram_path / class_path

        if not os.path.isdir(spectrogram_emotion_path):
            os.makedirs(spectrogram_emotion_path, mode=0o777, exist_ok=True)

        if not os.path.isdir(mel_spectrogram_emotion_path):
            os.makedirs(mel_spectrogram_emotion_path, mode=0o777, exist_ok=True)

        for i, file_path in enumerate(list(chain(walker_wav, walker_mp3))):
            spectrogram_image_path = spectrogram_emotion_path / f'{file_path.stem}.png' 
            mel_spectrogram_image_path = mel_spectrogram_emotion_path / f'{file_path.stem}.png' 

            if spectrogram_image_path.exists() and mel_spectrogram_image_path.exists():
                continue

            try:
                waveform, sample_rate = torchaudio.load(file_path)
                waveform = waveform + 1e-9
                n_fft = get_number_of_fourier_transform_bins(waveform)
                if not spectrogram_image_path.exists():
                    spectrogram: np.ndarray = create_spectrogram(waveform)
                    show_and_save_spectrogram_image(
                        spectrogram=spectrogram,
                        n_fft=n_fft,
                        sample_rate=sample_rate,
                        path=spectrogram_image_path
                    )

                if not mel_spectrogram_image_path.exists():
                    mel_spectrogram: np.ndarray = create_mel_spectrogram(waveform, sample_rate)
                    show_and_save_spectrogram_image(
                        spectrogram=mel_spectrogram,
                        n_fft=n_fft,
                        sample_rate=sample_rate,
                        path=mel_spectrogram_image_path
                    )
            except Exception:
                print(f"\033[31m[Datamodule] Failed to generate spectrogram for emotion: {emotion} at index {i}\033[0m")
    return spectrogram_path, mel_spectrogram_path


if __name__ == '__main__':
    label_by_video_emotions(verbose=True)
