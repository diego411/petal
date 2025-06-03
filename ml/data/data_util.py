from src.controller.dropbox_controller import create_dropbox_client, download_folder
from src.AppConfig import AppConfig
from pathlib import Path
import numpy as np
import pandas as pd
import os
import sys
from itertools import chain
import torchaudio
from torchaudio.transforms._transforms import Spectrogram, MelSpectrogram
import torchaudio.transforms as T
from torchaudio.functional import compute_deltas
from typing import Tuple, List, Optional, Dict
from torch import Tensor
from pydub import AudioSegment
import librosa
from ml.data.vision import show_and_save_spectrogram_image, show_and_save_mel_spectrogram, show_and_save_spectrogram_librosa
from src.utils.hash import hash_file_name


BASE_DATA_PATH = Path.home() / '.data/petal/'
ekman_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']


def get_emotions(dataset_type: str) -> List[str]:
    emotions = ekman_emotions
    if dataset_type != 'post-labeled':
        emotions = ekman_emotions + ['neutral']
    
    return emotions


def get_audio_path(dataset_type: str) -> Path:
    return BASE_DATA_PATH / dataset_type / ('split-audio' if dataset_type == 'pre-labeled' else 'audio')

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

def get_audio_files(dataset_type: str, emotion: str) -> List[Path]:
    path = BASE_DATA_PATH / dataset_type / 'audio' / emotion
    walker_wav = sorted(p for p in Path(path).glob(f'*.wav'))
    walker_mp3 = sorted(p for p in Path(path).glob(f'*.mp3'))
    
    return list(chain(walker_wav, walker_mp3))

def get_number_of_fourier_transform_bins(waveform: Tensor | np.ndarray) -> int:
    waveform_length = waveform.shape[-1]
    n_fft = 2 ** (waveform_length - 1).bit_length() // 2
    return max(n_fft, 8)  # Ensure n_fft is at least 8

def filter_spectrogram(spectrogram_tensor: Tensor) -> np.ndarray:
    if (spectrogram_tensor.min() < 0).item():
        spectrogram_tensor = spectrogram_tensor.abs().add(1e-6)

    spectrogram_numpy: np.ndarray = spectrogram_tensor.log2()[0, :, :].numpy().T
    
    infinity_filter = spectrogram_numpy != -np.inf
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

def create_delta_spectrogram(spectrogram: Tensor) -> np.ndarray:
    delta = compute_deltas(spectrogram)
    return filter_spectrogram(delta)

def create_mel_spectrogram(waveform, sample_rate):
    mel_spectrogram_transform = create_mel_spectrogram_transform(waveform, sample_rate)
    mel_spectrogram_tensor: Tensor = mel_spectrogram_transform(waveform)
    return filter_spectrogram(
        mel_spectrogram_tensor
    )

def create_spectrogram_librosa(waveform: np.ndarray, n_fft, hop_length) -> np.ndarray:
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)

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
            if emotion == 'skip':
                cursor = time_stamp
                continue
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

def split_pre_labeled_audios():
    pre_labeled_audio_path = BASE_DATA_PATH / 'pre-labeled' / 'audio'
    split_pre_labeled_audio_path = BASE_DATA_PATH / 'pre-labeled' / 'split-audio'
    emotion_dirs = list(filter(
        lambda path: path.is_dir(),
        pre_labeled_audio_path.iterdir()
    ))
    
    for emotion_dir in emotion_dirs:
        split_emotion_path = split_pre_labeled_audio_path / emotion_dir.stem
        if not split_emotion_path.exists():
            os.makedirs(split_emotion_path, mode=0o777, exist_ok=True)
        for audio_path in emotion_dir.iterdir():
            try:
                audio = AudioSegment.from_file(audio_path)
            except Exception as e:
                print(e)
                continue

            segment_length = 3000
            cursor = 0
            index = 0
            while True:
                end = cursor + segment_length
                stop = False
                if end > len(audio):
                    end = len(audio)
                    stop = True
                
                segment = audio[cursor:end]
                segment.export(split_emotion_path / f'{audio_path.stem}_{index}.wav', format='wav')
                if stop:
                    break
                index = index + 1
                cursor = end

def generate_spectrogram_for(
    audio_path: Path,
    spectrogram_type: str,
    spectrogram_backend: str,
    with_deltas: bool,
    target_path: Path
) -> Dict[str, Path]:
    paths = {}
    spectrogram_image_path = None 
    mel_spectrogram_image_path = None
    delta_spectrogram_image_path = None
    delta_delta_spectrogram_image_path = None

    file_name = f'{hash_file_name(audio_path.stem)}.png'

    if spectrogram_type == 'spectrogram':
        spectrogram_path = target_path / 'spectrogram' 
        if not spectrogram_path.exists():
            os.makedirs(spectrogram_path, mode=0o777, exist_ok=True)
        spectrogram_image_path = spectrogram_path / file_name
        paths['spectrogram_path'] = spectrogram_image_path
    
    if spectrogram_type == 'mel-spectrogram':
        mel_spectrogram_path = target_path / 'mel-spectrogram'
        if not mel_spectrogram_path.exists():
            os.makedirs(mel_spectrogram_path, mode=0o777, exist_ok=True)
        mel_spectrogram_image_path = mel_spectrogram_path / file_name
        paths['mel_spectrogram_path'] = mel_spectrogram_image_path

    if with_deltas:
        delta_spectrogram_path = target_path / 'delta-spectrogram'
        delta_delta_spectrogram_path = target_path / 'delta-delta-spectrogram'
        if not delta_spectrogram_path.exists():
            os.makedirs(delta_spectrogram_path, mode=0o777, exist_ok=True)
        if not delta_delta_spectrogram_path.exists():
            os.makedirs(delta_delta_spectrogram_path, mode=0o777, exist_ok=True)
        delta_spectrogram_image_path = delta_spectrogram_path / file_name
        delta_delta_spectrogram_image_path = delta_delta_spectrogram_path / file_name
        paths['delta_spectrogram_path'] = delta_spectrogram_image_path
        paths['delta_delta_spectrogram_path'] = delta_delta_spectrogram_image_path

    if spectrogram_backend == 'torch':
        compute_spectrograms_torch(
            file_path=audio_path,
            spectrogram_type=spectrogram_type,
            with_deltas=with_deltas,
            spectrogram_image_path=spectrogram_image_path,
            delta_spectrogram_image_path=delta_spectrogram_image_path,
            delta_delta_spectrogram_image_path=delta_delta_spectrogram_image_path,
            mel_spectrogram_image_path=mel_spectrogram_image_path
        )
    elif spectrogram_backend == 'librosa':
        compute_spectrograms_librosa(
            file_path=audio_path,
            spectrogram_type=spectrogram_type,
            with_deltas=with_deltas,
            spectrogram_image_path=spectrogram_path
        )

    return paths 


def compute_spectrograms_torch(
    file_path: Path,
    spectrogram_type: str,
    with_deltas: bool,
    spectrogram_image_path: Optional[Path],
    delta_spectrogram_image_path: Optional[Path],
    delta_delta_spectrogram_image_path: Optional[Path],
    mel_spectrogram_image_path: Optional[Path]
):
    torch_should_generate_spectrograms = spectrogram_type == 'spectrogram' and spectrogram_image_path is not None and not spectrogram_image_path.exists() 
    torch_should_generate_mel_spectrograms = spectrogram_type == 'mel-spectrogram' and mel_spectrogram_image_path is not None and not mel_spectrogram_image_path.exists()
    torch_should_generate_deltas = spectrogram_type == 'spectrogram' and with_deltas and ((delta_spectrogram_image_path is not None and not delta_spectrogram_image_path.exists()) or (delta_delta_spectrogram_image_path is not None and delta_delta_spectrogram_image_path.exists()))

    torch_should_generate = torch_should_generate_spectrograms or torch_should_generate_mel_spectrograms or torch_should_generate_deltas

    if not torch_should_generate:
        return

    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform + 1e-9
    n_fft = get_number_of_fourier_transform_bins(waveform)

    if torch_should_generate_mel_spectrograms:
        mel_spectrogram: np.ndarray = create_mel_spectrogram(waveform, sample_rate)
        show_and_save_mel_spectrogram(
            mel_spectrogram=mel_spectrogram,
            n_fft=n_fft,
            sample_rate=sample_rate,
            path=mel_spectrogram_image_path
        )

    if torch_should_generate_spectrograms or torch_should_generate_deltas:
        spectrogram_transform: Spectrogram = create_spectrogram_transform(waveform)
        spectrogram_tensor: Tensor = spectrogram_transform(waveform)
        filtered_spectrogram: np.ndarray = filter_spectrogram(spectrogram_tensor) 

    if torch_should_generate_spectrograms:
        show_and_save_spectrogram_image(
            spectrogram=filtered_spectrogram,
            n_fft=n_fft,
            sample_rate=sample_rate,
            path=spectrogram_image_path
        )

    if torch_should_generate_deltas:
        delta_spectrogram: Tensor = compute_deltas(spectrogram_tensor)
        filtered_delta_spectrogram: np.ndarray = filter_spectrogram(delta_spectrogram)

        if delta_spectrogram_image_path is not None and (not delta_spectrogram_image_path.exists()):
            show_and_save_spectrogram_image(
                spectrogram=filtered_delta_spectrogram,
                n_fft=n_fft,
                sample_rate=sample_rate,
                path=delta_spectrogram_image_path
            )

        if delta_delta_spectrogram_image_path is not None and (not delta_delta_spectrogram_image_path.exists()):
            delta_delta_spectrogram: Tensor = compute_deltas(delta_spectrogram)
            filtered_delta_delta_spectrogram: np.ndarray = filter_spectrogram(delta_delta_spectrogram)

            show_and_save_spectrogram_image(
                spectrogram=filtered_delta_delta_spectrogram,
                n_fft=n_fft,
                sample_rate=sample_rate,
                path=delta_delta_spectrogram_image_path
            )

def compute_spectrograms_librosa(
    file_path: Path,
    spectrogram_type: str,
    with_deltas: bool,
    spectrogram_image_path: Optional[Path]
):
    if spectrogram_type == 'mel-spectrogram':
        raise NotImplementedError("Mel spectrograms not yet supported for spectrogram backend librosa")
    if with_deltas:
        raise NotImplementedError("Deltas not yet supported for spectrogram backend librosa")
    
    librosa_should_generate_spectrograms = spectrogram_type == 'spectrogram' and (spectrogram_image_path is not None and not spectrogram_image_path.exists())
    librosa_should_generate = librosa_should_generate_spectrograms

    if not librosa_should_generate:
        return

    waveform, sample_rate = librosa.load(file_path, sr=None)
    n_fft = get_number_of_fourier_transform_bins(waveform)
    hop_length = n_fft // 4

    if spectrogram_image_path is not None and not spectrogram_image_path.exists():
        spectrogram = create_spectrogram_librosa(
            waveform=waveform,
            n_fft=n_fft,
            hop_length=hop_length
        )

        show_and_save_spectrogram_librosa(
            spectrogram=spectrogram,
            sample_rate=sample_rate,
            hop_length=hop_length,
            path=spectrogram_image_path
        )

def create_spectrogram_images(
    dataset_type: str,
    spectrogram_type: str,
    with_deltas: bool,
    binary: bool,
    verbose: bool,
    spectrogram_backend: str
) -> Tuple[Path, Path, Path, pd.DataFrame]:
    download_data(verbose)
    ekman_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']

    all_emotions = ekman_emotions
    if dataset_type != 'post-labeled':
        all_emotions = ekman_emotions + ['neutral']

    if dataset_type == 'post-labeled':
        label_by_video_emotions(verbose)
    
    if dataset_type == 'pre-labeled':
        split_pre_labeled_audios()

    type_path = Path(dataset_type) / 'binary' if binary else dataset_type

    spectrogram_path = BASE_DATA_PATH / type_path / 'spectrograms'
    delta_spectrogram_path = BASE_DATA_PATH / type_path / 'delta-spectrograms'
    delta_delta_spectrogram_path = BASE_DATA_PATH / type_path / 'delta-delta-spectrograms'
    mel_spectrogram_path = BASE_DATA_PATH / type_path / 'mel-spectrograms'
    librosa_spectrogram_path = BASE_DATA_PATH / type_path / 'librosa-spectrograms'
    
    if not os.path.isdir(spectrogram_path):
        os.makedirs(spectrogram_path, mode=0o777, exist_ok=True)

    if not os.path.isdir(mel_spectrogram_path):
        os.makedirs(mel_spectrogram_path, mode=0o777, exist_ok=True)
    
    df = pd.DataFrame(columns=[
        'label_index',
        'label',
        'spectrogram_path',
        'delta_spectrogram_path',
        'delta_delta_spectrogram_path'
    ])
    df_index = 0

    for emotion_index, emotion in enumerate(all_emotions):
        print(f"[Datamodule] Starting to generate spectrograms for emotion: {emotion}")

        path = get_audio_path(dataset_type) / emotion
        walker_wav  = sorted(p for p in Path(path).glob(f'*.wav'))
        walker_mp3 = sorted(p for p in Path(path).glob(f'*.mp3'))

        class_path = emotion
        if binary:
            if emotion != 'neutral':
                class_path = 'not-neutral'
    
        spectrogram_emotion_path = spectrogram_path / class_path 
        delta_spectrogram_emotion_path = delta_spectrogram_path / class_path
        delta_delta_spectrogram_emotion_path = delta_delta_spectrogram_path / class_path
        mel_spectrogram_emotion_path = mel_spectrogram_path / class_path
        librosa_spectrogram_emotion_path = librosa_spectrogram_path / class_path

        if not os.path.isdir(spectrogram_emotion_path):
            os.makedirs(spectrogram_emotion_path, mode=0o777, exist_ok=True)

        if not os.path.isdir(delta_spectrogram_emotion_path):
            os.makedirs(delta_spectrogram_emotion_path, mode=0o777, exist_ok=True)
            
        if not os.path.isdir(delta_delta_spectrogram_emotion_path):
            os.makedirs(delta_delta_spectrogram_emotion_path, mode=0o777, exist_ok=True)
            
        if not os.path.isdir(mel_spectrogram_emotion_path):
            os.makedirs(mel_spectrogram_emotion_path, mode=0o777, exist_ok=True)

        if not os.path.isdir(librosa_spectrogram_emotion_path):
            os.makedirs(librosa_spectrogram_emotion_path, mode=0o777, exist_ok=True)

        for i, file_path in enumerate(list(chain(walker_wav, walker_mp3))):
            spectrogram_file_name = f'{file_path.stem}.png' 
            spectrogram_image_path = spectrogram_emotion_path / spectrogram_file_name
            delta_spectrogram_image_path = delta_spectrogram_emotion_path / spectrogram_file_name 
            delta_delta_spectrogram_image_path = delta_delta_spectrogram_emotion_path / spectrogram_file_name 
            mel_spectrogram_image_path = mel_spectrogram_emotion_path / spectrogram_file_name
            librosa_spectrogram_image_path = librosa_spectrogram_emotion_path / spectrogram_file_name

            if spectrogram_image_path.exists() and mel_spectrogram_image_path.exists() and spectrogram_backend == 'torch':
                continue
            elif librosa_spectrogram_image_path.exists() and spectrogram_backend == 'librosa':
                continue

            success = True
            try:
                if spectrogram_backend == 'torch':
                    compute_spectrograms_torch(
                        file_path=file_path,
                        spectrogram_type=spectrogram_type,
                        with_deltas=with_deltas,
                        spectrogram_image_path=spectrogram_image_path,
                        delta_spectrogram_image_path=delta_spectrogram_image_path,
                        delta_delta_spectrogram_image_path=delta_delta_spectrogram_image_path,
                        mel_spectrogram_image_path=mel_spectrogram_image_path
                    )
                elif spectrogram_backend == 'librosa':
                    compute_spectrograms_librosa(
                        file_path=file_path,
                        spectrogram_type=spectrogram_type,
                        with_deltas=with_deltas,
                        spectrogram_image_path=librosa_spectrogram_image_path
                    )
                else:
                    raise RuntimeError('Unexpected spectrogram backend')
            except Exception:
                success = False
                print(f"\033[31m[Datamodule] Failed to generate spectrogram for emotion: {emotion} at index {i}\033[0m")
            
            if success and with_deltas:
                df.loc[df_index] = pd.Series({
                    'label_index': emotion_index,
                    'label': emotion,
                    'spectrogram_path': spectrogram_image_path,
                    'delta_spectrogram_path': delta_spectrogram_image_path,
                    'delta_delta_spectrogram_path': delta_delta_spectrogram_image_path
                })
                df_index = df_index + 1
    print('\033[32m[Datamodule] Finished spectrogram generation\033[0m')
    return spectrogram_path, mel_spectrogram_path, librosa_spectrogram_path, df

def get_audios(dataset_type: str, binary: bool) -> List[dict]:
    audios = []
    emotions = get_emotions(dataset_type)

    for i, emotion in enumerate(emotions):
        for file_path in get_audio_files(dataset_type, emotion):
            label_index = i
            class_name = emotion
            if binary:
                label_index = 0 if emotion == 'neutral' else 1
                class_name = 'neutral' if emotion == 'neutral' else 'not-neutral'

            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.mean(dim=0)  # Convert stereo to mono if needed

            min_samples_for_vggish = int(sr * 0.96)

            if len(waveform) < min_samples_for_vggish:
                continue

            audios.append({
                'label_idx': label_index,
                'class_name': class_name,
                'waveform': waveform,
                'sample_rate': sr
            })

    return audios

if __name__ == '__main__':
    split_pre_labeled_audios()
