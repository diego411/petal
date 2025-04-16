import numpy as np
import librosa
from typing import Tuple
from pathlib import Path
import os
import soundfile as sf
from pydub import AudioSegment
import os
from PIL import Image
import random
from ml.data.data_util import BASE_DATA_PATH



def get_augment_audio_path(audio_path: Path, technique: str, dataset_type: str) -> Path:
    AUGMENT_AUDIO_PATH = BASE_DATA_PATH / dataset_type / 'augmented-audios'
    if not AUGMENT_AUDIO_PATH.exists():
        os.makedirs(AUGMENT_AUDIO_PATH)

    return AUGMENT_AUDIO_PATH / f'{technique}_{audio_path.name}'

def get_augment_image_path(image_path: Path, technique: str, dataset_type: str) -> Path:
    AUGMENT_IMAGE_PATH = BASE_DATA_PATH / dataset_type / 'augmented-images'
    if not AUGMENT_IMAGE_PATH.exists():
        os.makedirs(AUGMENT_IMAGE_PATH)
    
    return AUGMENT_IMAGE_PATH / f'{technique}_{image_path.name}'

def random_audio_segment(audio_path) -> Tuple[np.ndarray, int | float]:
    audio, sample_rate = librosa.load(audio_path, sr=None)
    total_samples = len(audio)
    segment_samples = int(total_samples / 3)
    
    start_idx = np.random.randint(0, total_samples - segment_samples)
    return audio[start_idx : start_idx + segment_samples], sample_rate

def apply_pitch_shift(audio_path: Path, target_path: Path, n_semitones: int):
    y, sr = librosa.load(audio_path)
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_semitones)

    sf.write(target_path, y_shifted, sr)

def adjust_volume(audio_path: Path, target_path: Path, db_change: float):
    audio = AudioSegment.from_file(audio_path)
    adjusted_audio = audio + db_change

    adjusted_audio.export(target_path, format=target_path.suffix[1:])  # remove dot from ext

def spectrogram_channel_shuffle(image_path: Path, target_path: Path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    channels = [0, 1, 2]
    random.shuffle(channels)
    shuffled_img_np = img_np[:, :, channels]

    Image.fromarray(shuffled_img_np).save(target_path)

def spectrogram_random_shifts(
    image_path: Path,
    target_path: Path,
    max_time_shift: int = 30,
    max_pitch_shift: int = 20
):
    img = Image.open(image_path).convert("RGB")
    spec = np.array(img)

    time_shift = random.randint(-max_time_shift, max_time_shift)
    pitch_shift = random.randint(-max_pitch_shift, max_pitch_shift)

    spec = np.roll(spec, shift=pitch_shift, axis=0)
    spec = np.roll(spec, shift=time_shift, axis=1)

    if pitch_shift > 0:
        spec[:pitch_shift, :, :] = 0
    elif pitch_shift < 0:
        spec[pitch_shift:, :, :] = 0

    if time_shift > 0:
        spec[:, :time_shift, :] = 0
    elif time_shift < 0:
        spec[:, time_shift:, :] = 0

    Image.fromarray(spec).save(target_path)

AUGMENT_TECHNIQUES = {
    'SP': {
        'label': 'SP',
        'type': 'audio',
        'apply': lambda audio_path, target_path: apply_pitch_shift(
            audio_path=audio_path,
            target_path=target_path,
            n_semitones=2
        )
    },
    'VA': {
        'label': 'VA',
        'type': 'audio',
        'apply': lambda audio_path, target_path: adjust_volume(
            audio_path=audio_path,
            target_path=target_path,
            db_change=5
        )
    },
    'SCS': {
        'label': 'SCS',
        'type': 'image',
        'apply': spectrogram_channel_shuffle
    },
    'SRS': {
        'label': 'SRS',
        'type': 'image',
        'apply': spectrogram_random_shifts
    }
}