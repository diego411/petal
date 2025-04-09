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

AUGMENT_AUDIO_PATH = Path.home() / '.data/petal/augmented-audios'
AUGMENT_IMAGE_PATH = Path.home() / '.data/petal/augmented-images'


def get_augment_audio_path(audio_path: Path) -> Path:
    if not AUGMENT_AUDIO_PATH.exists():
        os.makedirs(AUGMENT_AUDIO_PATH)

    return AUGMENT_AUDIO_PATH / audio_path.name

def get_augment_image_path(image_path: Path) -> Path:
    if not AUGMENT_IMAGE_PATH.exists():
        os.makedirs(AUGMENT_IMAGE_PATH)
    
    return AUGMENT_IMAGE_PATH / image_path.name

def random_audio_segment(audio_path) -> Tuple[np.ndarray, int | float]:
    audio, sample_rate = librosa.load(audio_path, sr=None)
    total_samples = len(audio)
    segment_samples = int(total_samples / 3)
    
    start_idx = np.random.randint(0, total_samples - segment_samples)
    return audio[start_idx : start_idx + segment_samples], sample_rate

def apply_pitch_shift(audio_path: Path, n_semitones: int) -> Path:
    y, sr = librosa.load(audio_path)
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_semitones)

    augment_audio_path = get_augment_audio_path(audio_path)
    sf.write(augment_audio_path, y_shifted, sr)

    return augment_audio_path

def adjust_volume(audio_path: Path, db_change: float) -> Path:
    audio = AudioSegment.from_file(audio_path)
    adjusted_audio = audio + db_change

    augment_audio_path = get_augment_audio_path(audio_path)
    adjusted_audio.export(augment_audio_path, format=augment_audio_path.suffix[1:])  # remove dot from ext

    return augment_audio_path

def spectrogram_channel_shuffle(image_path: Path) -> Path:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    channels = [0, 1, 2]
    random.shuffle(channels)
    shuffled_img_np = img_np[:, :, channels]

    augment_image_path = get_augment_image_path(image_path)
    Image.fromarray(shuffled_img_np).save(augment_image_path)
    return augment_image_path

def spectrogram_random_shifts(
    image_path: Path,
    max_time_shift: int = 30,
    max_pitch_shift: int = 20
) -> Path:
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

    augment_image_path = get_augment_image_path(image_path)
    Image.fromarray(spec).save(augment_image_path)
    return augment_image_path

AUGMENT_TECHNIQUES = {
    'SP': {
        'type': 'audio',
        'apply': lambda audio_path: apply_pitch_shift(audio_path, n_semitones=2)
    },
    'VA': {
        'type': 'audio',
        'apply': lambda audio_path: adjust_volume(audio_path, db_change=5)
    },
    'SCS': {
        'type': 'image',
        'apply': spectrogram_channel_shuffle
    },
    'SRS': {
        'type': 'image',
        'apply': spectrogram_random_shifts
    }
}