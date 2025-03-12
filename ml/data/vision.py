# TODO: move to data dir
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import Optional
from PIL import Image
import torchaudio.transforms as T

def plot_waveform(waveform: Tensor):
    plt.figure(figsize=(8, 5))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Sine waveform")
    plt.plot(waveform.t().numpy()[0:1000])
    plt.show()

def plot_fourier_transform(waveform: Tensor, sample_rate: float):
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    # Perform FFT
    N = waveform.shape[0]
    fft_result = torch.fft.fft(waveform)
    frequencies = torch.fft.fftfreq(N, d=1/sample_rate)

    # Take only the positive frequencies
    positive_mask = frequencies >= 0
    positive_freqs = frequencies[positive_mask]
    magnitude = torch.abs(fft_result[positive_mask])

    # Convert tensors to numpy for plotting
    positive_freqs = positive_freqs.numpy()
    magnitude = magnitude.numpy()

    # Plot the frequency spectrum
    plt.figure(figsize=(8, 5))
    plt.plot(positive_freqs, magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Fourier Transform (Frequency Spectrum)")
    plt.show()


def show_and_save_spectrogram_image(spectrogram: np.ndarray, n_fft: int, sample_rate: float, path: Optional[Path]):
    freqs = np.fft.rfftfreq(n_fft, d=1/sample_rate)  # Compute frequency values
    
    fig, ax = plt.subplots()
    img = ax.imshow(
        spectrogram,
        cmap='viridis',
        origin='lower',
        aspect='auto',
        extent=(
            0.0,
            spectrogram.shape[1],
            freqs[0],
            freqs[-1]
        )
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Spectrogram')
    cbar = fig.colorbar(img, ax=ax, label="Intensity")

    plt.show()

    if path is None:
        return

    # reset plot before saving image
    ax.axis('off')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.remove()

    # save spectrogram image
    fig.savefig(
        path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0,
        transparent=True
    )

    plt.close(fig)

def show_mel_spectrogram(mel_spectrogram: np.ndarray, n_fft: int, sample_rate: int, path: Optional[Path]):
    freqs = np.fft.rfftfreq(n_fft, d=1/sample_rate)  # Compute frequency values
    
    fig, ax = plt.subplots()
    img = ax.imshow(
        mel_spectrogram,
        cmap='viridis',
        origin='lower',
        aspect='auto',
        extent=(
            0.0,
            mel_spectrogram.shape[1],
            freqs[0],
            freqs[-1]
        )
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel Frequency')
    ax.set_title('Mel Spectrogram')
    cbar = fig.colorbar(img, ax=ax, label="Decibels (dB)")

    plt.show()

    if path is None:
        return

    # reset plot before saving image
    ax.axis('off')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.remove()

    # save spectrogram image
    fig.savefig(
        path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0,
        transparent=True
    )

    plt.close(fig)

def visualize_transform(image_path: Path, transform):
    image = Image.open(image_path)

    transformed_image = transform(image)

    # Display the original and transformed images
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(transformed_image)
    ax[1].set_title("Resized Image (224x224)")
    ax[1].axis("off")

    plt.show()
