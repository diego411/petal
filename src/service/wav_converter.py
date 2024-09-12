import time
import numpy as np
import scipy.io.wavfile as wav


def parse_raw(data):
    offset = len(data) % 4
    data = data[: -1 * offset if offset != 0 else len(data)]
    chunks = [data[i:i + 4] for i in range(0, len(data), 4)]
    out = []
    for chunk in chunks:
        try:
            value = int(chunk, 16) / 4095
            out.append(value)
        except:
            continue

    return out


def augment(voltages, augment_window, augment_padding):
    augmented_voltages = []
    n = len(voltages)

    for i in range(0, n, augment_window):
        # Get the current window of values
        current_window = voltages[i:i + augment_window]

        # Calculate the average of the current window
        if len(current_window) > 0:
            avg_value = sum(current_window) / len(current_window)
        else:
            avg_value = 0  # In case the window is empty

        # Append current window
        augmented_voltages.extend(current_window)
        # Append the average value `augment_padding` times
        augmented_voltages.extend([avg_value] * augment_padding)

    return augmented_voltages


def convert(voltages, sample_rate=10_000, path=None):
    # Assuming the voltage measurements range from 0 to 1, map them to audio samples (-1 to 1)
    audio_samples = np.array(voltages) * 2 - 1
    # Scale the audio samples to fit within the valid range for 16-bit audio (-32768 to 32767)
    # scaled_audio_samples = (audio_samples * 32767).astype(np.int16)

    # Write audio data to a WAV file
    extension = 'wav'
    file_path = path
    if file_path is None:
        date = time.ctime(time.time())
        file_path = f'audio/plant_audio_{date}.{extension}'
    wav.write(file_path, sample_rate, audio_samples.astype(np.float32))

    return file_path
