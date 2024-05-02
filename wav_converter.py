import time
import numpy as np
import scipy.io.wavfile as wav


def parse_raw(data):
    data = data[:-3]
    chunks = [data[i:i + 4] for i in range(0, len(data), 4)]
    out = []
    for chunk in chunks:
        try:
            value = int(chunk, 16) / 4095
            out.append(value)
        except:
            continue

    return out


def convert(voltages):
    # Assuming the voltage measurements range from 0 to 1, map them to audio samples (-1 to 1)
    audio_samples = np.array(voltages) * 2 - 1

    # Scale the audio samples to fit within the valid range for 16-bit audio (-32768 to 32767)
    scaled_audio_samples = (audio_samples * 32767).astype(np.int16)

    # Set the sample rate
    sample_rate = 10000  # e.g., 44.1 kHz

    # Write audio data to a WAV file
    extension = 'wav'
    date = time.ctime(time.time())
    file_path = f'audio/plant_audio_{date}.{extension}'
    wav.write(file_path, sample_rate, scaled_audio_samples)

    return file_path
