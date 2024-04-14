import time
import numpy as np
import scipy.io.wavfile as wav
import random

def convert(voltages):
    voltages = []
    for _ in range(10000 * 100): 
        voltages.append(random.uniform(0, 1))
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
