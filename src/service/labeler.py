import os
import json
from pydub import AudioSegment
from src.controller import dropbox_controller
import dropbox


def label_recording(recording_path: str, observations_path: str, observations: dict, dropbox_client: dropbox.Dropbox):
    split_recording_path = recording_path.split('.')[0]
    split_recording_path = split_recording_path.split('_')
    recording_start_timestamp = int(split_recording_path[len(split_recording_path) - 1])

    if observations is None:
        with open(observations_path, 'r') as file:
            observations = json.load(file)

    recording = AudioSegment.from_wav(recording_path)

    observations = sorted(observations, key=lambda x: x['timestamp'])
    for i in range(0, len(observations)):
        observation = observations[i]
        next_observation = observations[i + 1] if i + 1 < len(observations) else None

        start = observation['timestamp']
        end = next_observation['timestamp'] if next_observation is not None else None

        if 'emotion' in observation:
            emotion = observation['emotion']
        else:
            emotion = max({key: observation[key] for key in
                       ['happy', 'surprised', 'neutral', 'sad', 'angry', 'disgusted', 'fearful']}, key=observation.get)
        print(f'From {start} to {end} this emotion was predicted: {emotion}')

        # Convert these to milliseconds
        start_ms = start - recording_start_timestamp
        end_ms = end - recording_start_timestamp if end is not None else None

        if start_ms < 0:
            start_ms = 0  # Clip to the start of the recording

        if end_ms is None or end_ms > len(recording):
            end_ms = len(recording)  # Clip to the end of the recording

        if end_ms <= start_ms:
            continue

        snippet = recording[start_ms:end_ms]

        directory = f'./audio/{emotion}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = f"{directory}/{observation['id']}.wav"
        snippet.export(file_path, format="wav")

        dropbox_controller.upload_file_to_dropbox(
            dropbox_client=dropbox_client,
            file_path=file_path,
            dropbox_path=f"/PlantRecordings/Labeled/{emotion}/{observation['id']}.wav"
        )

        os.remove(file_path)
