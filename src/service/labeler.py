import os
from pydub import AudioSegment
from src.controller import dropbox_controller
from dropbox.exceptions import ApiError
from flask import current_app
from src.entity.Experiment import Experiment
from src.AppConfig import AppConfig


def merge_observations(observations: list, gap_threshold: int = AppConfig.MERGE_OBSERVATIONS_THRESHOLD):
    """
    Merge observations with small gaps between the same emotion.

    :param observations: List of observations sorted by timestamp.
    :param gap_threshold: Maximum duration (in ms) of gaps to merge.
    :return: Merged list of observations.
    """
    merged = []
    stack = list(reversed(observations))
    while len(stack) >= 1:
        observation = stack.pop()

        merged.append(observation)

        if len(stack) == 0:
            continue

        next_observation = stack[-1]
        threshold = (next_observation['timestamp'] - observation['timestamp']) + gap_threshold

        next_observation_with_same_emotion_in_range = find_next_with_same_emotion_in_range(
            observations=list(reversed(stack)),
            emotion=observation['emotion'],
            timestamp=observation['timestamp'],
            threshold=threshold
        )

        while next_observation_with_same_emotion_in_range is not None:
            item = stack.pop()
            if item == next_observation_with_same_emotion_in_range:
                break

    return merged


def find_next_with_same_emotion_in_range(observations: list, emotion: str, timestamp: int, threshold: int):
    for i in range(0, len(observations)):
        observation = observations[i]
        if observation['emotion'] == emotion and (observation['timestamp'] - timestamp) < threshold:
            return observation

    return None


def label_recording(
        experiment: Experiment,
        recording_path: str,
        observations: list = None,
        dropbox_path_prefix: str = None
):
    if observations is None or len(observations) == 0:
        return

    split_recording_path = recording_path.split('.')[0]
    split_recording_path = split_recording_path.split('_')
    recording_start_timestamp = int(split_recording_path[len(split_recording_path) - 1])

    try:
        recording = AudioSegment.from_wav(recording_path)
    except Exception:
        return  # TODO throw bad request exception or something (user feedback)

    observations = sorted(observations, key=lambda x: x['timestamp'])
    full_length = len(observations)

    observations = merge_observations(observations)
    merged_length = len(observations)

    current_app.logger.info(
        f"Preprocessing for experiment {experiment.id} filtered out {full_length - merged_length} observations"
    )

    for i in range(0, len(observations)):
        observation = observations[i]
        next_observation = observations[i + 1] if i + 1 < len(observations) else None

        start = observation['timestamp']
        end = next_observation['timestamp'] if next_observation is not None else None

        if 'emotion' in observation:
            emotion = observation['emotion']
        else:
            emotion = max({key: observation[key] for key in
                           ['happy', 'surprised', 'neutral', 'sad', 'angry', 'disgusted', 'fearful']},
                          key=observation.get)
        current_app.logger.info(
            f"""
                Processing the {i}-th observation of experiment {experiment.id} 
                From {start} to {end} the emotion \"{emotion}\" was predicted!
            """
        )

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

        file_path = f"{directory}/{experiment.id}.wav"
        snippet.export(file_path, format="wav")

        dropbox_file_name = f"{experiment.id}_{i}"
        if dropbox_path_prefix:
            dropbox_file_name = f"{dropbox_path_prefix}_{dropbox_file_name}"
        # TODO: can you do a bulk upload somehow?
        try:
            dropbox_controller.upload_file_to_dropbox(
                file_path=file_path,
                dropbox_path=f"/PlantRecordings/Labeled/{emotion}/{dropbox_file_name}.wav"
            )
        except ApiError as e:
            current_app.logger.error(e.error)

        os.remove(file_path)
