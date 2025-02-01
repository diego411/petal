import os
from pydub import AudioSegment
from src.controller import dropbox_controller
from dropbox.exceptions import ApiError
from flask import current_app
from src.entity.Experiment import Experiment
from src.AppConfig import AppConfig
from typing import List
from src.entity.Obervation import Observation
from datetime import datetime, timedelta


def merge_observations(observations: List[Observation], gap_threshold: int = AppConfig.MERGE_OBSERVATIONS_THRESHOLD):
    """
    Merge observations with small gaps between the same emotion.

    :param observations: List of observations sorted by timestamp.
    :param gap_threshold: Maximum duration (in ms) of gaps to merge.
    :return: Merged list of observations.
    """
    merged = []
    stack = list(reversed(observations))
    while len(stack) >= 1:
        observation: Observation = stack.pop()

        merged.append(observation)

        if len(stack) == 0:
            continue

        next_observation: Observation = stack[-1]
        threshold: timedelta = (next_observation.observed_at - observation.observed_at) + timedelta(
            seconds=gap_threshold)

        next_observation_with_same_emotion_in_range = find_next_with_same_emotion_in_range(
            observations=list(reversed(stack)),
            label=observation.label,
            timestamp=observation.observed_at,
            threshold=threshold
        )

        while next_observation_with_same_emotion_in_range is not None:
            item = stack.pop()
            if item == next_observation_with_same_emotion_in_range:
                break

    return merged


def find_next_with_same_emotion_in_range(
        observations: List[Observation],
        label: str,
        timestamp: datetime,
        threshold: timedelta
):
    for i in range(0, len(observations)):
        observation: Observation = observations[i]
        if observation.label == label and (observation.observed_at - timestamp) < threshold:
            return observation

    return None


def label_recording(
        experiment: Experiment,
        recording_path: str,
        observations: List[Observation] = None,
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

    observations = sorted(observations, key=lambda obs: obs.observed_at)
    full_length = len(observations)

    observations = merge_observations(observations)
    merged_length = len(observations)

    current_app.logger.info(
        f"Preprocessing for experiment {experiment.id} filtered out {full_length - merged_length} observations"
    )

    for i in range(0, len(observations)):
        observation: Observation = observations[i]
        next_observation = observations[i + 1] if i + 1 < len(observations) else None

        start = observation.observed_at.timestamp()
        end = next_observation.observed_at.timestamp() if next_observation is not None else None

        emotion = observation.label
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
