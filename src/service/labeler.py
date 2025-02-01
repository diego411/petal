import os
from pydub import AudioSegment
from src.controller import dropbox_controller
from dropbox.exceptions import ApiError
from flask import current_app
from src.entity.Experiment import Experiment
from src.AppConfig import AppConfig
from typing import List, Tuple, Optional
from src.entity.Obervation import Observation
from src.entity.Recording import Recording
from datetime import datetime, timedelta


def merge_observations(observations: List[Observation], gap_threshold: int = AppConfig.MERGE_OBSERVATIONS_THRESHOLD):
    """
    Merge observations with small gaps between the same label.

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
            milliseconds=gap_threshold)

        next_observation_with_same_label_in_range = find_next_with_same_label_in_range(
            observations=list(reversed(stack)),
            label=observation.label,
            timestamp=observation.observed_at,
            threshold=threshold
        )

        while next_observation_with_same_label_in_range is not None:
            item = stack.pop()
            if item == next_observation_with_same_label_in_range:
                break

    return merged


def find_next_with_same_label_in_range(
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


def get_start_and_end_ms(
        observation: Observation,
        next_observation: Optional[Observation],
        recording_start_timestamp: float,
        recording_length: int
) -> Tuple[float, float]:
    start: float = int(observation.observed_at.timestamp() * 1000)  # datetime.timestamp return timestamp in seconds
    end: float = int(next_observation.observed_at.timestamp() * 1000) if next_observation is not None else None

    # Convert these to milliseconds
    start_ms: float = start - recording_start_timestamp
    end_ms: float = end - recording_start_timestamp if end is not None else None

    if start_ms < 0:
        start_ms = 0  # Clip start of observation to the start of the recording

    if end_ms is None or end_ms > recording_length:
        end_ms = recording_length  # Clip end of observation to the end of the recording

    return start_ms, end_ms


def get_file_path(label: str, experiment_id: int) -> str:
    directory = f'./audio/{label}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    return f"{directory}/{experiment_id}.wav"


def upload_to_dropbox(experiment_id: int, index: int, dropbox_path_prefix: str, file_path: str, label: str):
    dropbox_file_name = f"{experiment_id}_{index}"
    if dropbox_path_prefix:
        dropbox_file_name = f"{dropbox_path_prefix}_{dropbox_file_name}"
    # TODO: can you do a bulk upload somehow?
    try:
        dropbox_controller.upload_file_to_dropbox(
            file_path=file_path,
            dropbox_path=f"/EmotionExperiment/{label}/{dropbox_file_name}.wav"
        )
    except ApiError as e:
        current_app.logger.error(e.error)


def label_recording(
        experiment: Experiment,
        recording_path: str,
        recording: Recording,
        observations: List[Observation] = None,
        dropbox_path_prefix: str = None
):
    if observations is None or len(observations) == 0:
        return

    recording_start_timestamp = int(
        recording.start_time.timestamp() * 1000
    )  # datetime.timestamp return timestamp without millis

    try:
        recording = AudioSegment.from_wav(recording_path)
    except Exception as e:
        print(e)
        current_app.logger.error(f"Creating audio segment out of recording file failed while labeling: {e}")
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
        label = observation.label

        start_ms: float  # start of observation relative to start of the recording
        end_ms: float  # end of observation relative to end of the recording
        start_ms, end_ms = get_start_and_end_ms(
            observation=observation,
            next_observation=next_observation,
            recording_start_timestamp=recording_start_timestamp,
            recording_length=len(recording)
        )

        if end_ms <= start_ms:
            continue

        start_date_string = datetime.fromtimestamp((recording_start_timestamp + start_ms) / 1000).strftime(
            "%Y-%m-%d %H:%M:%S")
        end_date_string = datetime.fromtimestamp((recording_start_timestamp + end_ms) / 1000).strftime(
            "%Y-%m-%d %H:%M:%S")
        current_app.logger.info(
            f"""
                Processing the {i}-th observation of experiment {experiment.id} 
                From {start_date_string} to {end_date_string} the label \"{label}\" was predicted!
            """
        )

        snippet = recording[start_ms:end_ms]

        file_path = get_file_path(label=label, experiment_id=experiment.id)
        snippet.export(file_path, format="wav")

        upload_to_dropbox(
            experiment_id=experiment.id,
            index=i,
            file_path=file_path,
            dropbox_path_prefix=dropbox_path_prefix,
            label=label,
        )

        os.remove(file_path)
