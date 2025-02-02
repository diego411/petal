import os
from datetime import datetime
from typing import Optional, List

import numpy as np
from flask import current_app
from psycopg2.extensions import cursor as Cursor
from dropbox.exceptions import ApiError

from src.AppConfig import AppConfig
from src.controller import dropbox_controller
from src.database.db import transactional
from src.entity.Recording import Recording
from src.entity.RecordingState import RecordingState
from src.entity.Experiment import Experiment
from src.service import measurement_service, wav_converter, user_service, labeler, observation_service


def get_all(user: id):
    recordings = (find_by_user_and_state(
        user,
        RecordingState.REGISTERED
    ) + find_by_user_and_state(
        user,
        RecordingState.RUNNING
    ))  # TODO: these should be filters
    return {
        'recordings': to_dtos(recordings)
    }


def get_all_without_experiment(user: id):
    recordings = (find_by_user_and_state_without_experiment(
        user,
        RecordingState.REGISTERED
    ) + find_by_user_and_state_without_experiment(
        user,
        RecordingState.RUNNING
    ))  # TODO: these should be filters
    return {
        'recordings': to_dtos(recordings)
    }


def to_dtos(recordings: List[Recording]) -> List[dict]:
    return list(map(
        lambda recording: to_dto(recording),
        recordings
    ))


def to_dto(recording: Recording) -> dict:
    dto = {
        'id': recording.id,
        'name': recording.name,
        'user': recording.user,
        'state': recording.state.name,
        'sample_rate': recording.sample_rate,
        'threshold': recording.threshold,
        'measurements': measurement_service.get_values_for_recording(
            recording.id,
            recording.threshold * 2
        )
    }

    if recording.start_time is not None:
        dto['start_time'] = recording.start_time.strftime('%Y-%m-%d %H:%M:%S')

    if recording.last_update is not None:
        dto['last_update'] = recording.last_update.strftime('%Y-%m-%d %H:%M:%S')

    return dto


@transactional()
def find_by_id(cursor: Cursor, recording_id: int) -> Optional[Recording]:
    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE id=%(id)s;
        ''',
        {'id': recording_id}
    )

    result = cursor.fetchone()
    if result is None:
        return None
    return Recording.from_(result)


@transactional()
def find_not_stopped_by_user_and_name(cursor: Cursor, user: int, name: str) -> Optional[Recording]:
    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE user_id=%(user_id)s AND name=%(name)s AND state != %(state)s
        ''',
        {'user_id': user, 'name': name, 'state': RecordingState.STOPPED.value}
    )

    result = cursor.fetchone()
    if result is None:
        return None

    return Recording.from_(result)


@transactional()
def find_by_user_and_state(cursor: Cursor, user: int, state: RecordingState) -> List[Recording]:
    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE state=%(state)s AND user_id=%(user_id)s
        ''',
        {'state': state.value, 'user_id': user}
    )

    result = cursor.fetchall()

    if result is None:
        return []

    return list(map(
        lambda recording: Recording.from_(recording),
        result
    ))


@transactional()
def find_by_user_and_state_without_experiment(cursor: Cursor, user: int, state: RecordingState) -> List[Recording]:
    cursor.execute(
        '''
            SELECT recording.*
            FROM recording
            LEFT JOIN experiment ON recording.id = experiment.recording_id
            WHERE state=%(state)s AND recording.user_id=%(user_id)s AND experiment.recording_id IS NULL;
        ''',
        {'state': state.value, 'user_id': user}
    )

    result = cursor.fetchall()

    if result is None:
        return []

    return list(map(
        lambda recording: Recording.from_(recording),
        result
    ))


@transactional()
def find_by_user(cursor: Cursor, user: int) -> List[Recording]:
    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE user_id=%(user_id)s;
        ''',
        {'user_id': user}
    )

    result = cursor.fetchall()
    if result is None:
        return []

    return list(map(
        lambda recording: Recording.from_(recording),
        result
    ))


@transactional()
def create(cursor: Cursor, name: str, user: int, state: RecordingState, sample_rate: int, threshold: int) -> int:
    cursor.execute(
        '''
            INSERT INTO recording (name, user_id, state, sample_rate, threshold)
            VALUES (%(name)s, %(user_id)s, %(state)s, %(sample_rate)s, %(threshold)s)
            RETURNING id;
        ''',
        {'name': name, 'user_id': user, 'state': state.value, 'sample_rate': sample_rate, 'threshold': threshold}
    )
    return cursor.fetchone()[0]


@transactional()
def delete(cursor: Cursor, recording: int):
    cursor.execute(
        """
            DELETE FROM recording
            WHERE id=%(id)s;
        """,
        {'id': recording}
    )

    cursor.execute(  # TODO: why doesn't cascade work for this?
        '''
            DELETE FROM measurement
            WHERE recording=%(recording)s;
        ''',
        {'recording': recording}
    )


@transactional()
def set_last_update(cursor: Cursor, recording: int, date: datetime):
    cursor.execute(
        '''
            UPDATE recording
            SET last_update=%(last_update)s
            WHERE id=%(id)s
        ''',
        {'id': recording, 'last_update': date}
    )


@transactional()
def set_start_time(cursor: Cursor, recording: int, date: datetime):
    cursor.execute(
        '''
            UPDATE recording
            SET start_time=%(start_time)s
            WHERE id=%(id)s
        ''',
        {'id': recording, 'start_time': date}
    )


@transactional()
def set_state(cursor: Cursor, recording: int, state: RecordingState):
    cursor.execute(
        """
            UPDATE recording
            SET state=%(state)s
            WHERE id=%(id)s;
        """,
        {'id': recording, 'state': state.value}
    )


def start(recording: Recording):
    now = datetime.now()
    set_last_update(recording.id, now)
    set_start_time(recording.id, now)
    set_state(recording.id, RecordingState.RUNNING)

    current_app.socketio.emit('recording-start', {
        'name': recording.name,
        'id': recording.id,
        'start_time': now.strftime('%d.%m.%Y %H:%M:%S'),
    })


def stop(recording: Recording):
    user = user_service.find_by_id(recording.user)
    measurements = measurement_service.get_values_for_recording(recording.id)
    len_measurements = len(measurements)
    start_time = recording.start_time
    last_update = recording.last_update
    delta_seconds = (last_update - start_time).seconds
    calculated_sample_rate = int(len_measurements / delta_seconds) if delta_seconds != 0 else 0
    current_app.logger.info(
        f"""
            Stopping recording with id {recording.id}. 
            Start time: {start_time}. 
            Last updated: {last_update}'.
            Delta seconds: {delta_seconds}.
            Number of measurements: {len_measurements}.
            Calculated sample rate: {calculated_sample_rate} 
        """
    )

    file_name = f'{recording.name}_{calculated_sample_rate}Hz_{int(start_time.timestamp() * 1000)}.wav'
    file_path = wav_converter.convert(
        measurements,
        sample_rate=calculated_sample_rate,
        path=f'audio/{file_name}'
    )

    set_state(recording.id, RecordingState.STOPPED)

    shared_link: str = dropbox_controller.upload_file_to_dropbox(
        file_path=file_path,
        dropbox_path=f'/PlantRecordings/{user.name}/{file_name}'
    )

    os.remove(file_path)

    if AppConfig.DELETE_MEASUREMENTS_AFTER_STOP:
        measurement_service.delete_for_recording(recording.id)

    current_app.socketio.emit('recording-stop', {
        'id': recording.id,
        'name': recording.name
    })

    return shared_link


def stop_and_label(experiment: Experiment, recording: Recording):
    start_time = recording.start_time
    delta_seconds = (recording.last_update - start_time).seconds
    measurements = measurement_service.get_values_for_recording(recording.id)
    sample_rate = int(len(measurements) / delta_seconds) if delta_seconds != 0 else 0
    observations = observation_service.find_by_experiment(experiment.id)

    file_name_prefix = f'{recording.name}_{sample_rate}Hz_{int(start_time.timestamp() * 1000)}'
    file_name = f'{file_name_prefix}.wav'
    file_path = wav_converter.convert(
        measurements,
        sample_rate=sample_rate,
        path=f'audio/{file_name}'
    )

    set_state(recording.id, RecordingState.STOPPED)

    labeler.label_recording(
        experiment=experiment,
        recording_path=file_path,
        recording=recording,
        observations=observations,
        dropbox_path_prefix=file_name_prefix
    )

    try:
        dropbox_controller.upload_file_to_dropbox(
            file_path=file_path,
            dropbox_path=f'/PlantRecordings/{recording.name}/{file_name}'
        )
    except ApiError as e:
        current_app.logger.error(e.error)

    os.remove(file_path)

    if AppConfig.DELETE_MEASUREMENTS_AFTER_STOP:
        measurement_service.delete_for_recording(recording.id)

    current_app.socketio.emit('recording-stop', {
        'id': recording.id,
        'name': recording.name
    })


def run_update(recording: Recording, data: bytes, now: datetime):
    parsed_data = process_parsed_data(
        parsed_data=wav_converter.parse_raw(data),
        recording_id=recording.id,
        sample_rate=recording.sample_rate or 142,
        # TODO: add column to recording tracking number of measurements
        number_of_persisted_measurements=measurement_service.get_count(recording.id),
        start_time=recording.start_time,
        now=now
    )

    current_app.socketio.emit(
        f'recording-update',
        {
            'measurements': parsed_data,
            'name': recording.name,
            'id': recording.id,
            'threshold': recording.threshold or 9000,
            'last_update': now.strftime('%Y-%m-%d %H:%M:%S')
        }
    )

    measurement_service.insert_many(recording.id, parsed_data, now)
    set_last_update(recording.id, now)

    current_app.logger.info(
        f'Entire update for recording with id {recording.id} took {int((datetime.now() - now).total_seconds() * 1000)}ms.'
    )


def process_parsed_data(
        parsed_data: List[float],
        recording_id: int,
        sample_rate: int,
        number_of_persisted_measurements: int,
        start_time: datetime,
        now: datetime
) -> List[float]:
    seconds_since_start: int = (now - start_time).seconds
    number_of_expected_measurements: int = int(seconds_since_start * sample_rate)

    assert number_of_persisted_measurements < number_of_expected_measurements, f"""
            Assertion for recording with id: {recording_id} failed:
            The number of already persisted measurements is larger than the expected number of measurements.
            This should not have happened.
            Number of persisted measurements: {number_of_persisted_measurements}
            Number of expected measurements: {number_of_expected_measurements}
            Canceling this update!
        """

    diff_number_measurements: int = number_of_expected_measurements - (
            number_of_persisted_measurements + len(parsed_data))

    """
        At the end of the update we want to persist a number of measurements so that we have the exact number of expected
        measurements.
        So we want that #expected = #persisted + #parsed
        
        Case 1: If #expected = #persisted + #parsed, don't do anything
        
        Case 2: If #expected < #persisted + #parsed (we have more measurements that expected) we need to remove the oldest
                (#persisted + #parsed) - #expected measurements from parsed or keep the newest #expected - #persisted
                measurements from parsed.
   
                Example 1: #expected = 142, #persisted = 0, #parsed = 4500 => we need to keep the newest
                (142 + 0) = 142 measurements from parsed (or remove the oldest 4358).
   
                Example 2: #expected = 84000, #persisted = 79740, #parsed = 5000 (84000 < 79740 + 5000) => we need to
                keep the newest (84000 - 79740) = 4260 from parsed (or remove the oldest 740).
   
        Case 3: If #expected > #persited + #parsed (we have less measurements than expected) we need to interpolate
                measurements onto parsed so that, parsed contains (#expected - #persisted) measurements
   
                Example: #expected = 8400, #persited = 79740, #parsed = 4000 (84000 > 79740 + 4000) => we need to
                interpolate measurements so that parsed contains 4260 measurements.
    """

    if diff_number_measurements == 0:  # Case 1: correct amount of measurements
        return parsed_data

    if diff_number_measurements < 0:  # Case 2: we have more measurements than expected
        parsed_data = parsed_data[-(number_of_expected_measurements - number_of_persisted_measurements):]

    elif diff_number_measurements > 0:  # Case 3: we have fewer measurements than expected
        parsed_data = interpolate_list(
            parsed_data,
            number_of_expected_measurements - number_of_persisted_measurements
        )

    return parsed_data


def interpolate_list(data, m):
    if m <= len(data):
        return data

    n = len(data)
    x = np.arange(n)  # Original x-values
    y = np.array(data)  # Original y-values

    # Create new x-values for the extended list, including original indices
    extended_x = np.linspace(0, n - 1, m - n)  # Adjust spacing to get 'm' points
    extended_x = np.sort(np.concatenate((x, extended_x)))

    # Interpolate to get the extended y-values
    extended_y = np.interp(extended_x, x, y)

    return extended_y.tolist()
