from src.database.db import transactional
from psycopg2.extensions import cursor as Cursor
from datetime import datetime
from src.entity.Recording import Recording
from src.entity.RecordingState import RecordingState
from src.service import measurement_service, wav_converter, user_service, labeler
from src.controller import dropbox_controller
from typing import Optional, List
from flask import current_app
import os
from src.AppConfig import AppConfig


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
def create(cursor: Cursor, name: str, user: int, state: RecordingState, sample_rate: int, threshold: int):
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

    dropbox_controller.upload_file_to_dropbox(
        dropbox_client=current_app.dropbox_client,
        file_path=file_path,
        dropbox_path=f'/PlantRecordings/{user.name}/{file_name}'
    )

    os.remove(file_path)

    set_state(recording.id, RecordingState.STOPPED)

    if AppConfig.DELETE_AFTER_STOP:
        delete(recording.id)

    current_app.socketio.emit('recording-stop', {
        'id': recording.id,
        'name': recording.name
    })


def stop_and_label(recording: Recording, emotions: dict):
    start_time = recording.start_time
    delta_seconds = (recording.last_update - start_time).seconds
    measurements = measurement_service.get_values_for_recording(recording.id)
    sample_rate = int(len(measurements) / delta_seconds) if delta_seconds != 0 else 0

    file_name_prefix = f'{recording.name}_{sample_rate}Hz_{int(start_time.timestamp() * 1000)}'
    file_name = f'{file_name_prefix}.wav'
    file_path = wav_converter.convert(
        measurements,
        sample_rate=sample_rate,
        path=f'audio/{file_name}'
    )

    labeler.label_recording(
        recording_path=file_path,
        observations_path='',
        observations=emotions,
        dropbox_client=current_app.dropbox_client,
        dropbox_path_prefix=file_name_prefix
    )

    dropbox_controller.upload_file_to_dropbox(
        dropbox_client=current_app.dropbox_client,
        file_path=file_path,
        dropbox_path=f'/PlantRecordings/{recording.name}/{file_name}'
    )

    os.remove(file_path)
    set_state(recording.id, RecordingState.STOPPED)

    if AppConfig.DELETE_AFTER_STOP:
        delete(recording.id)

    current_app.socketio.emit('recording-stop', {
        'id': recording.id,
        'name': recording.name
    })


def run_update(recording: Recording, data: bytes, now: datetime):
    sample_rate = recording.sample_rate or 142
    number_of_persisted_measurements = measurement_service.get_count(recording.id)
    start_time = recording.start_time
    parsed_data: list = wav_converter.parse_raw(data)

    seconds_since_start = (now - start_time).seconds
    expected_measurement_count = int(seconds_since_start * sample_rate)
    diff_number_measurements = expected_measurement_count - (number_of_persisted_measurements + len(parsed_data))
    if number_of_persisted_measurements == 0:
        parsed_data = parsed_data[:expected_measurement_count]
    elif diff_number_measurements > 0:
        parsed_data += [parsed_data[-1]] * diff_number_measurements

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
