from src.database.db import transactional
from psycopg2.extensions import cursor as Cursor
from datetime import datetime
from src.entity.Recording import Recording
from src.entity.RecordingState import RecordingState
from src.service import measurement_service
from typing import Optional, List


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
def find_by_state(cursor: Cursor, state: RecordingState) -> List[Recording]:
    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE state=%(state)s
        ''',
        {'state': state.value}
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
