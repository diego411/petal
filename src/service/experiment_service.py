from src.database.db import transactional
from psycopg2.extensions import cursor as Cursor
from datetime import datetime
from src.entity.Experiment import Experiment
from typing import Optional, List
from src.service import recording_service


def get_all(user: id):
    experiments: List[Experiment] = find_by_user(user)

    return {
        'experiments': to_dtos(experiments)
    }


def to_dtos(experiments: List[Experiment]) -> List[dict]:
    return list(map(
        lambda experiment: to_dto(experiment),
        experiments
    ))


def to_dto(experiment: Experiment) -> dict:
    recording = None
    if experiment.recording is not None:
        recording = recording_service.find_by_id(experiment.recording)

    dto = {
        "id": experiment.id,
        "name": experiment.name,
        "status": experiment.status,
        "user": experiment.user,
        "created_at": experiment.created_at.strftime('%Y-%m-%d %H:%M:%S'),
    }

    if recording is not None:
        dto['recording'] = recording_service.to_dto(recording)

    if experiment.started_at is not None:
        dto['started_at'] = experiment.started_at.strftime('%Y-%m-%d %H:%M:%S')

    return dto


@transactional()
def create(cursor: Cursor, name: str, user_id: int):
    cursor.execute(
        '''
            INSERT INTO experiment (name, status, user_id, created_at)
            VALUES (%(name)s, %(status)s, %(user_id)s, %(created_at)s)
            RETURNING id;
        ''',
        {'name': name, 'status': 'CREATED', 'user_id': user_id, 'created_at': datetime.now()}
    )

    return cursor.fetchone()[0]


@transactional()
def find_by_id(cursor: Cursor, experiment_id: int) -> Optional[Experiment]:
    cursor.execute(
        '''
            SELECT *
            FROM experiment
            WHERE id=%(experiment_id)s 
        ''',
        {'experiment_id': experiment_id}
    )

    result = cursor.fetchone()
    if result is None:
        return None

    return Experiment.from_(result)


@transactional()
def find_by_user(cursor: Cursor, user_id: int) -> List[Experiment]:
    cursor.execute(
        '''
            SELECT *
            FROM experiment
            WHERE user_id=%(user_id)s
            ORDER BY created_at DESC;
        ''',
        {'user_id': user_id}
    )

    result = cursor.fetchall()

    if result is None:
        return []

    return list(map(
        lambda experiment: Experiment.from_(experiment),
        result
    ))


@transactional()
def find_by_recording(cursor: Cursor, recording_id: int) -> Optional[Experiment]:
    cursor.execute(
        '''
            SELECT *
            FROM experiment
            WHERE recording_id=%(recording_id)s;
        ''',
        {'recording_id': recording_id}
    )

    result = cursor.fetchone()
    if result is None:
        return None

    return Experiment.from_(result)


@transactional()
def set_status(cursor: Cursor, experiment_id: int, status: str):
    cursor.execute(
        '''
            UPDATE experiment
            SET status=%(status)s
            WHERE id=%(experiment_id)s;
        ''',
        {'status': status, 'experiment_id': experiment_id}
    )


@transactional()
def set_started_at(cursor: Cursor, experiment_id: int, started_at: datetime):
    cursor.execute(
        '''
            UPDATE experiment
            SET started_at = %(started_at)s
            WHERE id=%(experiment_id)s;
        ''',
        {'started_at': started_at, 'experiment_id': experiment_id}
    )


@transactional()
def set_recording(cursor: Cursor, experiment_id: int, recording_id: int):
    cursor.execute(
        '''
            UPDATE experiment
            SET recording_id=%(recording_id)s
            WHERE id=%(experiment_id)s;
        ''',
        {'recording_id': recording_id, 'experiment_id': experiment_id}
    )
