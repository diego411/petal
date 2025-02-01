from datetime import datetime
from src.database.db import transactional
from psycopg2.extensions import cursor as Cursor
from typing import List
from src.entity.Obervation import Observation


@transactional()
def find_by_experiment(cursor: Cursor, experiment_id: int) -> List[Observation]:
    cursor.execute(
        '''
             SELECT *
             FROM observation
             WHERE experiment_id=%(experiment_id)s;
        ''',
        {'experiment_id': experiment_id}
    )

    result: List[tuple] = cursor.fetchall()

    return list(map(
        lambda observation: Observation.from_(observation),
        result
    ))


@transactional()
def create(cursor: Cursor, label: str, observed_at: datetime, experiment_id: int):
    cursor.execute(
        '''
            INSERT INTO observation (label, observed_at, experiment_id)
            VALUES (%(label)s, %(observed_at)s, %(experiment_id)s)
            RETURNING id;
        ''',
        {'label': label, 'observed_at': observed_at, 'experiment_id': experiment_id}
    )

    return cursor.fetchone()[0]


@transactional()
def delete_all_for_experiment(cursor: Cursor, experiment_id: int):
    cursor.execute(
        '''
            DELETE FROM observation
            WHERE experiment_id=%(experiment_id)s;
        ''',
        {'experiment_id': experiment_id}
    )
