from src.database.db import transactional
from sqlite3 import Cursor
from datetime import datetime
from typing import List


@transactional()
def get_values_for_recording(cursor: Cursor, recording: int, limit: int = None) -> List[float]:
    if limit is None:
        cursor.execute(
            '''
                SELECT value
                FROM measurement
                WHERE recording=:recording 
                ORDER BY created_at DESC;
            ''',
            {'recording': recording}
        )
    else:
        cursor.execute(
            f'''
                SELECT value
                FROM measurement
                WHERE recording=:recording
                ORDER BY created_at DESC
                LIMIT {limit}; 
            ''',
            {'recording': recording}
        )

    result = cursor.fetchall()

    if result is None:
        return []

    return [row[0] for row in result]


@transactional()
def get_count(cursor: Cursor, recording: int) -> int:
    cursor.execute(
        '''
            SELECT COUNT(*)
            FROM measurement
            WHERE recording=:recording;
        ''',
        {'recording': recording}
    )

    return cursor.fetchone()[0]


@transactional()
def insert_many(cursor: Cursor, recording: int, measurements: list, created_at: datetime):
    for measurement in measurements:
        cursor.execute(  # TODO: exeutemany
            '''
                INSERT INTO measurement (value, recording, created_at)
                VALUES (:value, :recording, :created_at);
            ''',
            {'value': measurement, 'recording': recording, 'created_at': created_at}
        )
