from src.database.db import transactional
from psycopg2.extensions import cursor as Cursor
from datetime import datetime
from typing import List


@transactional()
def get_values_for_recording(cursor: Cursor, recording: int, limit: int = None) -> List[float]:
    if limit is None:
        cursor.execute(
            '''
                SELECT value
                FROM measurement
                WHERE recording=%(recording)s 
                ORDER BY created_at DESC;
            ''',
            {'recording': recording}
        )
    else:
        cursor.execute(
            f'''
                SELECT value
                FROM measurement
                WHERE recording=%(recording)s
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
            WHERE recording=%(recording)s;
        ''',
        {'recording': recording}
    )

    return cursor.fetchone()[0]


@transactional()
def insert_many(cursor: Cursor, recording: int, measurements: list, created_at: datetime):
    data = list(map(
        lambda measurement: {'value': measurement, 'recording': recording, 'created_at': created_at},
        measurements
    ))

    cursor.executemany(
        '''
            INSERT INTO measurement (value, recording, created_at)
            VALUES (%(value)s, %(recording)s, %(created_at)s);
        ''',
        data
    )


@transactional()
def delete_for_recording(cursor: Cursor, recording_id: int):
    cursor.execute(
        '''
            DELETE from measurement
            WHERE recording=%(recording_id)s;
        ''',
        {'recording_id': recording_id}
    )
