import sqlite3
import datetime


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect('dbs/plant.db')


def init():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
        '''
    )

    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS recording (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                user INTEGER NOT NULL,
                state TEXT NOT NULL,
                sample_rate INTEGER NOT NULL,
                threshold INTEGER NOT NULL,
                start_time DATETIME,
                last_update DATETIME,
                FOREIGN KEY (user) REFERENCES user (id)
            );
        '''
    )

    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS measurement (
                id INTEGER PRIMARY KEY,
                value REAL,
                recording INTEGER NOT NULL,
                created_at DATETIME,
                FOREIGN KEY (recording) REFERENCES recording(id) ON DELETE CASCADE
            );
        '''
    )

    connection.commit()
    connection.close()


def create_user(name: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            INSERT INTO user (name)
            VALUES (:name);
        ''',
        {'name': name}
    )

    user_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return user_id


def create_recording(name: str, user: int, state: str, sample_rate: int, threshold: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            INSERT INTO recording (name, user, state, sample_rate, threshold)
            VALUES (:name, :user, :state, :sample_rate, :threshold);
        ''',
        {'name': name, 'user': user, 'state': state, 'sample_rate': sample_rate, 'threshold': threshold}
    )

    recording_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return recording_id


def set_last_update(recording: int, date: datetime.datetime):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            UPDATE recording
            SET last_update=:last_update
            WHERE id=:id
        ''',
        {'id': recording, 'last_update': date}
    )
    connection.commit()
    connection.close()


def set_start_time(recording: int, date: datetime.datetime):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            UPDATE recording
            SET start_time=:start_time
            WHERE id=:id
        ''',
        {'id': recording, 'start_time': date}
    )

    connection.commit()
    connection.close()


def set_state(recording: int, state: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
            UPDATE recording
            SET state=:state
            WHERE id=:id;
        """,
        {'id': recording, 'state': state}
    )

    connection.commit()
    connection.close()


def get_or_create_user(name: str):
    user = get_user_by_name(name)
    if user is not None:
        return user

    user_id = create_user(name)
    return get_user_by_id(user_id)


def get_user_by_id(user_id: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            SELECT *
            FROM user
            WHERE id=:id;
        ''',
        {'id': user_id}
    )

    result = cursor.fetchone()
    if result is None:
        return

    connection.close()
    return {
        'id': result[0],
        'name': result[1]
    }


def get_user_by_name(name: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            SELECT *
            FROM user
            WHERE name=:name;
        ''',
        {'name': name}
    )
    result = cursor.fetchone()
    if result is None:
        return

    connection.close()
    return {
        'id': result[0],
        'name': result[1],
    }


def get_recording_by_id(recording_id: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE id=:id;
        ''',
        {'id': recording_id}
    )

    result = cursor.fetchone()
    if result is None:
        return

    connection.close()
    return {
        'id': result[0],
        'name': result[1],
        'user': result[2],
        'state': result[3],
        'sample_rate': result[4],
        'threshold': result[5],
        'start_time': parse_sql_date(result[6]),
        'last_update': parse_sql_date(result[7])
    }


def get_recording(user: int, name: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE user=:user AND name=:name
        ''',
        {'user': user, 'name': name}
    )

    result = cursor.fetchone()
    if result is None:
        return

    connection.close()
    return {
        'id': result[0],
        'name': result[1],
        'user': result[2],
        'state': result[3],
        'sample_rate': result[4],
        'threshold': result[5],
        'start_time': parse_sql_date(result[6]),
        'last_update': parse_sql_date(result[7])
    }


def get_recordings_by_state(state: str) -> list:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE state=:state
        ''',
        {'state': state}
    )

    result = cursor.fetchall()
    connection.close()

    if result is None:
        return []

    return list(map(
        lambda recording: {
            'id': recording[0],
            'name': recording[1],
            'user': recording[2],
            'state': recording[3],
            'sample_rate': recording[4],
            'threshold': recording[5],
            'start_time': parse_sql_date(recording[6]),
            'last_update': parse_sql_date(recording[7])
        },
        result
    ))


def get_recordings_for_user(user: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            SELECT *
            FROM recording
            WHERE user=:user;
        ''',
        {'user': user}
    )

    result = cursor.fetchall()
    if result is None:
        return

    connection.close()
    return map(
        lambda recording: {
            'id': recording[0],
            'name': recording[1],
            'user': recording[2],
            'state': recording[3],
            'sample_rate': recording[4],
            'threshold': recording[5],
            'start_time': parse_sql_date(recording[6]),
            'last_update': parse_sql_date(recording[7])
        },
        result
    )


def get_measurements_for_recording(recording: int, limit: int = None) -> list:
    connection = get_connection()
    cursor = connection.cursor()

    if limit is None:
        cursor.execute(
            '''
                SELECT value
                FROM measurement
                ORDER BY created_at DESC
                WHERE recording=:recording; 
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


def get_number_of_measurements_for_recording(recording: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            SELECT COUNT(*)
            FROM measurement
            WHERE recording=:recording;
        ''',
        {'recording': recording}
    )

    return cursor.fetchone()[0]


def add_measurements(recording: int, measurements: list, created_at: datetime.datetime):
    datetime.datetime.now()
    connection = get_connection()
    cursor = connection.cursor()

    for measurement in measurements:
        cursor.execute(
            '''
                INSERT INTO measurement (value, recording, created_at)
                VALUES (:value, :recording, :created_at);
            ''',
            {'value': measurement, 'recording': recording, 'created_at': created_at}
        )

    connection.commit()
    connection.close()


def delete_recording(recording: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
            DELETE FROM recording
            WHERE id=:id;
        """,
        {'id': recording}
    )

    connection.commit()
    connection.close()


def parse_sql_date(date: str) -> datetime.datetime:
    if date is None:
        return None

    return datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
