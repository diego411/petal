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
                name TEXT NOT NULL,
                sample_rate INTEGER,
                threshold INTEGER
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
                start_time DATETIME NOT NULL,
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
                FOREIGN KEY (recording) REFERENCES recording(id)
            );
        '''
    )

    connection.commit()
    connection.close()


def create_user(name: str):
    user = get_user_by_name(name)
    if user is not None:
        return

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            INSERT INTO user (name)
            VALUES (:name);
        ''',
        {'name': name}
    )

    connection.commit()
    connection.close()


def create_recording(name: str, state: str, sample_rate: int, start_time: datetime.datetime):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            INSERT INTO recording (name, state, sample_rate, start_time)
            VALUES (:name, :state, :start_time);
        ''',
        {'name': name, 'state': state, 'sample_rate': sample_rate, 'start_time': start_time}
    )

    connection.commit()
    connection.close()


def create_measurement(value: float, recording: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            INSERT INTO measurement (value, recording)
            VALUES (:value, :recording);
        ''',
        {'value': value, 'recording': recording}
    )

    connection.commit()
    connection.close()


def set_sample_rate(name: str, sample_rate: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            UPDATE user
            SET sample_rate=:sample_rate
            WHERE name=:name;
        ''',
        {'name': name, 'sample_rate': sample_rate}
    )

    connection.commit()
    connection.close()


def set_threshold(name: str, threshold: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            UPDATE user
            SET threshold=:threshold
            WHERE name=:name;
        ''',
        {'name': name, 'threshold': threshold}
    )

    connection.commit()
    connection.close()


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
        'sample_rate': result[2],
        'threshold': result[3]
    }


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
            'start_time': recording[5]
        },
        result
    )
