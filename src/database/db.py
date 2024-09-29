import sqlite3


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect('dbs/plant.db')


def init():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                sample_rate INTEGER,
                threshold INTEGER
            )
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
            INSERT INTO users (name)
            VALUES (:name)
        ''',
        {'name': name}
    )

    connection.commit()
    connection.close()


def set_sample_rate(name: str, sample_rate: int):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
            UPDATE users
            SET sample_rate=:sample_rate
            WHERE name=:name
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
            UPDATE users
            SET threshold=:threshold
            WHERE name=:name
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
            FROM users
            WHERE name=:name
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
