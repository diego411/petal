import sqlite3
from sqlite3 import Cursor
import datetime
from functools import wraps


def transactional():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with sqlite3.connect('dbs/plant.db') as conn:
                cursor = conn.cursor()
                try:
                    result = func(cursor, *args, **kwargs)
                    conn.commit()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    raise e

                return result

        return wrapper

    return decorator


@transactional()
def init(cursor: Cursor):
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
                state INTEGER NOT NULL,
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


def parse_sql_date(date: str) -> datetime.datetime:
    if date is None:
        return None

    return datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
