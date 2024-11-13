import psycopg2
from psycopg2.extensions import cursor as Cursor
from functools import wraps
from src.AppConfig import AppConfig


def get_connection():
    host = 'localhost' if AppConfig.PROFILE == 'dev' else 'postgres'
    return psycopg2.connect(
        dbname=AppConfig.POSTGRES_DB,
        user=AppConfig.POSTGRES_USER,
        password=AppConfig.POSTGRES_PASSWORD,
        host=host,
        port='5432'
    )


def transactional():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            conn = get_connection()
            cursor = conn.cursor()
            try:
                result = func(cursor, *args, **kwargs)
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"An error occurred: {e}")
                raise e
            finally:
                cursor.close()
                conn.close()

            return result

        return wrapper

    return decorator


@transactional()
def init_tables(cursor: Cursor):
    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL
            );
        '''
    )

    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS recording (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                state INTEGER NOT NULL,
                sample_rate INTEGER NOT NULL,
                threshold INTEGER NOT NULL,
                start_time TIMESTAMP,
                last_update TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        '''
    )

    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS measurement (
                id SERIAL,
                value REAL,
                recording INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (recording) REFERENCES recording(id) ON DELETE CASCADE,
                PRIMARY KEY (id, created_at)
            ) PARTITION BY RANGE (created_at);
        '''
    )


@transactional()
def create_measurement_partition(cursor, offset: int = None):
    interval = ''
    if offset is not None:
        interval = f"+ INTERVAL '{offset} day'"
    cursor.execute(
        f'''
            DO $$
            DECLARE
                partition_name TEXT;
                partition_date DATE;
            BEGIN
                partition_date := CURRENT_DATE {interval};
                partition_name := 'measurements_' || to_char(partition_date, 'YYYY_MM_DD');
                EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF measurement FOR VALUES FROM (%L) TO (%L)',
                           partition_name,
                           partition_date,
                           partition_date + INTERVAL '1 day');
        END $$;
        '''
    )
    print("Creating partition with offset", offset)
