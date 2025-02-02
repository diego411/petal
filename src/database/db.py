import psycopg2
from psycopg2.extensions import cursor as Cursor
from functools import wraps
from src.AppConfig import AppConfig
from typing import List
from pathlib import Path


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


def get_all_migrations_scripts() -> List[str]:
    migrations_folder = Path('./src/database/migrations')
    migrations = []

    up_script = migrations_folder / 'up.sql'
    assert up_script.exists(), 'Invalid state: no up.sql found!'

    with up_script.open('r') as file:
        migrations.append(file.read())

    for sql_file in migrations_folder.glob("*.sql"):
        if sql_file.name == 'up.sql':
            continue

        with sql_file.open('r') as file:
            sql_content = file.read()
            migrations.append(sql_content)

    return migrations


@transactional()
def run_migrations(cursor: Cursor):
    migrations = get_all_migrations_scripts()

    for migration in migrations:
        cursor.execute(migration)


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
