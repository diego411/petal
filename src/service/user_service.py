from src.database.db import transactional
from psycopg2.extensions import cursor as Cursor
from src.entity.User import User
from typing import Optional
from src.utils import authentication


def get_or_create(name: str) -> User:
    user = find_by_name(name)
    if user is not None:
        return user

    user_id = create(name)
    return find_by_id(user_id)


@transactional()
def find_by_id(cursor: Cursor, user_id: int) -> Optional[User]:
    cursor.execute(
        '''
            SELECT *
            FROM users
            WHERE id=%(id)s;
        ''',
        {'id': user_id}
    )

    result = cursor.fetchone()
    if result is None:
        return None
    return User.from_(result)


@transactional()
def find_by_name(cursor: Cursor, name: str) -> Optional[User]:
    cursor.execute(
        '''
            SELECT *
            FROM users
            WHERE name=%(name)s;
        ''',
        {'name': name}
    )
    result = cursor.fetchone()
    if result is None:
        return None

    return User.from_(result)


@transactional()
def create(cursor: Cursor, name: str) -> int:
    cursor.execute(
        '''
            INSERT INTO users (name)
            VALUES (%(name)s)
            RETURNING id;
        ''',
        {'name': name}
    )
    user_id = cursor.fetchone()[0]
    return user_id


@transactional()
def get_password_hash(cursor: Cursor, user_id: int) -> str:
    cursor.execute(
        '''
            SELECT password_hash
            FROM users
            WHERE id=%(user_id)s;
        ''',
        {'user_id': user_id}
    )
    return cursor.fetchone()[0]


def check_password(user_id: int, password: str):
    password_hash = get_password_hash(user_id)
    return authentication.check_password(password, password_hash)


if __name__ == '__main__':
    print(check_password(1, 'test2'))
