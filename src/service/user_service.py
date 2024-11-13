from src.database.db import transactional
from psycopg2.extensions import cursor as Cursor
from src.entity.User import User
from typing import Optional


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
    print(user_id)
    return user_id
