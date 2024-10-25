from src.database.db import transactional
from sqlite3 import Cursor
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
            FROM user
            WHERE id=:id;
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
            FROM user
            WHERE name=:name;
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
            INSERT INTO user (name)
            VALUES (:name);
        ''',
        {'name': name}
    )
    user_id = cursor.lastrowid
    return user_id
