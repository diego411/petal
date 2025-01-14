import bcrypt
from datetime import datetime, timedelta

import jwt

SECRET_KEY = "DANK"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def check_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


def generate_user_token(user_id: int) -> str:
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=12)
    }

    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def authenticate_token(token: str) -> dict:
    payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    return payload


if __name__ == '__main__':
    token = generate_user_token(1)
    print(authenticate_token(token))
    try:
        print(authenticate_token(token + "1"))
    except jwt.exceptions.InvalidSignatureError:
        print("Invalid token")
