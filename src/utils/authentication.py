import bcrypt
from datetime import datetime, timedelta
from flask import request, abort
from functools import wraps
import jwt
from src.entity.Payload import Payload

SECRET_KEY = "DANK"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def check_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


def generate_user_token(user_id: int) -> str:
    payload = {
        'id': user_id,
        'resource': 'user',
        'exp': datetime.utcnow() + timedelta(hours=12),
    }

    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def authenticate_token(token: str) -> Payload:
    payload: dict = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    return Payload(id=payload['id'], resource=payload['resource'], exp=payload['exp'])


def authenticate(endpoint_type):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if endpoint_type == 'template':
                X_AUTH_TOKEN = request.cookies.get('X-AUTH-TOKEN')
            elif endpoint_type == 'api':
                X_AUTH_TOKEN = request.headers.get('X-AUTH-TOKEN')
            else:
                raise SystemExit('Invalid type for authenticate decorator!')
            print(X_AUTH_TOKEN)
            print(request.cookies)
            if X_AUTH_TOKEN is None:
                abort(401)  # No auth token

            try:
                payload: Payload = authenticate_token(X_AUTH_TOKEN)
                kwargs['payload'] = payload
            except jwt.exceptions.InvalidSignatureError:
                return abort(401)  # invalid auth token

            return f(*args, **kwargs)

        return decorated_function

    return decorator


if __name__ == '__main__':
    token = generate_user_token(1)
    print(authenticate_token(token))
    try:
        print(authenticate_token(token + "1"))
    except jwt.exceptions.InvalidSignatureError:
        print("Invalid token")
