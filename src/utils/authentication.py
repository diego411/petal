import bcrypt
from datetime import datetime, timedelta
from flask import request, abort, current_app
from functools import wraps
import jwt
from src.entity.Payload import Payload
from src.entity.exception.UnauthorizedException import UnauthorizedException
from src.AppConfig import AppConfig

SECRET_KEY = AppConfig.JWT_SECRET_KEY


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

    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    current_app.logger.info(f"Generated following token for user with id {user_id}: {token}")
    return token


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
                X_AUTH_TOKEN = request.headers.get('X-AUTH-TOKEN') or request.headers.get('X-Auth-Token')
                if X_AUTH_TOKEN is None:
                    X_AUTH_TOKEN = request.cookies.get('X-AUTH-TOKEN')
            else:
                raise SystemExit('Invalid type for authenticate decorator!')

            if X_AUTH_TOKEN is None:  # no auth token
                if endpoint_type == 'template':
                    abort(401)
                elif endpoint_type == 'api':
                    raise UnauthorizedException('api', 'No auth token supplied!')

            try:
                payload: Payload = authenticate_token(X_AUTH_TOKEN)
                kwargs['payload'] = payload
            except (jwt.exceptions.InvalidSignatureError, jwt.exceptions.DecodeError) as e:  # invalid auth token
                current_app.logger.info(f"Request on endpoint of type {endpoint_type} raised following error: {e}")
                if endpoint_type == 'template':
                    return abort(401)
                elif endpoint_type == 'api':
                    raise UnauthorizedException('api', 'Invalid auth token supplied!')

            return f(*args, **kwargs)

        return decorated_function

    return decorator
