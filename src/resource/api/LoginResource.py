from flask import request, make_response, redirect, url_for
from flask_restful import Resource
from src.utils import authentication
from src.service import user_service


class LoginResource(Resource):

    def post(self):
        body = request.json
        if 'username' not in body:
            return 'No username supplied!', 400

        if 'password' not in body:
            return 'No password supplied', 400

        username = body.get('username')
        password = body.get('password')

        user = user_service.find_by_name(username)

        if user is None:
            return "No user with that name exists!", 400

        is_password_correct = user_service.check_password(
            user.id,
            password
        )

        if not is_password_correct:
            return "Incorrect password", 401

        token = authentication.generate_user_token(user.id)

        response = make_response(token)
        response.set_cookie('X-AUTH-TOKEN', token, httponly=True, secure=True, samesite='Strict')
        response.headers['X-AUTH-TOKEN'] = token
        return response
