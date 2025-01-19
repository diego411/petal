from flask import make_response, current_app
from flask_restful import Resource
from src.utils.authentication import authenticate
from src.entity.Payload import Payload


class LogoutResource(Resource):

    @authenticate(endpoint_type='api')
    def post(self, payload: Payload):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"

        response = make_response("Logged out successfully")
        response.set_cookie('X-AUTH-TOKEN', '', max_age=0, httponly=True, secure=True, samesite='Strict')
        current_app.logger.info(f'Logged out user with id {payload.id}')

        return response
