from flask_restful import Resource
from flask import request
from src.service import experiment_service, user_service
from src.utils.authentication import authenticate
from src.entity import Payload


class ExperimentResource(Resource):

    @authenticate('api')
    def post(self, payload: Payload):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"

        data = request.json

        if 'name' not in data:
            return "Name is required", 400

        name = data['name']
        user = user_service.find_by_id(payload.id)

        if name == '':
            return "Name cannot be empty", 400

        experiment_id = experiment_service.create(name, user.id)

        return {'id': experiment_id}, 201
