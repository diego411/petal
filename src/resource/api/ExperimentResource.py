from flask_restful import Resource
from flask import request
from src.service import experiment_service, user_service, observation_service
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

    @authenticate('api')
    def delete(self, experiment_id: int, payload: Payload):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"

        experiment = experiment_service.find_by_id(experiment_id)

        if experiment is None:
            return {
                'error': 'Not found',
                'message': f'No experiment with id {experiment_id} found!'
            }, 404

        observation_service.delete_all_for_experiment(experiment_id)
        experiment_service.delete(experiment_id)

        return {
            'success': 'Deleted',
            'message': f'Successfully delete experiment with id {experiment_id}'
        }, 200
