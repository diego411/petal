from flask_restful import Resource
from flask import request
from src.utils.authentication import authenticate
from src.entity.Payload import Payload
from src.service import observation_service, experiment_service
from datetime import datetime
from typing import Optional


class ObservationResource(Resource):

    @authenticate(endpoint_type='api')
    def post(self, payload: Payload):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"

        body = request.json

        if 'label' not in body:
            return {
                'error': 'Bad request',
                'message': 'Field "label" needs to be supplied'
            }, 400

        if 'timestamp' not in body:
            return {
                'error': 'Bad request',
                'message': 'Field "timestamp" needs to be supplied'
            }, 400

        if 'experiment' not in body:
            return {
                'error': 'Bad request',
                'message': 'Field "experiment" needs to be supplied'
            }, 400

        label = body['label']
        timestamp = body['timestamp']
        experiment_id = body['experiment']

        experiment = experiment_service.find_by_id(experiment_id)

        if experiment is None:
            return {
                'error': 'Not found',
                'message': f'No experiment with id {experiment_id} found!'
            }, 404

        if experiment.user != payload.id:
            return {
                'error': 'Not authorized',
                'message': 'You are not authorized to create an observation for that experiment!'
            }, 401

        try:
            observed_at: datetime = datetime.fromtimestamp(timestamp)
        except ValueError:
            try:
                observed_at: datetime = datetime.fromtimestamp(timestamp / 1000)
            except ValueError as e:
                return {
                    'error': 'Bad request',
                    'message': f'Provided timestamp could not be parsed: {e}'
                }, 400

        observation_id = observation_service.create(label, observed_at, experiment_id)

        return {
            'success': 'Created',
            'id': observation_id,
            'message': f'Successfully created observation with id {observation_id}'
        }, 201
