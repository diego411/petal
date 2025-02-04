from flask import request
from flask_restful import Resource
from src.service import recording_service, user_service, experiment_service
from src.entity.RecordingState import RecordingState
from flask import current_app
from src.utils.authentication import authenticate
from src.entity.Payload import Payload
from src.entity.User import User


class RecordingResource(Resource):

    def __init__(self):
        self.socketio = current_app.socketio

    @authenticate(endpoint_type='api')
    def get(self, recording_id: int = None, payload: Payload = None):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"

        if recording_id is None:
            return recording_service.get_all(user=payload.id), 200

        recording = recording_service.find_by_id(recording_id)

        if recording is None:
            return {
                'error': 'Not found',
                'message': f'No recording with id {recording_id} found'
            }, 404

        if recording.user != payload.id:
            return {
                'error': 'Not authorized',
                'message': 'You are not authorized to access this recording'
            }, 401

        return recording_service.to_dto(recording), 200

    @authenticate('api')
    def post(self, payload: Payload):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"
        data = request.json

        if 'recording' not in data:
            return 'Field \"recording\" in request body is required', 400

        recording_name: str = data['recording']
        user: User = user_service.find_by_id(payload.id)

        sample_rate = data['sample_rate']
        threshold = data['threshold']

        recording = recording_service.find_not_stopped_by_user_and_name(user.id, recording_name)
        if recording is not None and (
                recording.state == RecordingState.RUNNING or recording.state == RecordingState.REGISTERED):
            return {'id': recording.id}, 200

        recording_id = recording_service.create(
            recording_name,
            user.id,
            RecordingState.REGISTERED,
            sample_rate,
            threshold,
        )

        current_app.socketio.emit('recording-register', {
            'id': recording_id
        })

        return {'id': recording_id}, 201

    @authenticate('api')
    def delete(self, recording_id, payload: Payload):
        recording = recording_service.find_by_id(recording_id)
        if recording is None:
            return "No recording with given id found", 404

        if recording.user != payload.id:
            return "You are not authorized to delete this recording", 401

        experiment = experiment_service.find_by_recording(recording.id)

        if experiment is not None:
            return {
                'error': 'An error occurred when stopping the recording',
                'message': f'This recording is linked to <a href="/experiment/{experiment.id}">this</a> experiment you can\'t manually stop it!'
            }, 400

        recording_service.delete(recording_id)

        self.socketio.emit('recording-delete', {
            'id': recording.id,
            'name': recording.name
        })

        return f'Deleted recording with id {recording_id}', 200
