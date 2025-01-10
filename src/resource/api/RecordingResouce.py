from flask import request
from flask_restful import Resource
from src.service import recording_service, user_service
from src.entity.RecordingState import RecordingState
from flask import current_app


class RecordingResource(Resource):

    def __init__(self):
        self.socketio = current_app.socketio
        print("SOCKETIO: ", self.socketio)

    def post(self):
        data = request.json

        if 'recording' not in data:
            return 'Field \"recording\" in request body is required', 400

        recording_name = data['recording']

        user_name = request.headers.get('User-Name')
        if user_name is None:
            return 'User-Name header needs to be provided', 400

        user = user_service.get_or_create(user_name)

        sample_rate = data['sample_rate']
        threshold = data['threshold']

        recording = recording_service.find_not_stopped_by_user_and_name(user.id, recording_name)
        if recording is not None and (
                recording.state == RecordingState.RUNNING or recording.state == RecordingState.REGISTERED):
            return {'id': recording.id}, 201

        recording_id = recording_service.create(
            recording_name,
            user.id,
            RecordingState.REGISTERED,
            sample_rate,
            threshold,
        )

        return {'id': recording_id}, 201

    def delete(self, recording_id):
        recording = recording_service.find_by_id(recording_id)
        recording_service.delete(recording_id)

        self.socketio.emit('recording-delete', {
            'id': recording.id,
            'name': recording.name
        })

        return f'Deleted recording with id {recording_id}', 200
