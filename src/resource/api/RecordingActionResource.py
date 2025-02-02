from flask_restful import Resource
from flask import request
import threading
from src.service import recording_service, experiment_service
from src.entity.Recording import Recording
from src.entity.Payload import Payload
from src.entity.RecordingState import RecordingState
import datetime
from flask_socketio import SocketIO
import logging
import dropbox
from flask import current_app
from src.utils.authentication import authenticate


class RecordingActionResource(Resource):

    def __init__(self):
        self.socketio: SocketIO = current_app.socketio
        self.logger: logging.Logger = current_app.logger
        self.dropbox_client: dropbox.Dropbox = current_app.dropbox_client

    @authenticate('api')
    def post(self, recording_id: str, action: str, payload: Payload):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"
        recording = recording_service.find_by_id(recording_id)

        if recording is None:
            return f"No recording with id: {recording_id} found!", 404

        if recording.user != payload.id:
            return "You are not authorized to execute an action on this recording", 401

        if action == 'start':
            return self.start(recording)
        elif action == 'update':
            return self.update(recording)
        elif action == 'stop':
            return self.stop(recording)

        return f"The is not \"{action}\" action for the recording resource", 404

    def start(self, recording: Recording):
        if recording.state == RecordingState.RUNNING:
            return 'Recording started already!', 400

        recording_service.start(recording)

        return f'Successfully started data collection for recording', 200

    # This should be a PATCH on /recording however NanoPy does not support PATCH :]
    def update(self, recording: Recording):
        now = datetime.datetime.now()

        if recording.state != RecordingState.RUNNING:
            return f'The data collection for the recording has not started yet', 400

        data: bytes = request.data
        thread = threading.Thread(target=recording_service.run_update(recording, data, now))
        thread.start()
        return f'Successfully started update for: {recording.name}', 202

    def stop(self, recording: Recording):
        recording_state = recording.state
        if recording_state == RecordingState.REGISTERED:
            return 'This recording has not been started yet. It cannot be stopped.', 400

        if recording_state == RecordingState.STOPPED:
            return 'This recording is already stopped.', 400

        experiment = experiment_service.find_by_recording(recording.id)

        if experiment is not None:
            return {
                'error': 'An error occurred when stopping the recording',
                'message': f'This recording is linked to <a href="/experiment/{experiment.id}">this</a> experiment you can\'t manually stop it! Finish the experiment instead!'
            }, 400

        try:
            shared_link: str = recording_service.stop(recording)
        except Exception as e:
            print(str(e))
            return {'error': 'An error occurred when stopping the recording', 'message': str(e)}, 500

        return {
            'message': f'Data collection for recording (#{recording.id}) successfully stopped and file saved',
            'shared_link': shared_link
        }, 200
