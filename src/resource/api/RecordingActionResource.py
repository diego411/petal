from flask_restful import Resource
from flask import request
import threading
from src.service import recording_service
from src.entity.Recording import Recording
from src.entity.RecordingState import RecordingState
import datetime
from flask_socketio import SocketIO
import logging
import dropbox
from flask import current_app


class RecordingActionResource(Resource):

    def __init__(self):
        self.socketio: SocketIO = current_app.socketio
        self.logger: logging.Logger = current_app.logger
        self.dropbox_client: dropbox.Dropbox = current_app.dropbox_client

    def post(self, recording_id: str, action: str):
        recording = recording_service.find_by_id(recording_id)

        if recording is None:
            return f"No recording with id: {recording_service} found!", 404

        if action == 'start':
            return self.start(recording)
        elif action == 'update':
            return self.update(recording)
        elif action == 'stop':
            return self.stop(recording)
        elif action == 'delete':
            return self.delete(recording)
        elif action == "stopAndLabel":
            return self.stop_and_label(recording)

        return f"The action: {action} is not supported for the recording resource", 400

    def start(self, recording: Recording):
        if recording.state == RecordingState.RUNNING:
            return 'Recording for this user started already. Stop it first', 400

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

        recording_service.stop(recording)

        return f'Data collection for recording with id {recording.id} successfully stopped and file saved.', 200

    def delete(self, recording: Recording):
        recording_service.delete(recording.id)

        self.socketio.emit('recording-delete', {
            'id': recording.id,
            'name': recording.name
        })

        return f'Deleted recording with id {recording.id}', 200

    def stop_and_label(self, recording: Recording):
        emotions = request.json
        if emotions is None:
            return 'No emotion data supplied.', 400

        if recording.state != RecordingState.RUNNING:
            return f'The data collection for the recording has not started yet', 400

        recording_service.stop_and_label(recording, emotions)

        return "Successfully stopped recording and labeled data", 200
