from flask_restful import Resource
from flask import request
from src.utils.authentication import authenticate
from src.entity.Payload import Payload
from src.entity.RecordingState import RecordingState
from src.service import experiment_service, recording_service
from datetime import datetime


class ExperimentActionResource(Resource):

    @authenticate('api')
    def post(self, experiment_id: int, action: str, payload: Payload):
        print(action)
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"
        experiment = experiment_service.find_by_id(experiment_id)

        body = request.json

        if experiment is None:
            return f"No experiment with id: {experiment_id} found!", 404

        if action == 'start':
            if experiment.status == 'RUNNING':
                return 'Experiment is already running', 400

            if experiment.status == 'FINISHED':
                return 'Experiment is already finished. It can\'t be started', 400

            if 'recording_id' not in body:
                return 'No recording id supplied!', 400

            recording = recording_service.find_by_id(body['recording_id'])

            if recording is None:
                return f"No recording with id: {body['recording']} found!", 404

            if recording.state == RecordingState.RUNNING:
                return 'Recording started already!', 400

            recording_service.start(recording)
            experiment_service.set_recording(experiment_id, recording.id)
            experiment_service.set_status(experiment_id, 'RUNNING')
            experiment_service.set_started_at(experiment_id, datetime.now())

            return "Successfully started experiment", 200
        elif action == 'stop':
            if experiment.status == 'REGISTERED':
                return 'Experiment has not been started yet!', 400

            if experiment.status == 'FINISHED':
                return 'Experiment is already Finished. It can\'t be stopped!', 400

            if 'emotions' not in body:
                return 'No emotion data supplied.', 400

            emotions = body['emotions']

            if 'recording_id' not in body:
                return 'No recording id supplied!', 400

            recording = recording_service.find_by_id(body['recording_id'])

            if recording.state != RecordingState.RUNNING:
                return f'The data collection for the recording has not started yet', 400

            recording_service.stop_and_label(recording, emotions)
            experiment_service.set_status(experiment_id, 'FINISHED')

            return "Sucessfully stopped experiment", 200

        return f"There is no \"{action}\" action for the experiment resource", 404
