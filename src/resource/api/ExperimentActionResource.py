from flask_restful import Resource
from flask import request
from src.utils.authentication import authenticate
from src.entity.Payload import Payload
from src.entity.RecordingState import RecordingState
from src.service import experiment_service, recording_service
from datetime import datetime
from src.entity.Experiment import Experiment


def start(experiment: Experiment):
    if experiment.status == 'RUNNING':
        return {
            'error': 'Bad request',
            'message': 'Experiment is already running!'
        }, 400

    if experiment.status == 'FINISHED':
        return {
            'error': 'Bad request',
            'message': 'Experiment is already finished. It can\'t be started!'
        }, 400

    body = request.json

    if 'recording' not in body:
        return {
            'error': 'Bad request',
            'message': 'No recording id supplied!'
        }, 400

    recording_id = body['recording']
    recording = recording_service.find_by_id(recording_id)

    if recording is None:
        return {
            'error': 'Not found',
            'message': f"No recording with id: {recording_id} found!"
        }, 404

    if recording.state == RecordingState.STOPPED:
        return {
            'error': 'Bad request',
            'message': 'The provided recording is already stopped!'
        }, 400

    experiment_service.set_recording(experiment.id, recording.id)

    if recording.state == RecordingState.REGISTERED:
        recording_service.start(recording)

    experiment_service.set_status(experiment.id, 'RUNNING')
    experiment_service.set_started_at(experiment.id, datetime.now())

    return "Successfully started experiment", 200


def stop(experiment: Experiment):
    if experiment.status == 'REGISTERED':
        return 'Experiment has not been started yet!', 400

    if experiment.status == 'FINISHED':
        return 'Experiment is already Finished. It can\'t be stopped!', 400

    body = request.json

    if 'emotions' not in body:
        return 'No emotion data supplied.', 400

    emotions: list = body['emotions']
    if 'recording_id' not in body:
        return 'No recording id supplied!', 400

    recording = recording_service.find_by_id(body['recording_id'])

    if recording.state != RecordingState.RUNNING:
        return f'The data collection for the recording has not started yet', 400

    experiment_service.set_status(experiment.id, 'FINISHED')
    recording_service.stop_and_label(experiment, recording, emotions)

    return "Successfully stopped experiment", 200


class ExperimentActionResource(Resource):

    @authenticate('api')
    def post(self, experiment_id: int, action: str, payload: Payload):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"
        experiment = experiment_service.find_by_id(experiment_id)

        if experiment.user != payload.id:
            return {
                'error': 'Not authorized',
                'message': 'You are not authorized to execute an action on this experiment!'
            }, 401

        if experiment is None:
            return {
                'error': 'Not found',
                'message': f"No experiment with id: {experiment_id} found!"
            }, 404

        if action == 'start':
            return start(experiment)
        elif action == 'stop':
            return stop(experiment)

        return f"There is no \"{action}\" action for the experiment resource", 404
