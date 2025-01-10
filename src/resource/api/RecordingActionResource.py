from flask_restful import Resource
from flask import request
from src.controller import dropbox_controller
import os
import threading
from src.service import recording_service, user_service
from src.entity.Recording import Recording
from src.entity.RecordingState import RecordingState
from src.service import measurement_service
from src.service import wav_converter, labeler
import datetime
from src.AppConfig import AppConfig
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
        print(action)
        print(recording_id)
        recording = recording_service.find_by_id(recording_id)
        if action == 'start':
            if recording.state == RecordingState.RUNNING:
                return 'Recording for this user started already. Stop it first', 400

            now = datetime.datetime.now()
            recording_service.set_last_update(recording_id, now)
            recording_service.set_start_time(recording_id, now)
            recording_service.set_state(recording_id, RecordingState.RUNNING)

            self.socketio.emit('recording-start', {
                'name': recording.name,
                'id': recording.id,
                'start_time': now.strftime('%d.%m.%Y %H:%M:%S'),
            })

            return f'Successfully started data collection for recording', 200
        elif action == 'update':
            recording_id = int(recording_id)
            now = datetime.datetime.now()

            recording = recording_service.find_by_id(recording_id)
            if recording is None:
                return f"Recording with id {recording_id} not found.", 404

            if recording.state != RecordingState.RUNNING:
                return f'The data collection for the recording has not started yet', 400

            data: bytes = request.data
            thread = threading.Thread(target=self.run_update(recording, data, now))
            thread.start()
            return f'Successfully started update for: {recording.name}', 200
        elif action == 'stop':
            user = user_service.find_by_id(recording.user)

            recording_state = recording.state
            if recording_state == RecordingState.REGISTERED:
                return 'This recording has not been started yet. It cannot be stopped.', 400

            if recording_state == RecordingState.STOPPED:
                return 'This recording is already stopped.', 400

            measurements = measurement_service.get_values_for_recording(recording_id)
            len_measurements = len(measurements)
            start_time = recording.start_time
            last_update = recording.last_update
            delta_seconds = (last_update - start_time).seconds
            calculated_sample_rate = int(len_measurements / delta_seconds) if delta_seconds != 0 else 0
            self.logger.info(
                f"""
                    Stopping recording with id {recording_id}. 
                    Start time: {start_time}. 
                    Last updated: {last_update}'.
                    Delta seconds: {delta_seconds}.
                    Number of measurements: {len_measurements}.
                    Calculated sample rate: {calculated_sample_rate} 
                """
            )

            file_name = f'{recording.name}_{calculated_sample_rate}Hz_{int(start_time.timestamp() * 1000)}.wav'
            file_path = wav_converter.convert(
                measurements,
                sample_rate=calculated_sample_rate,
                path=f'audio/{file_name}'
            )

            dropbox_controller.upload_file_to_dropbox(
                dropbox_client=self.dropbox_client,
                file_path=file_path,
                dropbox_path=f'/PlantRecordings/{user.name}/{file_name}'
            )

            os.remove(file_path)

            recording_service.set_state(recording_id, RecordingState.STOPPED)

            if AppConfig.DELETE_AFTER_STOP:
                recording_service.delete(recording_id)

            self.socketio.emit('recording-stop', {
                'id': recording.id,
                'name': recording.name
            })

            return f'Data collection for recording with id {recording_id} successfully stopped and file saved.', 200
        elif action == 'delete':
            recording = recording_service.find_by_id(recording_id)
            recording_service.delete(recording_id)

            self.socketio.emit('recording-delete', {
                'id': recording.id,
                'name': recording.name
            })

            return f'Deleted recording with id {recording_id}', 200
        elif action == "stopAndLabel":
            recording = recording_service.find_by_id(recording_id)
            if recording is None:
                return "Recording not found.", 404

            emotions = request.json
            if emotions is None:
                return 'No emotion data supplied.', 400

            if recording.state != RecordingState.RUNNING:
                return f'The data collection for the recording has not started yet', 400

            start_time = recording.start_time
            delta_seconds = (recording.last_update - start_time).seconds
            measurements = measurement_service.get_values_for_recording(recording_id)
            sample_rate = int(len(measurements) / delta_seconds) if delta_seconds != 0 else 0

            file_name_prefix = f'{recording.name}_{sample_rate}Hz_{int(start_time.timestamp() * 1000)}'
            file_name = f'{file_name_prefix}.wav'
            file_path = wav_converter.convert(
                measurements,
                sample_rate=sample_rate,
                path=f'audio/{file_name}'
            )

            labeler.label_recording(
                recording_path=file_path,
                observations_path='',
                observations=emotions,
                dropbox_client=self.dropbox_client,
                dropbox_path_prefix=file_name_prefix
            )

            dropbox_controller.upload_file_to_dropbox(
                dropbox_client=self.dropbox_client,
                file_path=file_path,
                dropbox_path=f'/PlantRecordings/{recording.name}/{file_name}'
            )

            os.remove(file_path)
            recording_service.set_state(recording_id, RecordingState.STOPPED)

            if AppConfig.DELETE_AFTER_STOP:
                recording_service.delete(recording_id)

            self.socketio.emit('recording-stop', {
                'id': recording.id,
                'name': recording.name
            })

            return "Successfully stopped recording and labeled data", 200

    def run_update(self, recording: Recording, data: bytes, now: datetime.datetime):
        sample_rate = recording.sample_rate or 142
        number_of_persisted_measurements = measurement_service.get_count(recording.id)
        start_time = recording.start_time
        parsed_data = wav_converter.parse_raw(data)

        seconds_since_start = (now - start_time).seconds
        expected_measurement_count = int(seconds_since_start * sample_rate)
        diff_number_measurements = expected_measurement_count - (number_of_persisted_measurements + len(parsed_data))
        if number_of_persisted_measurements == 0:
            parsed_data = parsed_data[:expected_measurement_count]
        elif diff_number_measurements > 0:
            parsed_data += [parsed_data[-1]] * diff_number_measurements

        self.socketio.emit(
            f'recording-update',
            {
                'measurements': parsed_data,
                'name': recording.name,
                'id': recording.id,
                'threshold': recording.threshold or 9000,
                'last_update': now.strftime('%Y-%m-%d %H:%M:%S')
            }
        )

        measurement_service.insert_many(recording.id, parsed_data, now)
        recording_service.set_last_update(recording.id, now)

        self.logger.info(
            f'Entire update for recording with id {recording.id} took {int((datetime.datetime.now() - now).total_seconds() * 1000)}ms.'
        )
