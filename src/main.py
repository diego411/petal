import datetime
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO

from src.AppConfig import AppConfig
from src.controller import dropbox_controller
from src.database import db
from src.models.JakobPlantEmotionClassifier import JakobPlantEmotionClassifier
from src.service import labeler
from src.service import wav_converter
from src.service import user_service
from src.service import recording_service
from src.service import measurement_service
from src.entity.Recording import Recording
from src.entity.RecordingState import RecordingState

bucket = []  # TODO: persists this
current_emotion = "none"  # TODO: persist this

socketio = SocketIO()


def create_app():
    app = Flask(__name__)
    app.config.from_object(AppConfig)
    socketio.init_app(app)

    db.init_tables()
    db.create_measurement_partition()
    db.create_measurement_partition(offset=1)
    jakob_classifier = JakobPlantEmotionClassifier()
    dropbox_client = dropbox_controller.create_dropbox_client(
        app_key=app.config['DROPBOX_APP_KEY'],
        app_secret=app.config['DROPBOX_APP_SECRET'],
        refresh_token=app.config['DROPBOX_REFRESH_TOKEN']
    )
    augment_window = app.config['AUGMENT_WINDOW']
    augment_padding = app.config['AUGMENT_PADDING']

    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_file = os.path.join('logs', 'app.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,  # Log level can be changed as needed
        format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Set log level
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    file_handler.setFormatter(formatter)

    app.logger.addHandler(file_handler)

    scheduler = BackgroundScheduler()
    scheduler.add_job(db.create_measurement_partition, 'cron', args=[1], hour=23, minute=52)
    scheduler.start()

    @app.context_processor
    def inject_version():
        return dict(version=app.config['VERSION'])

    @app.route('/')
    def index():
        return render_template(
            "index.html",
        )

    @app.route('/live-emotion')
    def live_emotion():
        return render_template(
            "live_emotion.html",
            initial_emotion=current_emotion,
            initial_image_src=os.path.join('static', f"{current_emotion}.svg")
        )

    @app.route('/audio-classification')
    def audio_classification():
        return render_template(
            "audio_classification.html"
        )

    @app.route('/states')
    def states():
        recordings = recording_service.find_by_state(RecordingState.REGISTERED) + recording_service.find_by_state(
            RecordingState.RUNNING)

        return render_template(
            "states.html",
            recordings=recording_service.to_dtos(recordings)
        )

    @app.route('/recording/register', methods=['POST'])
    def register():
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

        return jsonify({'id': recording_id}), 201

    @app.route('/recording/<recording_id>/start', methods=['POST'])
    def start(recording_id):
        recording = recording_service.find_by_id(recording_id)

        if recording.state == RecordingState.RUNNING:
            return 'Recording for this user started already. Stop it first', 400

        now = datetime.datetime.now()
        recording_service.set_last_update(recording_id, now)
        recording_service.set_start_time(recording_id, now)
        recording_service.set_state(recording_id, RecordingState.RUNNING)

        socketio.emit('recording-start', {
            'name': recording.name,
            'id': recording.id,
            'start_time': now.strftime('%d.%m.%Y %H:%M:%S'),
        })

        return f'Successfully started data collection for recording', 200

    def run_update(recording: Recording, data: bytes, now: datetime.datetime):
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

        socketio.emit(
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

        app.logger.info(
            f'Entire update for recording with id {recording.id} took {int((datetime.datetime.now() - now).total_seconds() * 1000)}ms.'
        )

    @app.route('/recording/<recording_id>/update', methods=['POST'])
    def update_recording(recording_id):
        recording_id = int(recording_id)
        now = datetime.datetime.now()

        recording = recording_service.find_by_id(recording_id)
        if recording is None:
            return f"Recording with id {recording_id} not found.", 404

        if recording.state != RecordingState.RUNNING:
            return f'The data collection for the recording has not started yet', 400

        data: bytes = request.data
        thread = threading.Thread(target=run_update(recording, data, now))
        thread.start()
        return f'Successfully started update for: {recording.name}', 200

    @app.route('/recording/<recording_id>/stop', methods=['POST'])
    def stop(recording_id):
        recording = recording_service.find_by_id(recording_id)
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
        app.logger.info(
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
            dropbox_client=dropbox_client,
            file_path=file_path,
            dropbox_path=f'/PlantRecordings/{user.name}/{file_name}'
        )

        os.remove(file_path)

        recording_service.set_state(recording_id, RecordingState.STOPPED)

        if app.config['DELETE_AFTER_STOP']:
            recording_service.delete(recording_id)

        socketio.emit('recording-stop', {
            'id': recording.id,
            'name': recording.name
        })

        return f'Data collection for recording with id {recording_id} successfully stopped and file saved.', 200

    @app.route('/recording/<recording_id>/delete', methods=['POST'])
    def delete(recording_id):
        recording = recording_service.find_by_id(recording_id)
        recording_service.delete(recording_id)

        socketio.emit('recording-delete', {
            'id': recording.id,
            'name': recording.name
        })

        return f'Deleted recording with id {recording_id}', 200

    @app.route('/update', methods=['POST'])
    def update():
        global bucket
        global current_emotion

        data = request.data
        bucket = bucket + wav_converter.augment(wav_converter.parse_raw(data), augment_window, augment_padding)

        if len(bucket) < 300_000:
            return jsonify({'current_emotion': current_emotion}), 200

        file_path = wav_converter.convert(bucket)

        try:
            predictions = jakob_classifier.classify(file_path)
        except Exception as e:
            logging.error(traceback.format_exc())
            return "something went wrong getting the predictions from the model", 500
        finally:
            log_size = len([name for name in os.listdir(app.config['AUDIO_DIR']) if
                            os.path.isfile(os.path.join(app.config['AUDIO_DIR'], name))])
            if log_size >= int(app.config['LOG_THRESHOLD']):
                os.remove(file_path)

        current_emotion = predictions['current_emotion']
        bucket = []

        socketio.emit(
            'update',
            {
                'emotion': current_emotion,
            }
        )

        return jsonify({'current_emotion': current_emotion}), 200

    @app.route('/state', methods=['GET'])
    def state():
        return jsonify({'current_emotion': current_emotion})

    @app.route('/recording/<recording_id>', methods=['GET'])
    def get_recording(recording_id):
        recording = recording_service.find_by_id(recording_id)
        if recording is None:
            return render_template(
                'empty_recording.html',
                id=recording_id  # TODO: fix naming
            )

        return render_template(
            'user_state.html',
            recording=recording_service.to_dto(recording),
        )

    @app.route('/classify', methods=['POST'])
    def classify():
        binary_data = request.data

        extension = 'wav'
        date = time.ctime(time.time())
        file_path = f'audio/req_file_{date}.{extension}'
        with open(file_path, 'wb') as f:
            f.write(binary_data)

        predictions = jakob_classifier.classify(file_path)
        os.remove(file_path)
        return jsonify(predictions), 200

    @app.route('/label-recordings', methods=['GET'])
    def label_recordings():
        return render_template(
            'label.html'
        )

    @app.route('/recording/<recording_id>/stopAndLabel', methods=['POST'])
    def stopAndLabel(recording_id):
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
            dropbox_client=dropbox_client,
            dropbox_path_prefix=file_name_prefix
        )

        dropbox_controller.upload_file_to_dropbox(
            dropbox_client=dropbox_client,
            file_path=file_path,
            dropbox_path=f'/PlantRecordings/{recording.name}/{file_name}'
        )

        os.remove(file_path)
        recording_service.set_state(recording_id, RecordingState.STOPPED)

        if app.config['DELETE_AFTER_STOP']:
            recording_service.delete(recording_id)

        socketio.emit('recording-stop', {
            'id': recording.id,
            'name': recording.name
        })

        return "Successfully stopped recording and labeled data", 200

    @app.route('/record-and-label', methods=['GET'])
    def record_and_label():
        recordings = recording_service.find_by_state(RecordingState.REGISTERED) + recording_service.find_by_state(
            RecordingState.RUNNING)

        return render_template(
            'record_and_label.html',
            recordings=recording_service.to_dtos(recordings)
        )

    @app.route('/label', methods=['POST'])
    def label():
        if 'recording' not in request.files or 'moodyExport' not in request.files:
            return jsonify({'error': 'Both the wav recording and the moody export file are required.'}), 400

        recording = request.files['recording']
        moody_export = request.files['moodyExport']

        recording_path = f'label_dumps/{recording.filename}'
        moody_export_path = f'label_dumps/{moody_export.filename}'
        recording.save(recording_path)
        moody_export.save(moody_export_path)

        labeler.label_recording(
            recording_path=recording_path,
            observations_path=moody_export_path,
            dropbox_client=dropbox_client
        )

        os.remove(recording_path)
        os.remove(moody_export_path)

        return "Successfully labeled data", 200

    @app.route('/scripts', methods=['GET'])
    def get_scripts():
        scripts = []
        path = Path('scripts')
        for item in path.iterdir():
            if item.is_dir():
                script = {"name": item.name, "versions": []}
                versions = []
                for version in item.iterdir():
                    if not version.is_dir() and version.suffix == '.npy':
                        content = ''
                        with version.open('r') as file:
                            content = file.read()  # .replace('"', '\\"').replace("'", "\\'")
                        versions.append({"identifier": version.name.split('.')[0], "content": content})

                sorted(versions, key=lambda element: element['identifier'])
                versions[-1]['identifier'] += " (latest)"
                script['versions'] = versions
                scripts.append(script)

        return render_template(
            'scripts.html',
            scripts=scripts,
            parsed_scripts=json.dumps(scripts)
        )

    return app


if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(debug=True)
    socketio.run(flask_app, host="0.0.0.0", port=5000)
