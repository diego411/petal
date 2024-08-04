from flask import Flask, request, jsonify, render_template, g
from flask_socketio import SocketIO
from src.service import wav_converter
from src.controller import dropbox_controller
from src.models.JakobPlantEmotionClassifier import JakobPlantEmotionClassifier
from src.AppConfig import AppConfig
import time
import datetime
import os
import traceback
import logging

state_map = {}
bucket = []
current_emotion = "none"

socketio = SocketIO()


def create_app():
    app = Flask(__name__)
    app.config.from_object(AppConfig)
    socketio.init_app(app)

    jakob_classifier = JakobPlantEmotionClassifier()
    dropbox_client = dropbox_controller.create_dropbox_client(
        app_key=app.config['DROPBOX_APP_KEY'],
        app_secret=app.config['DROPBOX_APP_SECRET'],
        refresh_token=app.config['DROPBOX_REFRESH_TOKEN']
    )

    @app.context_processor
    def inject_version():
        return dict(version=app.config['VERSION'])

    @app.route('/')
    def index():
        return render_template(
            "index.html",
        )

    @app.route('/liveEmotion')
    def live_emotion():
        return render_template(
            "live_emotion.html",
            initial_emotion=current_emotion,
            initial_image_src=os.path.join('static', f"{current_emotion}.svg")
        )

    @app.route('/audioClassification')
    def audio_classification():
        return render_template(
            "audio_classification.html"
        )

    @app.route('/states')
    def states():
        users = []
        for name, state in state_map.items():
            if state.get('start_time') is not None:
                users.append({
                    "name": name,
                    "start_time": state['start_time'].strftime('%d.%m.%Y %H:%M:%S'),
                    "bucket": state['bucket']
                })
                continue

            users.append({
                "name": name,
                "bucket": state['bucket']
            })

        return render_template(
            "states.html",
            users=users
        )

    @app.route('/register', methods=['POST'])
    def register():
        global state_map
        data = request.json

        if 'user' not in data:
            return 'Field \"user\" in request body is required', 400

        user = data['user']

        if user in state_map:
            return 'This user is already registered.', 400

        state_map[user] = {
            'bucket': []
        }

        return f'Successfully registered {user}.', 200

    @app.route('/start', methods=['POST'])
    def start():

        global state_map
        data = request.json

        if 'user' not in data:
            return 'Field \"user\" in request body is required', 400

        user = data['user']

        if user in state_map and state_map[user].get('start_time') is not None:
            return 'Recording for this user started already. Stop it first', 400

        now = datetime.datetime.now()
        state_map[user] = {
            'start_time': now,
            'bucket': [],
            'last_update': now
        }

        socketio.emit('user-start', {
            'name': user,
            'start_time': now.strftime('%d.%m.%Y %H:%M:%S'),
            'bucket': []
        })

        return f'Successfully started data collection for {user}', 200

    @app.route('/update/<user>', methods=['POST'])
    def update_by_name(user):

        global state_map

        if user not in state_map:
            state_map[user] = {
                'bucket': []
            }

        if state_map[user].get('start_time') is None:
            return f'The data collection for the user: {user} has not started yet', 400

        data = request.data

        user_bucket = state_map[user]['bucket']
        user_bucket = user_bucket + wav_converter.parse_raw(data)
        state_map[user]['bucket'] = user_bucket
        state_map[user]['last_update'] = datetime.datetime.now()
        socketio.emit(
            f'user-update',
            {
                'bucket': user_bucket,
                'name': user,
            }
        )

        return f'Successful update for: {user}', 200

    @app.route('/stop', methods=['POST'])
    def stop():
        global state_map

        data = request.json
        user = data['user']
        if user not in state_map or state_map[user].get('start_time') is None:
            return 'No recording in progress for this user.', 400

        user_bucket = state_map[user]['bucket']
        start_time = state_map[user]['start_time']
        delta_seconds = (state_map[user]['last_update'] - start_time).seconds
        sample_rate = int(len(user_bucket) / delta_seconds) if delta_seconds != 0 else 0
        file_name = f'{user}_{start_time.strftime("%d-%m-%Y_%H:%M:%S")}_{sample_rate}Hz.wav'
        file_path = wav_converter.convert(
            user_bucket,
            sample_rate=sample_rate,
            path=f'audio/{file_name}'
        )

        dropbox_controller.upload_file_to_dropbox(
            dropbox_client=dropbox_client,
            file_path=file_path,
            dropbox_path=f'/Data/{user}/{file_name}'
        )

        os.remove(file_path)

        del state_map[user]

        socketio.emit('user-stop', {
            'name': user
        })

        return f'Data collection for user: {user} successfully stopped and file saved.', 200

    @app.route('/update', methods=['POST'])
    def update():
        global bucket
        global current_emotion

        data = request.data
        bucket = bucket + wav_converter.augment(wav_converter.parse_raw(data))

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

    @app.route('/state/<user>', methods=['GET'])
    def user_state(user):
        global state_map

        if user not in state_map:
            return render_template(
                'empty_recording.html',
                user=user
            )

        user_data = state_map[user]

        if user_data.get('start_time') is not None:
            return render_template(
                'user_state.html',
                user=user,
                start_time=user_data['start_time'],
                initial_bucket=user_data['bucket']
            )

        return render_template(
            'user_state.html',
            user=user,
            initial_bucket=user_data['bucket']
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

    return app


if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(debug=True)
    socketio.run(flask_app, host="0.0.0.0", port=5000)
