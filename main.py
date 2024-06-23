from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import query_model
import wav_converter
import dropbox_controller
import time
import datetime
import os
import traceback
import logging

AUDIO_DIR = 'audio'
LOG_THRESHOLD = 20

app = Flask(__name__)
socketio = SocketIO(app)
query_model.init()

state_map = {}
bucket = []
current_emotion = "none"


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


@app.route('/states')
def states():
    return render_template(
        "states.html",
        users=state_map.keys()
    )


@app.route('/start', methods=['POST'])
def start():
    global state_map
    data = request.json
    if 'user' not in data:
        return 'Field \"user\" in request body is required', 400

    user = data['user']

    if user in state_map:
        return 'Recording for this user started already. Stop it first', 400

    now = datetime.datetime.now()
    state_map[user] = {
        'start_time': now,
        'bucket': [],
        'raw_bucket': []
    }

    return f'Successfully started data collection for {user}', 200


@app.route('/update/<user>', methods=['POST'])
def update_by_name(user):
    global state_map

    if user not in state_map:
        return f'The data collection for the user: {user} has not started yet', 400

    data = request.data

    user_bucket = state_map[user]['bucket']
    user_bucket = user_bucket + wav_converter.parse_raw(data)
    state_map[user]['bucket'] = user_bucket

    socketio.emit(
        f'update-{user}',
        {
            'bucket': user_bucket,
        }
    )

    return f'Successful update for: {user}', 200


@app.route('/stop', methods=['POST'])
def stop():
    global state_map

    data = request.json
    user = data['user']

    if user not in state_map:
        return 'Not recording in progress for this user.', 400

    user_bucket = state_map[user]['bucket']
    start_time = state_map[user]['start_time']
    now = datetime.datetime.now()
    delta_seconds = (now - start_time).seconds
    sample_rate = int(len(user_bucket) / delta_seconds)
    file_name = f'{user}_{start_time.strftime("%d-%m-%Y_%H:%M:%S")}_{sample_rate}Hz.wav'
    file_path = wav_converter.convert(
        user_bucket,
        sample_rate=sample_rate,
        path=f'audio/{file_name}'
    )

    dropbox_controller.upload_file_to_dropbox(
        file_path=file_path,
        dropbox_path=f'/Data/{user}/{file_name}'
    )

    os.remove(file_path)

    del state_map[user]

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
        predictions = query_model.classify(file_path)
    except Exception as e:
        logging.error(traceback.format_exc())
        return "something went wrong getting the predictions from the model", 500
    finally:
        log_size = len([name for name in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, name))])
        if log_size >= LOG_THRESHOLD:
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
    user_data = state_map[user] if user in state_map else None

    return render_template(
        'user_state.html',
        user=user,
        start_time=user_data['start_time'],
        initial_bucket=user_data['bucket']
    )


@app.route('/classify', methods=['POST'])
def classify():
    print("classify endpoint called")
    binary_data = request.data

    extension = 'wav'
    date = time.ctime(time.time())
    file_path = f'audio/req_file_{date}.{extension}'
    with open(file_path, 'wb') as f:
        f.write(binary_data)

    predictions = query_model.classify(file_path)
    os.remove(file_path)
    print(predictions)
    return jsonify(predictions), 200


if __name__ == '__main__':
    app.run(debug=True)
    socketio.run(app, host='0.0.0.0', port=5000)
