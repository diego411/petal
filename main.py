from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import query_model
import wav_converter
import time
import os
import traceback
import logging

AUDIO_DIR = 'audio'
LOG_THRESHOLD = 20

app = Flask(__name__)
socketio = SocketIO(app)
query_model.init()

bucket = []
current_emotion = "none"


@app.route('/')
def index():
    return render_template(
        "index.html",
        initial_emotion=current_emotion,
        initial_image_src=os.path.join('static', f"{current_emotion}.svg")
    )


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


@app.route("/state", methods=['GET'])
def state():
    return jsonify({'current_emotion': current_emotion})


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
