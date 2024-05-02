from flask import Flask, request, jsonify
import query_model
import wav_converter
import time
import os
import traceback
import logging

app = Flask(__name__)
query_model.init()

bucket = []
current_emotion = "None"


@app.route('/')
def index():
    return {"data": "Plant Emotion Classification v0.0.1"}


@app.route('/update', methods=['POST'])
def update():
    global bucket
    global current_emotion
    data = request.data
    bucket = bucket + wav_converter.parse_raw(data)

    if len(bucket) < 300_000:
        return jsonify({'current_emotion': current_emotion}), 200

    file_path = wav_converter.convert(bucket)

    try:
        predictions = query_model.classify(file_path)
    except Exception as e:
        logging.error(traceback.format_exc())
        return "something went wrong getting the predictions from the model", 500
    finally:
        os.remove(file_path)

    current_emotion = predictions['current_emotion']
    bucket = []

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
