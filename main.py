from flask import Flask, request, jsonify
import query_model
import wav_converter
import time
import os
import traceback
import logging

app = Flask(__name__)
query_model.init()

@app.route('/')
def index():
    return { "data": "Plant Emotion Classification v0.0.1" }


@app.route('/update', methods=['POST'])
def update():
    data = request.json

    if 'voltages' not in data:
        return "malformed request, not volatages provided", 400

    voltages = data['voltages']
    file_path = wav_converter.convert(voltages)
    try:
        predictions = query_model.classify(file_path)
    except Exception as e:
        logging.error(traceback.format_exc())
        return "bad request", 400
    finally:
        os.remove(file_path)

    print(predictions)
    return jsonify(predictions), 200

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
    
