from flask import request, current_app
from flask_restful import Resource
from src.service import wav_converter, labeler
from src.AppConfig import AppConfig
from src.models.JakobPlantEmotionClassifier import JakobPlantEmotionClassifier
import traceback
import os
import time


# TODO: model this properly (using recording entities?)
class LegacyResource(Resource):

    def __init__(self):
        self.bucket = []  # TODO: persist this
        self.current_emotion = 'none'  # TODO: persist this
        self.logger = current_app.logger
        self.socketio = current_app.socketio
        self.dropbox_client = current_app.dropbox_client
        self.jakob_classifier = JakobPlantEmotionClassifier()

    def post(self, action: str):
        if action == 'update':
            data = request.data
            self.bucket = self.bucket + wav_converter.augment(wav_converter.parse_raw(data), AppConfig.AUGMENT_WINDOW,
                                                              AppConfig.AUGMENT_PADDING)

            if len(self.bucket) < 300_000:
                return {'current_emotion': self.current_emotion}, 200

            file_path = wav_converter.convert(self.bucket)

            try:
                predictions = self.jakob_classifier.classify(file_path)
            except Exception as e:
                self.logger.error(traceback.format_exc())
                return "something went wrong getting the predictions from the model", 500
            finally:
                log_size = len([name for name in os.listdir(AppConfig.AUDIO_DIR) if
                                os.path.isfile(os.path.join(AppConfig.AUDIO_DIR, name))])
                if log_size >= int(AppConfig.LOG_THRESHOLD):
                    os.remove(file_path)

            self.current_emotion = predictions['current_emotion']
            self.bucket = []

            self.socketio.emit(
                'update',
                {
                    'emotion': self.current_emotion,
                }
            )

            return {'current_emotion': self.current_emotion}, 200
        elif action == 'classify':
            binary_data = request.data

            extension = 'wav'
            date = time.ctime(time.time())
            file_path = f'audio/req_file_{date}.{extension}'
            with open(file_path, 'wb') as f:
                f.write(binary_data)

            predictions = self.jakob_classifier.classify(file_path)
            os.remove(file_path)
            return predictions, 200

