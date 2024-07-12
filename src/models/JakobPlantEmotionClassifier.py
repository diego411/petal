from typing import Dict, Union

import tensorflow as tf
from scipy.io import wavfile
import numpy as np


class JakobPlantEmotionClassifier:
    def __init__(self, batch_size: int = 16, hop: int = 3, save_path: str = './src/models/plant_mfcc_resnet'):
        self.batch_size = batch_size
        self.hop = hop
        self.num_mfcc = 60
        self.window = 20

        self.model = tf.keras.models.load_model(save_path)

    def classify(self, path_to_data: str):
        dataset = self._create_dataset(path_to_data)
        return self._run_inference(dataset)

    def _create_dataset(self, path_to_data: str):
        """
            Function that creates a tensorflow dataset from a single plant file.

            :param path_to_data: The file path of the plant file
            :return: Tensorflow dataset with the plant data
            """

        sample_rate, data = wavfile.read(path_to_data)
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std

        dataset = tf.data.Dataset.from_generator(
            self._get_data_generator(data, sample_rate),
            output_types=(tf.float32,),
            output_shapes=(tf.TensorShape([self.window * sample_rate]),),
        )
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _get_data_generator(self, data: np.ndarray, sample_rate: int):
        """
        Generator that generates the data

        :param data: The plant data in an array
        :param sample_rate: The sample rate of the file
        :return: Generator that yields data and label.
        """

        def generator():
            for second in range(self.window, int(data.shape[0] / sample_rate), self.hop):
                sample = np.reshape(
                    data[(second - self.window) * sample_rate: second * sample_rate],
                    (-1,),
                )
                yield sample,

        return generator

    def _run_inference(self, dataset: tf.data.Dataset) -> Dict[str, Union[int, str]]:
        """
        Run inference on a plant dataset.

        :param classifier: The classifier to use for the prediction
        :param dataset: The dataset that contains the plant data in windows.
        """
        emotions = ["anger", "surprise", "disgust", "joy", "fear", "sadness", "neutral"]
        predictions = {}
        results = self.model.predict(dataset)
        emotion_ids = np.argmax(results, axis=1)
        print(results.shape)
        unique_emotions_ids = np.unique(emotion_ids, return_counts=True)
        highest_val = 0
        main_emotion = ""
        for i, emotion in enumerate(emotions):
            emotion_index = self.find_first_index(unique_emotions_ids[0], i)
            val = int(unique_emotions_ids[1][emotion_index]) if emotion_index > -1 else 0
            if val > highest_val:
                highest_val = val
                main_emotion = emotion
            predictions[emotion] = val

        print("{:<10} {:<10}".format('Emotion', '#Occurrences'))
        for key, value in predictions.items():
            print("{:<10} {:<10}".format(key, value))
        predictions["current_emotion"] = main_emotion
        return predictions

    def find_first_index(self, array: np.ndarray, element) -> int:
        all_indexes = np.where(array == element)[0]
        if len(all_indexes) == 0:
            return -1
        return all_indexes[0]
