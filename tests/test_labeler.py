import unittest
from src.service.labeler import merge_observations, get_start_and_end_ms
from src.entity.Obervation import Observation
from datetime import datetime
from typing import List
from pydub import AudioSegment


def create_observation(data: dict):
    try:
        observed_at = datetime.fromtimestamp(data['timestamp'])
    except ValueError:
        observed_at = datetime.fromtimestamp(data['timestamp'] / 1000)

    return Observation(
        id=data['id'] if 'id' in data else 0,
        label=data['emotion'],
        observed_at=observed_at,
        experiment=0
    )


class TestMergeObservations(unittest.TestCase):

    def test_no_merge_needed(self):
        observations = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "neutral", "timestamp": 5}),
            create_observation({"emotion": "sad", "timestamp": 10}),
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(observations, result)

    def test_same_emotions_back_to_back(self):
        observations = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "happy", "timestamp": 2}),
            create_observation({"emotion": "sad", "timestamp": 5}),
        ]
        expected = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "sad", "timestamp": 5}),
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)

    def test_same_emotion_back_to_back_with_larger_gap(self):
        observations = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "happy", "timestamp": 5}),
        ]
        result = merge_observations(observations, gap_threshold=3)
        expected = [
            create_observation({"emotion": "happy", "timestamp": 0})
        ]
        self.assertEqual(expected, result)

    def test_multiple_gaps_that_should_merge(self):
        observations = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "neutral", "timestamp": 3}),
            create_observation({"emotion": "sad", "timestamp": 4}),
            create_observation({"emotion": "happy", "timestamp": 5}),
            create_observation({"emotion": "angry", "timestamp": 10})
        ]

        expected = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "angry", "timestamp": 10}),
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)

    def test_single_gap_that_should_merge(self):
        observations = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "neutral", "timestamp": 5}),
            create_observation({"emotion": "sad", "timestamp": 10}),
            create_observation({"emotion": "neutral", "timestamp": 11})
        ]
        result = merge_observations(observations, gap_threshold=3000)
        expected = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "neutral", "timestamp": 5}),
        ]
        self.assertEqual(expected, result)

    def test_multiple_gaps_that_should_not_merge(self):
        observations = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "neutral", "timestamp": 3}),
            create_observation({"emotion": "sad", "timestamp": 4}),
            create_observation({"emotion": "happy", "timestamp": 6.1}),
            create_observation({"emotion": "angry", "timestamp": 10})
        ]

        expected = [
            create_observation({"emotion": "happy", "timestamp": 0}),
            create_observation({"emotion": "neutral", "timestamp": 3}),
            create_observation({"emotion": "sad", "timestamp": 4}),
            create_observation({"emotion": "happy", "timestamp": 6.1}),
            create_observation({"emotion": "angry", "timestamp": 10})
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)

    def test_real_data(self):
        observations = [
            create_observation({'emotion': 'neutral', 'timestamp': 1738092129792, 'date': '22:09.792', 'id': 1}),
            create_observation({'emotion': 'happy', 'timestamp': 1738092133926, 'date': '22:13.926', 'id': 2}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738092135707, 'date': '22:15.707', 'id': 3}),
            create_observation({'emotion': 'happy', 'timestamp': 1738092136404, 'date': '22:16.404', 'id': 4}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738092137255, 'date': '22:17.255', 'id': 5}),
            create_observation({'emotion': 'happy', 'timestamp': 1738092137818, 'date': '22:17.818', 'id': 6}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738092139030, 'date': '22:19.030', 'id': 7}),
            create_observation({'emotion': 'happy', 'timestamp': 1738092139532, 'date': '22:19.532', 'id': 8}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738092142051, 'date': '22:22.051', 'id': 9}),
        ]
        expected = [
            create_observation({'emotion': 'neutral', 'timestamp': 1738092129792, 'date': '22:09.792', 'id': 1}),  #
            create_observation({'emotion': 'happy', 'timestamp': 1738092136404, 'date': '22:16.404', 'id': 4}),  #
            create_observation({'emotion': 'neutral', 'timestamp': 1738092139030, 'date': '22:19.030', 'id': 7}),  #
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)

    def test_real_data_2(self):
        observations = [
            create_observation({'emotion': 'neutral', 'timestamp': 1738175369893, 'id': 1, 'date': '29:29.893'}),
            create_observation({'emotion': 'sad', 'timestamp': 1738175371755, 'id': 2, 'date': '29:31.755'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175371848, 'id': 3, 'date': '29:31.848'}),
            create_observation({'emotion': 'angry', 'timestamp': 1738175372856, 'id': 4, 'date': '29:32.856'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175372955, 'id': 5, 'date': '29:32.955'}),
            create_observation({'emotion': 'angry', 'timestamp': 1738175373349, 'id': 6, 'date': '29:33.349'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175373854, 'id': 7, 'date': '29:33.854'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175374365, 'id': 8, 'date': '29:34.365'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175375056, 'id': 9, 'date': '29:35.056'}),
            create_observation({'emotion': 'angry', 'timestamp': 1738175375199, 'id': 10, 'date': '29:35.199'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175376060, 'id': 11, 'date': '29:36.060'}),
            create_observation({'emotion': 'angry', 'timestamp': 1738175377364, 'id': 12, 'date': '29:37.364'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175377469, 'id': 13, 'date': '29:37.469'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175381388, 'id': 14, 'date': '29:41.388'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175381488, 'id': 15, 'date': '29:41.488'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175381785, 'id': 16, 'date': '29:41.785'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175381891, 'id': 17, 'date': '29:41.891'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175384236, 'id': 18, 'date': '29:44.236'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175384302, 'id': 19, 'date': '29:44.302'}),
            create_observation({'emotion': 'sad', 'timestamp': 1738175387022, 'id': 20, 'date': '29:47.022'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175387217, 'id': 21, 'date': '29:47.217'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175388314, 'id': 22, 'date': '29:48.314'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175388419, 'id': 23, 'date': '29:48.419'}),
            create_observation({'emotion': 'sad', 'timestamp': 1738175389114, 'id': 24, 'date': '29:49.114'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175389325, 'id': 25, 'date': '29:49.325'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175394841, 'id': 26, 'date': '29:54.841'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175395649, 'id': 27, 'date': '29:55.649'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175396662, 'id': 28, 'date': '29:56.662'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175397595, 'id': 29, 'date': '29:57.595'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175398655, 'id': 30, 'date': '29:58.655'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175399360, 'id': 31, 'date': '29:59.360'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175400873, 'id': 32, 'date': '30:00.873'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175401564, 'id': 33, 'date': '30:01.564'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175403774, 'id': 34, 'date': '30:03.774'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175405815, 'id': 35, 'date': '30:05.815'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175406484, 'id': 36, 'date': '30:06.484'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175407218, 'id': 37, 'date': '30:07.218'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175407890, 'id': 38, 'date': '30:07.890'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175408188, 'id': 39, 'date': '30:08.188'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175408791, 'id': 40, 'date': '30:08.791'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175409310, 'id': 41, 'date': '30:09.310'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175410239, 'id': 42, 'date': '30:10.239'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175411326, 'id': 43, 'date': '30:11.326'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175411431, 'id': 44, 'date': '30:11.431'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175412225, 'id': 45, 'date': '30:12.225'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175415041, 'id': 46, 'date': '30:15.041'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175417965, 'id': 47, 'date': '30:17.965'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175419577, 'id': 48, 'date': '30:19.577'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175421109, 'id': 49, 'date': '30:21.109'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175421896, 'id': 50, 'date': '30:21.896'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175423279, 'id': 51, 'date': '30:23.279'}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175423445, 'id': 52, 'date': '30:23.445'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175424049, 'id': 53, 'date': '30:24.049'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175424939, 'id': 54, 'date': '30:24.939'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175428650, 'id': 55, 'date': '30:28.650'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175428755, 'id': 56, 'date': '30:28.755'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175429057, 'id': 57, 'date': '30:29.057'})
        ]
        expected = [
            create_observation({'date': '29:29.893', 'emotion': 'neutral', 'id': 1, 'timestamp': 1738175369893}),
            create_observation({'date': '29:32.856', 'emotion': 'angry', 'id': 4, 'timestamp': 1738175372856}),
            create_observation({'date': '29:33.854', 'emotion': 'neutral', 'id': 7, 'timestamp': 1738175373854}),
            create_observation({'date': '29:35.199', 'emotion': 'angry', 'id': 10, 'timestamp': 1738175375199}),
            create_observation({'date': '29:37.469', 'emotion': 'neutral', 'id': 13, 'timestamp': 1738175377469}),
            create_observation({'date': '29:41.785', 'emotion': 'surprised', 'id': 16, 'timestamp': 1738175381785}),
            create_observation({'date': '29:44.302', 'emotion': 'neutral', 'id': 19, 'timestamp': 1738175384302}),
            create_observation({'date': '29:48.314', 'emotion': 'surprised', 'id': 22, 'timestamp': 1738175388314}),
            create_observation({'date': '29:48.419', 'emotion': 'neutral', 'id': 23, 'timestamp': 1738175388419}),
            create_observation({'date': '29:54.841', 'emotion': 'happy', 'id': 26, 'timestamp': 1738175394841}),
            create_observation({'date': '29:57.595', 'emotion': 'neutral', 'id': 29, 'timestamp': 1738175397595}),
            create_observation({'date': '30:00.873', 'emotion': 'happy', 'id': 32, 'timestamp': 1738175400873}),
            create_observation({'date': '30:01.564', 'emotion': 'neutral', 'id': 33, 'timestamp': 1738175401564}),
            create_observation({'date': '30:06.484', 'emotion': 'surprised', 'id': 36, 'timestamp': 1738175406484}),
            create_observation({'emotion': 'happy', 'timestamp': 1738175408188, 'id': 39, 'date': '30:08.188'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175412225, 'id': 45, 'date': '30:12.225'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175419577, 'id': 48, 'date': '30:19.577'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175423279, 'id': 51, 'date': '30:23.279'}),
            create_observation({'emotion': 'surprised', 'timestamp': 1738175424939, 'id': 54, 'date': '30:24.939'}),
            create_observation({'emotion': 'neutral', 'timestamp': 1738175429057, 'id': 57, 'date': '30:29.057'})
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)


def get_observations() -> List[Observation]:
    return [
        Observation(id=62, label='neutral', observed_at=datetime(2025, 2, 1, 17, 23, 2, 672000), experiment=53),  # 0
        Observation(id=63, label='surprised', observed_at=datetime(2025, 2, 1, 17, 23, 3, 4000), experiment=53),  # 1
        Observation(id=64, label='neutral', observed_at=datetime(2025, 2, 1, 17, 23, 4, 552000), experiment=53),  # 2
        Observation(id=65, label='happy', observed_at=datetime(2025, 2, 1, 17, 23, 6, 223000), experiment=53),  # 3
        Observation(id=66, label='neutral', observed_at=datetime(2025, 2, 1, 17, 23, 10, 696000), experiment=53),  # 4
        Observation(id=67, label='surprised', observed_at=datetime(2025, 2, 1, 17, 23, 12, 505000), experiment=53),  # 5
        Observation(id=68, label='neutral', observed_at=datetime(2025, 2, 1, 17, 23, 13, 807000), experiment=53),  # 6
        Observation(id=69, label='sad', observed_at=datetime(2025, 2, 1, 17, 23, 18, 629000), experiment=53),  # 7
        Observation(id=70, label='neutral', observed_at=datetime(2025, 2, 1, 17, 23, 18, 742000), experiment=53),  # 8
        Observation(id=71, label='happy', observed_at=datetime(2025, 2, 1, 17, 23, 27, 998000), experiment=53),  # 9
        Observation(id=72, label='neutral', observed_at=datetime(2025, 2, 1, 17, 23, 29, 142000), experiment=53),  # 10
        Observation(id=73, label='happy', observed_at=datetime(2025, 2, 1, 17, 23, 57, 788000), experiment=53),  # 11
        Observation(id=74, label='neutral', observed_at=datetime(2025, 2, 1, 17, 23, 57, 993000), experiment=53),  # 12
        Observation(id=75, label='surprised', observed_at=datetime(2025, 2, 1, 17, 24, 1, 315000), experiment=53),  # 13
        Observation(id=76, label='neutral', observed_at=datetime(2025, 2, 1, 17, 24, 1, 520000), experiment=53),  # 14
        Observation(id=77, label='surprised', observed_at=datetime(2025, 2, 1, 17, 24, 1, 833000), experiment=53),  # 15
        Observation(id=78, label='neutral', observed_at=datetime(2025, 2, 1, 17, 24, 1, 936000), experiment=53),  # 16
        Observation(id=79, label='surprised', observed_at=datetime(2025, 2, 1, 17, 24, 2, 393000), experiment=53),  # 17
        Observation(id=80, label='neutral', observed_at=datetime(2025, 2, 1, 17, 24, 2, 493000), experiment=53),  # 18
        Observation(id=81, label='surprised', observed_at=datetime(2025, 2, 1, 17, 24, 2, 597000), experiment=53),  # 19
        Observation(id=82, label='neutral', observed_at=datetime(2025, 2, 1, 17, 24, 2, 691000), experiment=53)  # 20
    ]


class TestLabelRecording(unittest.TestCase):

    def test_real_use_case(self):
        observations = get_observations()
        result = merge_observations(observations, gap_threshold=3000)
        expected_observations = [
            observations[0],
            observations[3],
            observations[4],
            observations[7],
            observations[8],
            observations[11],
            observations[12],
            observations[15],
            observations[18]
        ]

        self.assertEqual(expected_observations, result)

        recording_segment = AudioSegment.from_wav('./basilikum_142Hz_1738426977359.wav')
        recording_start_timestamp = 1738426977359
        recording_start = datetime.fromtimestamp(recording_start_timestamp / 1000)
        print(
            f"Recording started at {recording_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
        )

        first_observation = expected_observations[0]
        second_observation = expected_observations[1]
        print(f"First observation started at {first_observation.observed_at.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        delta_start_first_observation = first_observation.observed_at - recording_start
        print(
            f"Time passed between start of recording and first observation: {int(delta_start_first_observation.total_seconds() * 1000)}"
        )

        delta_start_second_observation = second_observation.observed_at - recording_start
        print(
            f"Time passed between start of recording and second observation: {int(delta_start_second_observation.total_seconds() * 1000)}"
        )

        start_ms, end_ms = get_start_and_end_ms(
            observation=expected_observations[0],
            next_observation=expected_observations[1],
            recording_start_timestamp=recording_start_timestamp,
            recording_length=len(recording_segment)
        )

        self.assertEqual(5313, start_ms)
        self.assertEqual(8864, end_ms)

        last_observation = expected_observations[-1]
        start_ms, end_ms = get_start_and_end_ms(
            observation=last_observation,
            next_observation=None,
            recording_start_timestamp=recording_start_timestamp,
            recording_length=len(recording_segment)
        )

        self.assertEqual((last_observation.observed_at.timestamp() * 1000) - recording_start_timestamp, start_ms)
        self.assertEqual(len(recording_segment), end_ms)
