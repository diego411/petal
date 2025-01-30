import unittest
from src.service.labeler import merge_observations


class TestMergeObservations(unittest.TestCase):

    def test_no_merge_needed(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "neutral", "timestamp": 5000},
            {"emotion": "sad", "timestamp": 10000},
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(observations, result)

    def test_same_emotions_back_to_back(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "happy", "timestamp": 2000},
            {"emotion": "sad", "timestamp": 5000},
        ]
        expected = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "sad", "timestamp": 5000},
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)

    def test_same_emotion_back_to_back_with_larger_gap(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "happy", "timestamp": 5000},
        ]
        result = merge_observations(observations, gap_threshold=3000)
        expected = [
            {"emotion": "happy", "timestamp": 0}
        ]
        self.assertEqual(expected, result)

    def test_multiple_gaps_that_should_merge(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "neutral", "timestamp": 3000},
            {"emotion": "sad", "timestamp": 4000},
            {"emotion": "happy", "timestamp": 5000},
            {"emotion": "angry", "timestamp": 10000}
        ]

        expected = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "angry", "timestamp": 10000},
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)

    def test_single_gap_that_should_merge(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "neutral", "timestamp": 5000},
            {"emotion": "sad", "timestamp": 10000},
            {"emotion": "neutral", "timestamp": 11000}
        ]
        result = merge_observations(observations, gap_threshold=3000)
        expected = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "neutral", "timestamp": 5000},
        ]
        self.assertEqual(expected, result)

    def test_multiple_gaps_that_should_not_merge(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "neutral", "timestamp": 3000},
            {"emotion": "sad", "timestamp": 4000},
            {"emotion": "happy", "timestamp": 6100},
            {"emotion": "angry", "timestamp": 10000}
        ]

        expected = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "neutral", "timestamp": 3000},
            {"emotion": "sad", "timestamp": 4000},
            {"emotion": "happy", "timestamp": 6100},
            {"emotion": "angry", "timestamp": 10000}
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)

    def test_real_data(self):
        observations = [
            {'emotion': 'neutral', 'timestamp': 1738092129792, 'date': '22:09.792', 'id': 1},
            {'emotion': 'happy', 'timestamp': 1738092133926, 'date': '22:13.926', 'id': 2},
            {'emotion': 'neutral', 'timestamp': 1738092135707, 'date': '22:15.707', 'id': 3},
            {'emotion': 'happy', 'timestamp': 1738092136404, 'date': '22:16.404', 'id': 4},
            {'emotion': 'neutral', 'timestamp': 1738092137255, 'date': '22:17.255', 'id': 5},
            {'emotion': 'happy', 'timestamp': 1738092137818, 'date': '22:17.818', 'id': 6},
            {'emotion': 'neutral', 'timestamp': 1738092139030, 'date': '22:19.030', 'id': 7},
            {'emotion': 'happy', 'timestamp': 1738092139532, 'date': '22:19.532', 'id': 8},
            {'emotion': 'neutral', 'timestamp': 1738092142051, 'date': '22:22.051', 'id': 9},
        ]
        expected = [
            {'emotion': 'neutral', 'timestamp': 1738092129792, 'date': '22:09.792', 'id': 1},  #
            {'emotion': 'happy', 'timestamp': 1738092136404, 'date': '22:16.404', 'id': 4},  #
            {'emotion': 'neutral', 'timestamp': 1738092139030, 'date': '22:19.030', 'id': 7},  #
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)

    def test_real_data_2(self):
        observations = [
            {'emotion': 'neutral', 'timestamp': 1738175369893, 'id': 1, 'date': '29:29.893'},
            {'emotion': 'sad', 'timestamp': 1738175371755, 'id': 2, 'date': '29:31.755'},
            {'emotion': 'neutral', 'timestamp': 1738175371848, 'id': 3, 'date': '29:31.848'},
            {'emotion': 'angry', 'timestamp': 1738175372856, 'id': 4, 'date': '29:32.856'},
            {'emotion': 'neutral', 'timestamp': 1738175372955, 'id': 5, 'date': '29:32.955'},
            {'emotion': 'angry', 'timestamp': 1738175373349, 'id': 6, 'date': '29:33.349'},
            {'emotion': 'neutral', 'timestamp': 1738175373854, 'id': 7, 'date': '29:33.854'},
            {'emotion': 'happy', 'timestamp': 1738175374365, 'id': 8, 'date': '29:34.365'},
            {'emotion': 'neutral', 'timestamp': 1738175375056, 'id': 9, 'date': '29:35.056'},
            {'emotion': 'angry', 'timestamp': 1738175375199, 'id': 10, 'date': '29:35.199'},
            {'emotion': 'neutral', 'timestamp': 1738175376060, 'id': 11, 'date': '29:36.060'},
            {'emotion': 'angry', 'timestamp': 1738175377364, 'id': 12, 'date': '29:37.364'},
            {'emotion': 'neutral', 'timestamp': 1738175377469, 'id': 13, 'date': '29:37.469'},
            {'emotion': 'surprised', 'timestamp': 1738175381388, 'id': 14, 'date': '29:41.388'},
            {'emotion': 'neutral', 'timestamp': 1738175381488, 'id': 15, 'date': '29:41.488'},
            {'emotion': 'surprised', 'timestamp': 1738175381785, 'id': 16, 'date': '29:41.785'},
            {'emotion': 'neutral', 'timestamp': 1738175381891, 'id': 17, 'date': '29:41.891'},
            {'emotion': 'surprised', 'timestamp': 1738175384236, 'id': 18, 'date': '29:44.236'},
            {'emotion': 'neutral', 'timestamp': 1738175384302, 'id': 19, 'date': '29:44.302'},
            {'emotion': 'sad', 'timestamp': 1738175387022, 'id': 20, 'date': '29:47.022'},
            {'emotion': 'neutral', 'timestamp': 1738175387217, 'id': 21, 'date': '29:47.217'},
            {'emotion': 'surprised', 'timestamp': 1738175388314, 'id': 22, 'date': '29:48.314'},
            {'emotion': 'neutral', 'timestamp': 1738175388419, 'id': 23, 'date': '29:48.419'},
            {'emotion': 'sad', 'timestamp': 1738175389114, 'id': 24, 'date': '29:49.114'},
            {'emotion': 'neutral', 'timestamp': 1738175389325, 'id': 25, 'date': '29:49.325'},
            {'emotion': 'happy', 'timestamp': 1738175394841, 'id': 26, 'date': '29:54.841'},
            {'emotion': 'neutral', 'timestamp': 1738175395649, 'id': 27, 'date': '29:55.649'},
            {'emotion': 'happy', 'timestamp': 1738175396662, 'id': 28, 'date': '29:56.662'},
            {'emotion': 'neutral', 'timestamp': 1738175397595, 'id': 29, 'date': '29:57.595'},
            {'emotion': 'happy', 'timestamp': 1738175398655, 'id': 30, 'date': '29:58.655'},
            {'emotion': 'neutral', 'timestamp': 1738175399360, 'id': 31, 'date': '29:59.360'},
            {'emotion': 'happy', 'timestamp': 1738175400873, 'id': 32, 'date': '30:00.873'},
            {'emotion': 'neutral', 'timestamp': 1738175401564, 'id': 33, 'date': '30:01.564'},
            {'emotion': 'surprised', 'timestamp': 1738175403774, 'id': 34, 'date': '30:03.774'},
            {'emotion': 'neutral', 'timestamp': 1738175405815, 'id': 35, 'date': '30:05.815'},
            {'emotion': 'surprised', 'timestamp': 1738175406484, 'id': 36, 'date': '30:06.484'},
            {'emotion': 'neutral', 'timestamp': 1738175407218, 'id': 37, 'date': '30:07.218'},
            {'emotion': 'surprised', 'timestamp': 1738175407890, 'id': 38, 'date': '30:07.890'},
            {'emotion': 'happy', 'timestamp': 1738175408188, 'id': 39, 'date': '30:08.188'},
            {'emotion': 'neutral', 'timestamp': 1738175408791, 'id': 40, 'date': '30:08.791'},
            {'emotion': 'surprised', 'timestamp': 1738175409310, 'id': 41, 'date': '30:09.310'},
            {'emotion': 'neutral', 'timestamp': 1738175410239, 'id': 42, 'date': '30:10.239'},
            {'emotion': 'surprised', 'timestamp': 1738175411326, 'id': 43, 'date': '30:11.326'},
            {'emotion': 'happy', 'timestamp': 1738175411431, 'id': 44, 'date': '30:11.431'},
            {'emotion': 'neutral', 'timestamp': 1738175412225, 'id': 45, 'date': '30:12.225'},
            {'emotion': 'happy', 'timestamp': 1738175415041, 'id': 46, 'date': '30:15.041'},
            {'emotion': 'neutral', 'timestamp': 1738175417965, 'id': 47, 'date': '30:17.965'},
            {'emotion': 'surprised', 'timestamp': 1738175419577, 'id': 48, 'date': '30:19.577'},
            {'emotion': 'neutral', 'timestamp': 1738175421109, 'id': 49, 'date': '30:21.109'},
            {'emotion': 'surprised', 'timestamp': 1738175421896, 'id': 50, 'date': '30:21.896'},
            {'emotion': 'neutral', 'timestamp': 1738175423279, 'id': 51, 'date': '30:23.279'},
            {'emotion': 'happy', 'timestamp': 1738175423445, 'id': 52, 'date': '30:23.445'},
            {'emotion': 'neutral', 'timestamp': 1738175424049, 'id': 53, 'date': '30:24.049'},
            {'emotion': 'surprised', 'timestamp': 1738175424939, 'id': 54, 'date': '30:24.939'},
            {'emotion': 'neutral', 'timestamp': 1738175428650, 'id': 55, 'date': '30:28.650'},
            {'emotion': 'surprised', 'timestamp': 1738175428755, 'id': 56, 'date': '30:28.755'},
            {'emotion': 'neutral', 'timestamp': 1738175429057, 'id': 57, 'date': '30:29.057'}
        ]
        expected = [
            {'date': '29:29.893', 'emotion': 'neutral', 'id': 1, 'timestamp': 1738175369893},
            {'date': '29:32.856', 'emotion': 'angry', 'id': 4, 'timestamp': 1738175372856},
            {'date': '29:33.854', 'emotion': 'neutral', 'id': 7, 'timestamp': 1738175373854},
            {'date': '29:35.199', 'emotion': 'angry', 'id': 10, 'timestamp': 1738175375199},
            {'date': '29:37.469', 'emotion': 'neutral', 'id': 13, 'timestamp': 1738175377469},
            {'date': '29:41.785', 'emotion': 'surprised', 'id': 16, 'timestamp': 1738175381785},
            {'date': '29:44.302', 'emotion': 'neutral', 'id': 19, 'timestamp': 1738175384302},
            {'date': '29:48.314', 'emotion': 'surprised', 'id': 22, 'timestamp': 1738175388314},
            {'date': '29:48.419', 'emotion': 'neutral', 'id': 23, 'timestamp': 1738175388419},
            {'date': '29:54.841', 'emotion': 'happy', 'id': 26, 'timestamp': 1738175394841},
            {'date': '29:57.595', 'emotion': 'neutral', 'id': 29, 'timestamp': 1738175397595},
            {'date': '30:00.873', 'emotion': 'happy', 'id': 32, 'timestamp': 1738175400873},
            {'date': '30:01.564', 'emotion': 'neutral', 'id': 33, 'timestamp': 1738175401564},
            {'date': '30:06.484', 'emotion': 'surprised', 'id': 36, 'timestamp': 1738175406484},
            {'emotion': 'happy', 'timestamp': 1738175408188, 'id': 39, 'date': '30:08.188'},
            {'emotion': 'neutral', 'timestamp': 1738175412225, 'id': 45, 'date': '30:12.225'},
            {'emotion': 'surprised', 'timestamp': 1738175419577, 'id': 48, 'date': '30:19.577'},
            {'emotion': 'neutral', 'timestamp': 1738175423279, 'id': 51, 'date': '30:23.279'},
            {'emotion': 'surprised', 'timestamp': 1738175424939, 'id': 54, 'date': '30:24.939'},
            {'emotion': 'neutral', 'timestamp': 1738175429057, 'id': 57, 'date': '30:29.057'}
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(expected, result)
