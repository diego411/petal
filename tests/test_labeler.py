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
        self.assertEqual(result, observations)

    def test_merge_single_gap(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "happy", "timestamp": 2000},
            {"emotion": "sad", "timestamp": 5000},
        ]
        expected = [
            {"emotion": "happy", "timestamp": 2000},
            {"emotion": "sad", "timestamp": 5000},
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(result, expected)

    def test_merge_multiple_gaps(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "happy", "timestamp": 2000},
            {"emotion": "happy", "timestamp": 4000},
            {"emotion": "sad", "timestamp": 10000},
        ]
        expected = [
            {"emotion": "happy", "timestamp": 4000},
            {"emotion": "sad", "timestamp": 10000},
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(result, expected)

    def test_no_emotions_merge_with_large_gap(self):
        observations = [
            {"emotion": "happy", "timestamp": 0},
            {"emotion": "happy", "timestamp": 5000},
        ]
        result = merge_observations(observations, gap_threshold=3000)
        self.assertEqual(result, expected)
