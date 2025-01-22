import unittest
from datetime import datetime, timedelta
from src.service import recording_service
from typing import List
import json


def get_test_measurement_data() -> List[float]:
    with open("test_measurements.json", "r") as f:
        return json.load(f)


class TestRecordingService(unittest.TestCase):

    def test_correct_number_of_measurements(self):
        parsed_data = get_test_measurement_data()

        processed_parsed_data = recording_service.process_parsed_data(
            parsed_data=parsed_data,
            recording_id=1,
            sample_rate=142,
            number_of_persisted_measurements=80_700,
            start_time=datetime(2025, 1, 22, 0, 0, 0),
            now=datetime(2025, 1, 22, 0, 10, 0)  # 10 minutes later
        )

        self.assertEqual(len(processed_parsed_data), len(parsed_data))
        self.assertListEqual(parsed_data, processed_parsed_data)

    def test_more_than_expected_measurements(self):
        parsed_data = get_test_measurement_data()

        sample_rate = 142
        passed_time: timedelta = timedelta(seconds=10)
        start_time = datetime(2025, 1, 22, 0, 0, 0)

        processed_parsed_data = recording_service.process_parsed_data(
            parsed_data=parsed_data,
            recording_id=1,
            sample_rate=sample_rate,
            number_of_persisted_measurements=0,
            start_time=start_time,
            now=start_time + passed_time
        )

        self.assertGreater(parsed_data, processed_parsed_data)
        self.assertEqual(len(processed_parsed_data), passed_time.seconds * sample_rate)

    def test_less_than_expected_measurements(self):
        parsed_data = get_test_measurement_data()

        sample_rate = 142
        number_of_persisted_measurements = 80_700
        start_time = datetime(2025, 1, 22, 0, 0, 0)
        passed_time = timedelta(minutes=10, seconds=10)

        processed_parsed_data = recording_service.process_parsed_data(
            parsed_data=parsed_data,
            recording_id=1,
            sample_rate=sample_rate,
            number_of_persisted_measurements=number_of_persisted_measurements,
            start_time=start_time,
            now=start_time + passed_time
        )

        self.assertGreater(processed_parsed_data, parsed_data)
        self.assertEqual(
            (passed_time.seconds * sample_rate) - number_of_persisted_measurements,
            len(processed_parsed_data)
        )

        for value in parsed_data:
            self.assertIn(value, processed_parsed_data)
            self.assertGreater(float(1), value)
            self.assertGreater(value, float(0))
