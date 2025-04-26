from ..helper_functions import find_phase_window, window_data

import unittest
from obspy import UTCDateTime
import numpy as np

class TestWindowData(unittest.TestCase):
    def test_window_data(self):
        time_array = np.array([1, 2, 3, 4, 5])
        data_array = np.array([[10, 20, 30, 40, 50], [100, 200, 300, 400, 500]])

        t_min = 2
        t_max = 4

        expected_filtered_time_array = np.array([2, 3, 4])
        expected_filtered_data_array = np.array([[20, 30, 40], [200, 300, 400]])

        result_time_array, result_data_array = window_data(time_array, data_array, t_min, t_max)

        np.testing.assert_array_equal(result_time_array, expected_filtered_time_array)
        np.testing.assert_array_equal(result_data_array, expected_filtered_data_array)

    def test_window_data_empty(self):
        time_array = np.array([1, 2, 3, 4, 5])
        data_array = np.array([[10, 20, 30, 40, 50], [100, 200, 300, 400, 500]])

        t_min = 6
        t_max = 8

        expected_filtered_time_array = np.array([])
        expected_filtered_data_array = np.array([[], []])

        result_time_array, result_data_array = window_data(time_array, data_array, t_min, t_max)

        np.testing.assert_array_equal(result_time_array, expected_filtered_time_array)
        np.testing.assert_array_equal(result_data_array, expected_filtered_data_array)

    def test_window_data_single_row(self):
        time_array = np.array([1, 2, 3, 4, 5])
        data_array = np.array([10, 20, 30, 40, 50])

        t_min = 2
        t_max = 4

        result_time_array, result_data_array = window_data(time_array, data_array, t_min, t_max)

        expected_filtered_time_array = np.array([2, 3, 4])
        expected_filtered_data_array = np.array([20, 30, 40])

        np.testing.assert_array_equal(result_time_array, expected_filtered_time_array)
        np.testing.assert_array_equal(result_data_array, expected_filtered_data_array)


class TestFindPhaseWindow(unittest.TestCase):
    def test_find_phase_window(self):
        event_depth = 10.0
        event_latitude = 35.0
        event_longitude = -120.0
        station_latitude = 34.0
        station_longitude = -118.0
        T = 10.0
        phase = 'P'

        expected_window_start = 27.833888861958044
        expected_window_end = 42.251682948590719

        result = find_phase_window(event_depth, event_latitude, event_longitude,
                                   station_latitude, station_longitude, T, phase)

        self.assertEqual(result[0], expected_window_start)
        self.assertEqual(result[1], expected_window_end)

    def test_find_phase_window_no_arrivals(self):
        event_depth = 10.0
        event_latitude = 0
        event_longitude = 0
        station_latitude = 0
        station_longitude = 160
        T = 10.0
        phase = 'S'

        with self.assertRaises(ValueError):
            find_phase_window(event_depth, event_latitude, event_longitude,
                              station_latitude, station_longitude, T, phase)