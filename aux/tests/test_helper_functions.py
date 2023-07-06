from ..helper_functions import find_phase_window
import pytest

def test_find_phase_window():
    # Test case 1: Valid inputs, arrivals fit in the window
    event_depth = 10.0
    event_latitude = 0.0
    event_longitude = 10.0
    station_latitude = 0.0
    station_longitude = 45.0
    T = 10.0
    phase = "P"
    result = find_phase_window(event_depth, event_latitude, event_longitude, 
                               station_latitude, station_longitude, T, phase)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] < result[1]

    # Additional test cases...
    # TODO: Add more test cases as needed