import numpy as np
from numpy.testing import assert_allclose
from math import pi

from ..coordinate_transforms import sph2cart, cart2sph, geo2cyl, cart2polar


def test_sph2cart() -> None:
    # Test case 1: Zero radius
    rad = 0
    lat = pi / 4  # 45 degrees
    lon = pi / 2  # 90 degrees
    expected_result = np.array([0, 0, 0])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result)

    # Test case 2: Non-zero radius and lat/lon
    rad = 1
    lat = pi / 3  # 60 degrees
    lon = pi / 4  # 45 degrees
    expected_result = np.array([0.353553, 0.353553, 0.866025])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 3: Negative radius and lat/lon (should raise ValueError)
    rad = -1
    lat = -pi / 6  # -30 degrees
    lon = -pi / 3  # -60 degrees
    try:
        result = sph2cart(rad, lat, lon)
        assert False  # The line above should raise an exception, so this line should not be reached
    except ValueError:
        assert True

    # Test case 4: Large radius and lat/lon
    rad = 1000
    lat = 0
    lon = pi
    expected_result = np.array([-1000, 0, 0])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 5: Radius and lat/lon near boundaries
    rad = 1e-6
    lat = -pi / 2 + 1e-9  # Just below the south pole
    lon = pi  # 180 degrees
    expected_result = np.array([0, 0, -1e-6])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 6: Radius and lat/lon at the north pole
    rad = 2
    lat = pi / 2  # 90 degrees
    lon = 0  # 0 degrees
    expected_result = np.array([0, 0, 2])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 7: Radius and lat/lon at the prime meridian
    rad = 3
    lat = 0  # 0 degrees
    lon = 0  # 0 degrees
    expected_result = np.array([3, 0, 0])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Additional test cases...
    # TODO: Add more test cases as needed

def test_cart2sph() -> None:
    # Test case 1: Cartesian coordinates at the origin (should raise ValueError)
    x = 0.0
    y = 0.0
    z = 0.0
    try:
        result = cart2sph(x, y, z)
        assert False  # The line above should raise an exception, so this line should not be reached
    except ValueError:
        assert True

    # Test case 2: Non-zero Cartesian coordinates
    x = 1.0
    y = 1.0
    z = 1.0
    expected_result = np.array([np.sqrt(3.0), 0.6154797087, np.pi/4])
    result = cart2sph(x, y, z)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 3: Negative Cartesian coordinates
    x = -2.0
    y = 0.0
    z = 1.5
    expected_result = np.array([np.sqrt(6.25), 0.6435011088, np.arctan2(0, -2)])
    result = cart2sph(x, y, z)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 4: Cartesian coordinates with different magnitudes
    x = 0.5
    y = 2.0
    z = 0.1
    expected_result = np.array([np.sqrt(4.26), np.arcsin(0.1/np.sqrt(4.26)), np.arctan2(2.0, 0.5)])
    result = cart2sph(x, y, z)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 5: Cartesian coordinates on the x-y plane
    x = -3.0
    y = 4.0
    z = 0.0
    expected_result = np.array([5.0, 0.0, np.arctan2(4.0, -3.0)])
    result = cart2sph(x, y, z)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 6: Cartesian coordinates on the z-axis
    x = 0.0
    y = 0.0
    z = -2.5
    try:
        result = cart2sph(x, y, z)
        assert False  # The line above should raise an exception, so this line should not be reached
    except ValueError:
        assert True

    # Test case 7: Zero x-coordinate (should raise ValueError)
    x = 0.0
    y = 1.0
    z = 1.0
    try:
        result = cart2sph(x, y, z)
        assert False  # The line above should raise an exception, so this line should not be reached
    except ValueError:
        assert True

    # Additional test cases...
    # TODO: Add more test cases as needed

def test_geo2cyl() -> None:
    # Test case 1: Positive radial position
    point = [1000, np.pi / 4, np.pi / 2]  # [radial position, latitude, longitude]
    rotation_matrix = np.eye(3)  # Identity matrix
    expected_result = [707.106781, 707.106781, np.pi / 2]  # [s, z, phi]
    result = geo2cyl(point, rotation_matrix)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 2: Negative radial position (should raise exception)
    point = [-500, np.pi / 6, np.pi / 4]  # [radial position, latitude, longitude]
    rotation_matrix = np.eye(3)  # Identity matrix
    try:
        result = geo2cyl(point, rotation_matrix)
        assert False  # The line above should raise an exception, so this line should not be reached
    except ValueError:
        assert True

    # Test case 3: Zero radial position (should raise exception)
    point = [0, np.pi / 3, np.pi / 6]  # [radial position, latitude, longitude]
    rotation_matrix = np.eye(3)  # Identity matrix
    try:
        result = geo2cyl(point, rotation_matrix)
        assert False  # The line above should raise an exception, so this line should not be reached
    except Exception:
        assert True

    # Test case 4: Zero latitude and longitude
    point = [100, 0, 0]  # [radial position, latitude, longitude]
    rotation_matrix = np.eye(3)  # Identity matrix
    expected_result = [100, 0, 0]  # [s, z, phi]
    result = geo2cyl(point, rotation_matrix)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 5: Nonzero radial position, latitude, and longitude with rotation of 30 degrees about each axis
    point = [200, np.pi / 6, np.pi / 3]  # [radial position, latitude, longitude]
    rotation_matrix = np.array([[0.8660254, -0.5, 0], [0.5, 0.8660254, 0], [0, 0, 1]])  # Rotation matrix for 30 degrees about each axis
    expected_result = [100 * np.sqrt(3), 100, np.pi / 6]  # [s, z, phi]
    result = geo2cyl(point, rotation_matrix)
    assert_allclose(result, expected_result, atol=1e-6)

    # Additional test cases...
    # TODO: Add more test cases as needed


def test_cart2polar() -> None:
    # Test case 1: Positive s and z
    s = np.array([1, 2])
    z = np.array([2, 3])
    expected_result = np.array([[np.sqrt(5), np.arctan(2 / 1)],
                                [np.sqrt(13), np.arctan(3 / 2)]])
    result = cart2polar(s, z)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 2: Zero s and positive z
    s = np.array([0, 0])
    z = np.array([3, 4])
    expected_result = np.array([[3, np.pi / 2],
                                [4, np.pi / 2]])
    result = cart2polar(s, z)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 3: Zero s and negative z
    s = np.array([0, 0])
    z = np.array([-4, -5])
    expected_result = np.array([[4, -np.pi / 2],
                                [5, -np.pi / 2]])
    result = cart2polar(s, z)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 4: Zero s and zero z
    s = np.array([0, 0])
    z = np.array([0, 0])
    expected_result = np.array([[0, 0],
                                [0, 0]])
    result = cart2polar(s, z)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 5: Negative s (should raise exception)
    s = np.array([-1, 2])
    z = np.array([5, 6])
    try:
        result = cart2polar(s, z)
        assert False  # The line above should raise an exception, so this line should not be reached
    except ValueError:
        assert True

    # Additional test cases...
    # TODO: Add more test cases as needed