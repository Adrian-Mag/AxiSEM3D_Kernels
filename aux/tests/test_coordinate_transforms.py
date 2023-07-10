import numpy as np
from numpy.testing import assert_allclose
from math import pi
import unittest 

from ..coordinate_transforms import sph2cart, cart2sph, sph2cyl, cart_geo2cart_src, cart2polar, cart2cyl


class TestSph2Cart(unittest.TestCase):
    def test_sph2cart(self):
        points = np.array([
            [1, 0, 0],                  # Equator, prime meridian
            [1, np.pi/2, 0],            # North pole
            [1, np.pi/4, np.pi/4],       # 45 degrees latitude, 45 degrees longitude
            [1, np.pi/3, np.pi/6]        # 60 degrees latitude, 30 degrees longitude
        ])

        expected_results = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0.5, 0.5, 0.70710678],
            [0.4330127019, 0.25, 0.866025]
        ])

        for point, expected_result in zip(points, expected_results):
            with self.subTest(point=point):
                result = sph2cart(point)
                np.testing.assert_allclose(result, expected_result, atol=1e-6)


class TestCart2Sph(unittest.TestCase):
    def test_cart2sph(self):
        points = np.array([
            [1, 0, 0],                  # x-axis
            [0, 1, 0],                  # y-axis
            [0, 0, 1],                  # z-axis
            [1, 1, 1],                  # Point in first octant
            [-1, -1, -1],               # Point in seventh octant
            [0, 0, 0],                  # Origin
            [0, 1, 1],                  # x = 0, y > 0
            [0, -1, -1],                # x = 0, y < 0
            [2, 0, 0],                  # Non-zero x, y = 0
            [0, 0, 2],                  # Non-zero z, x = y = 0
        ])

        expected_results = np.array([
            [1, 0, 0],
            [1, 0, np.pi/2],
            [1, np.pi/2, 0],
            [np.sqrt(3), 0.61548, np.pi/4],
            [np.sqrt(3), -0.61548, -3*np.pi/4],
            [0, 0, 0],
            [np.sqrt(2), np.pi/4, np.pi/2],
            [np.sqrt(2), -np.pi/4, -np.pi/2],
            [2, 0, 0],
            [2, np.pi/2, 0],
        ])

        for point, expected_result in zip(points, expected_results):
            with self.subTest(point=point):
                result = cart2sph(point)
                np.testing.assert_allclose(result, expected_result, atol=1e-6)


class TestSph2Cyl(unittest.TestCase):
    def test_sph2cyl(self):
        points = [
            [1, 0, 0],              # Point on the positive x-axis
            [1, np.pi/2, 0],        # Point on the positive y-axis
            [1, np.pi, 0],          # Point on the negative x-axis
            [1, 3*np.pi/2, 0],      # Point on the negative y-axis
            [1, np.pi/4, np.pi/6],  # Point in the first quadrant
        ]

        expected_results = [
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [np.sqrt(2)/2, np.sqrt(2)/2, np.pi/6],
        ]

        for point, expected_result in zip(points, expected_results):
            with self.subTest(point=point):
                result = sph2cyl(point)
                np.testing.assert_allclose(result, expected_result, atol=1e-6)


class TestCart2Polar(unittest.TestCase):
    def test_cart2polar(self):
        s_values = np.array([1, 2, 0, 3])
        z_values = np.array([0, 0, 2, -4])

        expected_results = np.array([
            [1, 0],
            [2, 0],
            [2, np.pi/2],
            [5, -0.927295],
        ])

        result = cart2polar(s_values, z_values)
        np.testing.assert_allclose(result, expected_results, atol=1e-6)

    def test_cart2polar_negative_s(self):
        s_values = np.array([-1, 2, 3])
        z_values = np.array([0, 0, 0])

        with self.assertRaises(ValueError):
            cart2polar(s_values, z_values)


class TestCart2Cyl(unittest.TestCase):
    def test_cart2cyl(self):
        points = np.array([
            [1, 0, 0],                  # Point on the positive x-axis
            [0, 1, 0],                  # Point on the positive y-axis
            [-1, 0, 0],                 # Point on the negative x-axis
            [0, -1, 0],                 # Point on the negative y-axis
            [1, 1, 1],                  # Point in the first quadrant
            [0, 0, 0],                  # Origin
            [0, 1, 2],                  # x = 0, y > 0
            [0, -1, -2],                # x = 0, y < 0
        ])

        expected_results = np.array([
            [1, 0, 0],
            [1, 0, np.pi/2],
            [1, 0, np.pi],
            [1, 0, -np.pi/2],
            [np.sqrt(2), 1, np.pi/4],
            [0, 0, 0],
            [1, 2, np.pi/2],
            [1, -2, -np.pi/2],
        ])

        for point, expected_result in zip(points, expected_results):
            with self.subTest(point=point):
                result = cart2cyl(point)
                np.testing.assert_allclose(result, expected_result, atol=1e-6)
