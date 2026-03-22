import numpy as np
import mpmath


def sph2cart(points: np.ndarray) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates for an array of
    points.
    Args:
        points (np.ndarray): An array of shape (N, 3) containing the spherical
        coordinates [rad, lat, lon].
    Returns:
        np.ndarray: An array containing the Cartesian coordinates [x, y, z] for
        each input point.
    """
    points = np.asarray(points, dtype=float)
    single_point = points.ndim == 1
    points = np.atleast_2d(points)
    rad, lat, lon = points.T  # Transpose to access each coordinate separately

    if np.any(rad < 0):
        raise ValueError("Radius must be non-negative.")

    cos_lat = np.cos(lat)
    x = rad * cos_lat * np.cos(lon)
    y = rad * cos_lat * np.sin(lon)
    z = rad * np.sin(lat)

    cartesian_coords = np.array([x, y, z]).T  # Transpose back to shape (N, 3)
    if single_point:
        return cartesian_coords.squeeze()
    return cartesian_coords


def sph2cart_mpmath(points: np.ndarray, precision=64) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates for an array of
    points.
    Args:
        points (np.ndarray): An array of shape (N, 3) containing the spherical
        coordinates [rad, lat, lon].
        precision (int): Number of decimal places for arbitrary precision
        (default is 64).
    Returns:
        np.ndarray: An array containing the Cartesian coordinates [x, y, z] for
        each input point.
    """
    mpmath.mp.dps = precision  # Set the decimal places for mpmath

    points = np.asarray(points, dtype=float)
    single_point = points.ndim == 1
    points = np.atleast_2d(points)
    rad, lat, lon = points.T  # Transpose to access each coordinate separately

    if np.any(rad < 0):
        raise ValueError("Radius must be non-negative.")

    x = np.array([
        mpmath.mpf(radius) * mpmath.cos(latitude) * mpmath.cos(longitude)
        for radius, latitude, longitude in zip(rad, lat, lon)
    ], dtype=object)
    y = np.array([
        mpmath.mpf(radius) * mpmath.cos(latitude) * mpmath.sin(longitude)
        for radius, latitude, longitude in zip(rad, lat, lon)
    ], dtype=object)
    z = np.array([
        mpmath.mpf(radius) * mpmath.sin(latitude)
        for radius, latitude in zip(rad, lat)
    ], dtype=object)

    cartesian_coords = np.array([x, y, z]).T  # Transpose back to shape (N, 3)
    if single_point:
        return cartesian_coords.squeeze()
    return cartesian_coords


def cart2sph(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
        point (np.ndarray): Array containing the Cartesian coordinates [x, y,
        z].
            - x (float): x-coordinate.
            - y (float): y-coordinate.
            - z (float): z-coordinate.

    Returns:
        np.ndarray: Array containing the spherical coordinates [rad, lat, lon].
            - rad (float): The radius.
            - lat (float): The latitude in radians.
            - lon (float): The longitude in radians.
    """
    points = np.asarray(points, dtype=float)
    single_point = points.ndim == 1
    points = np.atleast_2d(points)

    # Extract x, y, z coordinates from the input array
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Calculate radius
    radius = np.sqrt(x**2 + y**2 + z**2)

    # Calculate inclination angle (theta); origin maps to 0
    with np.errstate(invalid='ignore'):
        inclination = np.where(radius == 0, 0.0, np.arcsin(z / radius))

    # Calculate azimuthal angle (phi)
    azimuth = np.arctan2(y, x)

    # Combine the spherical coordinates into a single ndarray
    spherical_coords = np.column_stack((radius, inclination, azimuth))

    if single_point:
        return spherical_coords.squeeze()
    return spherical_coords


def cart2sph_mpmath(points: np.ndarray, precision=64) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
        point (np.ndarray): Array containing the Cartesian coordinates [x, y,
        z].
            - x (float): x-coordinate.
            - y (float): y-coordinate.
            - z (float): z-coordinate.
        precision (int): Number of decimal places for arbitrary precision
        (default is 64).

    Returns:
        np.ndarray: Array containing the spherical coordinates [rad, lat, lon].
            - rad (float): The radius.
            - lat (float): The latitude in radians.
            - lon (float): The longitude in radians.
    """
    mpmath.mp.dps = precision  # Set the decimal places for mpmath

    points = np.asarray(points, dtype=float)
    single_point = points.ndim == 1
    points = np.atleast_2d(points)

    # Extract x, y, z coordinates from the input array
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Calculate radius using mpmath
    radius = np.array([
        mpmath.sqrt(x_val ** 2 + y_val ** 2 + z_val ** 2)
        for x_val, y_val, z_val in zip(x, y, z)
    ], dtype=object)

    # Calculate inclination angle (theta) using mpmath
    inclination = np.array([
        mpmath.mpf("0.0") if rad == 0 else mpmath.asin(z_val / rad)
        for z_val, rad in zip(z, radius)
    ], dtype=object)

    # Calculate azimuthal angle (phi) using mpmath
    azimuth = np.array([
        mpmath.atan2(y_val, x_val) for x_val, y_val in zip(x, y)
    ], dtype=object)

    # Combine the spherical coordinates into a single ndarray
    spherical_coords = np.column_stack((radius, inclination, azimuth))

    if single_point:
        return spherical_coords.squeeze()
    return spherical_coords


def sph2cyl(point: list) -> list:
    """
    Convert spherical coordinates to cylindrical coordinates.

    Args:
        point (list): A list containing the spherical coordinates [r, theta,
        phi]. The angle values theta and phi should be in radians.

    Returns:
        list: A list containing the cylindrical coordinates [s, z, phi].

    Raises:
        ValueError: If the radial position (r) is negative.
    """
    if point[0] < 0:
        raise ValueError("Radial position must be positive")

    r = point[0]
    theta = point[1]
    phi = point[2]

    s = r * np.cos(theta)
    z = r * np.sin(theta)

    return np.array([s, z, phi])


def cart_geo2cart_src(points: np.ndarray,
                      rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Rotate coordinates from the Earth frame to the source frame for an array of
    points.
    Args:
        points (np.ndarray): An array of shape (N, 3) containing the Cartesian
        coordinates [x, y, z] in the Earth frame.
        rotation_matrix (np.ndarray): A 3x3 numpy array representing the
        rotation matrix.
    Returns:
        np.ndarray: An array containing the rotated Cartesian coordinates [x',
        y', z'] for each input point.
    """
    if points.shape[-1] != 3:
        raise ValueError("Invalid input: 'points' array"
                         " should have shape (N, 3).")

    if rotation_matrix.shape != (3, 3):
        raise ValueError("Invalid input: 'rotation_matrix'"
                         "should be a 3x3 numpy array.")

    rotated_points = np.matmul(rotation_matrix.T, points.T)
    return rotated_points.T


def cart2polar(s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Transform in-plane cylindrical coordinates (cartesian) to polar
    coordinates.

    Args:
        s (np.ndarray): Distance from cylindrical axis in meters.
        z (np.ndarray): Distance along cylindrical axis in meters.

    Returns:
        np.ndarray: Array containing [radius, theta] in meters and radians.
    Raises:
        ValueError: If any element in `s` is not positive.

    """
    if np.any(s < 0):
        raise ValueError("Distance `s` must be non-negative.")

    theta = np.where(s == 0,
                     np.where(z > 0, np.pi / 2,
                              np.where(z < 0, -np.pi / 2, 0)),
                     np.arctan2(z, s))
    r = np.sqrt(s**2 + z**2)

    return np.column_stack((r, theta))


def cart2polar_mpmath(
    s: np.ndarray, z: np.ndarray, precision=64
) -> np.ndarray:
    """
    Transform in-plane cylindrical coordinates (cartesian) to polar
    coordinates.

    Args:
        s (np.ndarray): Distance from cylindrical axis in meters.
        z (np.ndarray): Distance along cylindrical axis in meters.

    Returns:
        np.ndarray: Array containing [radius, theta] in meters and radians.
    Raises:
        ValueError: If any element in `s` is not positive.

    """
    mpmath.mp.dps = precision  # Set the decimal places for mpmath

    if np.any(s < 0):
        raise ValueError("Distance `s` must be non-negative.")

    theta = np.where(s == 0,
                     np.where(z > 0, np.pi / 2,
                              np.where(z < 0, -np.pi / 2, 0)),
                     np.array([
                         mpmath.atan2(z[i], s)
                         for i, s in enumerate(s)
                     ]))
    r = np.sqrt(s**2 + z**2)

    return np.column_stack((r, theta))


def cart2cyl(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to cylindrical coordinates.

    Parameters:
        point (np.ndarray): Array containing the Cartesian coordinates [x, y,
        z].
            - x (float): x-coordinate.
            - y (float): y-coordinate.
            - z (float): z-coordinate.

    Returns:
        np.ndarray: Array containing the cylindrical coordinates [s, phi, z].
            - s (float): Distance from the cylindrical axis.
            - phi (float): Angle in radians measured from the positive x-axis.
            - z (float): Distance along the cylindrical axis.

    Raises:
        None.

    """
    x, y, z = points.T

    s = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)

    return np.array([s, z, phi]).T


def cart2cyl_mpmath(points: np.ndarray, precision=64) -> np.ndarray:
    """
    Convert Cartesian coordinates to cylindrical coordinates.

    Parameters:
        point (np.ndarray): Array containing the Cartesian coordinates [x, y,
        z].
            - x (float): x-coordinate.
            - y (float): y-coordinate.
            - z (float): z-coordinate.

    Returns:
        np.ndarray: Array containing the cylindrical coordinates [s, phi, z].
            - s (float): Distance from the cylindrical axis.
            - phi (float): Angle in radians measured from the positive x-axis.
            - z (float): Distance along the cylindrical axis.

    Raises:
        None.

    """
    mpmath.mp.dps = precision  # Set the decimal places for mpmath

    x, y, z = points.T

    s = np.array([mpmath.sqrt(val) for val in x*x + y*y])
    phi = np.array([mpmath.atan2(y[i], x) for i, x in enumerate(x)])

    return np.array([s, z, phi]).T
