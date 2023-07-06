import numpy as np


def sph2cart(rad: float, lat: float, lon: float) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        rad (float): The radius (must be non-negative).
        lat (float): The latitude in radians.
        lon (float): The longitude in radians.

    Returns:
        np.ndarray: An array containing the Cartesian coordinates [x, y, z].
    """
    if rad < 0:
        raise ValueError("Radius must be non-negative.")

    cos_lat = np.cos(lat)
    x = rad * cos_lat * np.cos(lon)
    y = rad * cos_lat * np.sin(lon)
    z = rad * np.sin(lat)

    return np.asarray([x, y, z])


def cart2sph(x: float, y: float, z: float) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
    - x (float): x-coordinate.
    - y (float): y-coordinate.
    - z (float): z-coordinate.

    Returns:
    - np.ndarray: Array containing the spherical coordinates [rad, lat, lon].

    Raises:
    - ValueError: If the radius (rad) is zero or if x is zero.
    """

    if x == 0.0:
        raise ValueError("Invalid input: x-coordinate cannot be zero.")

    rad = np.sqrt(x*x + y*y + z*z)
    if rad == 0.0:
        raise ValueError("Invalid input: radius (rad) cannot be zero.")

    lat = np.arcsin(z / rad)
    lon = np.arctan2(y, x)
    
    return np.asarray([rad, lat, lon])


def geo2cyl(point: list, rotation_matrix: np.ndarray) -> list:
    """
    Convert geographical coordinates to cylindrical coordinates in the
    seismic frame.

    Args:
        point (list): [radial position in m, latitude in rad, longitude in rad]
        rotation_matrix (np.ndarray): Rotation matrix to transform coordinates.

    Returns:
        list: [radial position in m, vertical position in m, azimuth from
        source in rad]

    Raises:
        ValueError: If the point list is not of length 3.
        ValueError: If the rotation matrix is not of shape (3, 3).
        ValueError: If the radial position is negative
    """
    if len(point) != 3:
        raise ValueError("Invalid input: 'point' list should contain 3 elements.")

    if rotation_matrix.shape != (3, 3):
        raise ValueError("Invalid input: 'rotation_matrix' should be a 3x3 numpy array.")

    if point[0] < 0:
        raise ValueError("Radial position must be positive")

    radial_pos = point[0]
    latitude = point[1]
    longitude = point[2]

    cos_lat = np.cos(latitude)
    sin_lat = np.sin(latitude)
    cos_lon = np.cos(longitude)
    sin_lon = np.sin(longitude)

    # Transform geographical coordinates to Cartesian coordinates in the
    # Earth frame
    x_earth = radial_pos * cos_lat * cos_lon
    y_earth = radial_pos * cos_lat * sin_lon
    z_earth = radial_pos * sin_lat

    # Rotate coordinates from the Earth frame to the source frame
    rotated_coords = np.matmul(rotation_matrix.transpose(), np.asarray([x_earth, y_earth, z_earth]))
    x, y, z = rotated_coords

    # Convert to cylindrical coordinates in the source frame
    s = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return [s, z, phi]


def cart2polar(s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Transform in-plane cylindrical coordinates (cartesian) to polar coordinates.

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
                     np.where(z > 0, np.pi / 2, np.where(z < 0, -np.pi / 2, 0)),
                     np.arctan(z / s))
    r = np.sqrt(s**2 + z**2)

    return np.column_stack((r, theta))