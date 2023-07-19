import numpy as np


def sph2cart(point: np.ndarray) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        rad (float): The radius (must be non-negative).
        lat (float): The latitude in radians.
        lon (float): The longitude in radians.

    Returns:
        np.ndarray: An array containing the Cartesian coordinates [x, y, z].
    """
    rad, lat, lon = point

    if rad < 0:
        raise ValueError("Radius must be non-negative.")

    cos_lat = np.cos(lat)
    x = rad * cos_lat * np.cos(lon)
    y = rad * cos_lat * np.sin(lon)
    z = rad * np.sin(lat)

    return np.array([x, y, z])


def cart2sph(point: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
        point (np.ndarray): Array containing the Cartesian coordinates [x, y, z].
            - x (float): x-coordinate.
            - y (float): y-coordinate.
            - z (float): z-coordinate.

    Returns:
        np.ndarray: Array containing the spherical coordinates [rad, lat, lon].
            - rad (float): The radius.
            - lat (float): The latitude in radians.
            - lon (float): The longitude in radians.
    """
    x, y, z = point

    rad = np.sqrt(x * x + y * y + z * z)

    if x == 0.0:
        if y > 0:
            lon = np.pi / 2
        elif y < 0:
            lon = -np.pi / 2
        else:
            lon = 0
    else:
        lon = np.arctan2(y, x)

    if rad == 0.0:
        lat = 0
    else:
        lat = np.arcsin(z / rad)

    return np.array([rad, lat, lon])


def sph2cyl(point: list) -> list:
    """
    Convert spherical coordinates to cylindrical coordinates.

    Args:
        point (list): A list containing the spherical coordinates [r, theta, phi].
                      The angle values theta and phi should be in radians.

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


def cart_geo2cart_src(point: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    # Rotate coordinates from the Earth frame to the source frame
    if len(point) != 3:
        raise ValueError("Invalid input: 'point' list should contain 3 elements.")

    if rotation_matrix.shape != (3, 3):
        raise ValueError("Invalid input: 'rotation_matrix' should be a 3x3 numpy array.")

    return np.matmul(rotation_matrix.transpose(), point)


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
                     np.arctan2(z, s))
    r = np.sqrt(s**2 + z**2)

    return np.column_stack((r, theta))


def cart2cyl(point: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to cylindrical coordinates.

    Parameters:
        point (np.ndarray): Array containing the Cartesian coordinates [x, y, z].
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
    x, y, z = point

    s = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)

    return np.array([s, z, phi])