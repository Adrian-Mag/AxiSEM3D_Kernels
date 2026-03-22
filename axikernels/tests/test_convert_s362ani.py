"""
Tests for convert_s362ani_to_radius.py
=======================================
Validates the radius-based S362ANI NetCDF conversion.
"""

import os
import sys
import numpy as np
import numpy.testing as npt
import netCDF4 as nc
import pytest

# Allow importing the script from the example directory
EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "axisem3d_root", "AxiSEM3D", "examples", "adrian_kernel_3D",
)
EXAMPLE_DIR = os.path.normpath(EXAMPLE_DIR)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from convert_s362ani_to_radius import convert_s362ani, SOURCE_FILE, EARTH_RADIUS_M  # noqa: E402

SOURCE_NC = SOURCE_FILE   # original depth-based file


@pytest.fixture(scope="module")
def src_data():
    """Load the original source file for comparison."""
    with nc.Dataset(SOURCE_NC, "r") as ds:
        depth_km  = ds["depth"][:]
        latitude  = ds["latitude"][:]
        longitude = ds["longitude"][:]
        dvs       = ds["dvs"][:]
        dvsv      = ds["dvsv"][:]
        dvsh      = ds["dvsh"][:]
    return {
        "depth_km":  np.asarray(depth_km),
        "latitude":  np.asarray(latitude),
        "longitude": np.asarray(longitude),
        "dvs":       np.asarray(dvs),
        "dvsv":      np.asarray(dvsv),
        "dvsh":      np.asarray(dvsh),
    }


@pytest.fixture(scope="module")
def nc_file(tmp_path_factory, src_data):
    """Run the conversion into a temporary file."""
    tmp_dir = tmp_path_factory.mktemp("s362ani")
    output_path = str(tmp_dir / "S362ANI_radius.nc")
    convert_s362ani(source_file=SOURCE_NC, output_file=output_path)
    return output_path


def test_file_created(nc_file):
    """Output file must exist."""
    assert os.path.isfile(nc_file), f"File not found: {nc_file}"


def test_has_radius_not_depth(nc_file):
    """Output must have 'radius' variable, not 'depth'."""
    with nc.Dataset(nc_file, "r") as ds:
        assert "radius" in ds.variables, "Missing variable: radius"
        assert "depth" not in ds.variables, "Should not contain 'depth'"


def test_radius_monotonically_increasing(nc_file):
    """Radius axis must be strictly increasing."""
    with nc.Dataset(nc_file, "r") as ds:
        radius = ds["radius"][:].data
    diffs = np.diff(radius)
    assert np.all(diffs > 0), "Radius axis is not monotonically increasing"


def test_radius_values_match_depth_conversion(nc_file, src_data):
    """
    radius = EARTH_RADIUS_M - depth_km * 1000
    After flipping: radius[0] corresponds to the deepest original depth,
    radius[-1] to the shallowest.
    """
    depth_km = src_data["depth_km"]  # ascending: 25, 50, ..., 2890

    # Expected radius array (before flip): shallow → deep
    expected_radius_pre_flip = EARTH_RADIUS_M - depth_km * 1000.0
    # After flip (ascending): deepest first → shallowest last
    expected_radius = expected_radius_pre_flip[::-1]

    with nc.Dataset(nc_file, "r") as ds:
        radius = ds["radius"][:].data

    npt.assert_allclose(radius, expected_radius, rtol=1e-9,
                        err_msg="Radius values do not match depth conversion")


def test_first_and_last_radius(nc_file, src_data):
    """Check exact first and last radius values."""
    depth_km = src_data["depth_km"]
    with nc.Dataset(nc_file, "r") as ds:
        radius = ds["radius"][:].data

    # After flip, radius[0] = R_E - deepest_depth * 1000  (inner boundary)
    expected_first = EARTH_RADIUS_M - float(depth_km[-1]) * 1000.0
    # radius[-1] = R_E - shallowest_depth * 1000  (near Moho)
    expected_last  = EARTH_RADIUS_M - float(depth_km[0])  * 1000.0

    npt.assert_allclose(radius[0],  expected_first, rtol=1e-9)
    npt.assert_allclose(radius[-1], expected_last,  rtol=1e-9)


def test_data_variables_present(nc_file):
    """Output must contain dvs, dvsv, dvsh."""
    with nc.Dataset(nc_file, "r") as ds:
        for var in ("dvs", "dvsv", "dvsh"):
            assert var in ds.variables, f"Missing variable: {var}"


def test_data_shape(nc_file, src_data):
    """Data variables must have shape (n_radius, n_lat, n_lon)."""
    depth_km  = src_data["depth_km"]
    latitude  = src_data["latitude"]
    longitude = src_data["longitude"]
    expected  = (len(depth_km), len(latitude), len(longitude))

    with nc.Dataset(nc_file, "r") as ds:
        for var in ("dvs", "dvsv", "dvsh"):
            shape = ds[var][:].shape
            assert shape == expected, (
                f"{var}: expected shape {expected}, got {shape}"
            )


def test_data_values_match_original(nc_file, src_data):
    """
    A specific (lat, lon) slice at the shallowest depth in the converted file
    must equal the same slice in the original shallowest depth layer.

    Original layer 0 (depth=25 km → radius=6346000 m) becomes the last
    layer in the converted file after flipping.
    """
    dvs_src = src_data["dvs"]   # shape (25, 91, 181)

    with nc.Dataset(nc_file, "r") as ds:
        dvs_out = ds["dvs"][:].data  # shape (25, 91, 181)

    # Original depth[0] = 25 km → converted radius[-1] = 6346000 m
    npt.assert_allclose(
        dvs_out[-1, :, :],   # shallowest in converted (was depth index 0)
        dvs_src[0,  :, :],   # original depth index 0
        rtol=1e-9,
        err_msg="Shallowest layer values do not match between original and converted",
    )

    # Original depth[-1] = 2890 km → converted radius[0] = 3481000 m
    npt.assert_allclose(
        dvs_out[0,  :, :],   # deepest in converted
        dvs_src[-1, :, :],   # original deepest depth
        rtol=1e-9,
        err_msg="Deepest layer values do not match between original and converted",
    )
