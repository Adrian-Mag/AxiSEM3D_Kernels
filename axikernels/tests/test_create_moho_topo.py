"""
Tests for create_moho_topography.py
====================================
Validates the generated Moho undulation NetCDF file.
"""

import os
import sys
import tempfile
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

from create_moho_topography import create_moho_topography  # noqa: E402


@pytest.fixture(scope="module")
def nc_file(tmp_path_factory):
    """Generate the Moho topography NetCDF into a temp directory."""
    tmp_dir = tmp_path_factory.mktemp("moho")
    output_path = str(tmp_dir / "moho_topography.nc")
    create_moho_topography(output_path=output_path)
    return output_path


def test_file_created(nc_file):
    """Output file must exist."""
    assert os.path.isfile(nc_file), f"File not found: {nc_file}"


def test_variables_present(nc_file):
    """NetCDF must contain latitude, longitude, undulation_MOHO."""
    with nc.Dataset(nc_file, "r") as ds:
        assert "latitude" in ds.variables, "Missing variable: latitude"
        assert "longitude" in ds.variables, "Missing variable: longitude"
        assert "undulation_MOHO" in ds.variables, "Missing variable: undulation_MOHO"


def test_latitude_range(nc_file):
    """Latitude must span -90 to 90."""
    with nc.Dataset(nc_file, "r") as ds:
        lat = ds["latitude"][:].data
    npt.assert_allclose(lat[0],  -90.0, atol=1e-6)
    npt.assert_allclose(lat[-1],  90.0, atol=1e-6)


def test_longitude_range(nc_file):
    """Longitude must span -180 to 180."""
    with nc.Dataset(nc_file, "r") as ds:
        lon = ds["longitude"][:].data
    npt.assert_allclose(lon[0],  -180.0, atol=1e-6)
    npt.assert_allclose(lon[-1],  180.0, atol=1e-6)


def test_max_undulation_near_5000m(nc_file):
    """Peak undulation must be ≈ 5000 m (within ±100 m)."""
    with nc.Dataset(nc_file, "r") as ds:
        und = ds["undulation_MOHO"][:].data
    npt.assert_allclose(und.max(), 5000.0, atol=100.0)


def test_undulation_at_pole_near_zero(nc_file):
    """Undulation far from centre (lat=90°, lon=90°) must be ≈ 0."""
    with nc.Dataset(nc_file, "r") as ds:
        lat = ds["latitude"][:].data
        lon = ds["longitude"][:].data
        und = ds["undulation_MOHO"][:].data

    ilat = np.argmin(np.abs(lat - 90.0))
    ilon = np.argmin(np.abs(lon - 90.0))
    npt.assert_allclose(und[ilat, ilon], 0.0, atol=1.0)   # < 1 m


def test_undulation_shape(nc_file):
    """Data shape must match (n_lat, n_lon)."""
    with nc.Dataset(nc_file, "r") as ds:
        lat = ds["latitude"][:].data
        lon = ds["longitude"][:].data
        und = ds["undulation_MOHO"][:].data
    assert und.shape == (len(lat), len(lon))


def test_all_undulations_non_negative(nc_file):
    """Gaussian undulation must be ≥ 0 everywhere."""
    with nc.Dataset(nc_file, "r") as ds:
        und = ds["undulation_MOHO"][:].data
    assert np.all(und >= 0.0), "Found negative undulation values"


def test_peak_location_near_center(nc_file):
    """Peak undulation must be at the Gaussian centre (0°N, 20°E)."""
    with nc.Dataset(nc_file, "r") as ds:
        lat = ds["latitude"][:].data
        lon = ds["longitude"][:].data
        und = ds["undulation_MOHO"][:].data
    peak_idx = np.unravel_index(und.argmax(), und.shape)
    assert abs(lat[peak_idx[0]] - 0.0) <= 1.0, f"Peak lat={lat[peak_idx[0]]}"
    assert abs(lon[peak_idx[1]] - 20.0) <= 1.0, f"Peak lon={lon[peak_idx[1]]}"


def test_gaussian_footprint_at_30deg(nc_file):
    """Undulation at ~30° angular distance must be < 15% of peak (1/10 target)."""
    with nc.Dataset(nc_file, "r") as ds:
        lat = ds["latitude"][:].data
        lon = ds["longitude"][:].data
        und = ds["undulation_MOHO"][:].data
    # Point at (0°, 50°) is 30° from centre (0°, 20°)
    ilat = np.argmin(np.abs(lat - 0.0))
    ilon = np.argmin(np.abs(lon - 50.0))
    ratio = und[ilat, ilon] / und.max()
    assert ratio < 0.15, f"At 30° offset: {ratio:.3f} of peak (expect < 0.15)"
    assert ratio > 0.01, f"At 30° offset: {ratio:.3f} of peak (too small)"
