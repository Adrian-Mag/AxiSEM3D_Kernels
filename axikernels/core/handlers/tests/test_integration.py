"""
test_integration.py – End-to-end integration tests against real AxiSEM3D
simulation data from the handlers_demo example.

All tests are skipped if the demo simulation data is not present.
"""
import os
from math import radians

import numpy as np
import obspy
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
_DEMO_ELEMENTS = os.path.join(
    _REPO_ROOT, 'examples', 'handlers_demo', 'sim_run', 'output', 'elements'
)
_DEMO_STATIONS = os.path.join(
    _REPO_ROOT, 'examples', 'handlers_demo', 'sim_run', 'input', 'GSN_small.txt'
)

# Skip all tests when demo data is absent
pytestmark = pytest.mark.skipif(
    not os.path.isdir(_DEMO_ELEMENTS),
    reason="handlers_demo simulation data not found"
)

# ---------------------------------------------------------------------------
# Module-scoped fixture (expensive I/O done once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_eo():
    from axikernels.core.handlers.element_output import ElementOutput
    return ElementOutput(path_to_element_output=_DEMO_ELEMENTS)


# ---------------------------------------------------------------------------
# Test 1 – mantle surface point, nontrivial waveform
# ---------------------------------------------------------------------------

def _expected_npts(eo):
    """Derive expected sample count from first element group's time axis."""
    first_group = next(iter(eo.element_groups_info))
    return len(eo.element_groups_info[first_group]['metadata']['data_time'])


def test_load_data_mantle_nontrivial(demo_eo):
    """Surface point in mantle returns a valid waveform with >100 unique values."""
    npts = _expected_npts(demo_eo)
    point = np.array([[6371000.0, radians(30), 0.0]])
    data = demo_eo.load_data(points=point, channels=['UZ'], in_deg=False)

    assert data.shape == (1, 1, npts), f"Expected shape (1, 1, {npts}), got {data.shape}"
    assert not np.any(np.isnan(data)), "Data contains NaN"
    unique_vals = np.unique(data)
    assert len(unique_vals) > 100, (
        f"Expected >100 unique values (non-trivial waveform), got {len(unique_vals)}"
    )


# ---------------------------------------------------------------------------
# Test 2 – outer core point, nontrivial waveform
# ---------------------------------------------------------------------------

def test_load_data_outer_core_nontrivial(demo_eo):
    """Point in the outer core returns a valid waveform with >100 unique values."""
    npts = _expected_npts(demo_eo)
    point = np.array([[2500000.0, radians(10), 0.0]])
    data = demo_eo.load_data(points=point, channels=['UZ'], in_deg=False)

    assert data.shape == (1, 1, npts), f"Expected shape (1, 1, {npts}), got {data.shape}"
    assert not np.any(np.isnan(data)), "Outer core data contains NaN"
    unique_vals = np.unique(data)
    assert len(unique_vals) > 100, (
        f"Expected >100 unique values in outer core, got {len(unique_vals)}"
    )


# ---------------------------------------------------------------------------
# Test 3 – multi-channel retrieval
# ---------------------------------------------------------------------------

def test_load_data_multi_channel(demo_eo):
    """Retrieving 3 channels returns correct shape and non-trivial data per channel."""
    npts = _expected_npts(demo_eo)
    point = np.array([[5000000.0, radians(20), 0.0]])
    channels = ['UR', 'UT', 'UZ']
    data = demo_eo.load_data(points=point, channels=channels, in_deg=False)

    assert data.shape == (1, 3, npts), f"Expected shape (1, 3, {npts}), got {data.shape}"
    assert not np.any(np.isnan(data)), "Multi-channel data contains NaN"
    for ch_idx, ch in enumerate(channels):
        unique_vals = np.unique(data[0, ch_idx, :])
        assert len(unique_vals) > 50, (
            f"Channel {ch} has only {len(unique_vals)} unique values (expected >50)"
        )


# ---------------------------------------------------------------------------
# Test 4 – batch of points spanning the mantle
# ---------------------------------------------------------------------------

def test_load_data_batch_points(demo_eo):
    """Batch of 5 mantle points returns correct shape; distinct waveforms."""
    radii = [4000000.0, 4500000.0, 5000000.0, 5500000.0, 6371000.0]
    colats = [radians(v) for v in [5, 10, 20, 30, 45]]
    points = np.array([[r, t, 0.0] for r, t in zip(radii, colats)])

    data = demo_eo.load_data(points=points, channels=['UZ'], in_deg=False)

    npts = _expected_npts(demo_eo)
    assert data.shape == (5, 1, npts), f"Expected shape (5, 1, {npts}), got {data.shape}"

    # Count how many points have non-NaN, non-trivial data
    unique_counts = []
    for i in range(5):
        if not np.all(np.isnan(data[i, 0, :])):
            unique_counts.append(len(np.unique(data[i, 0, :])))
    assert len(unique_counts) >= 4, (
        f"Expected at least 4 of 5 points inside domain, got {len(unique_counts)}"
    )
    good_count = sum(1 for u in unique_counts if u > 50)
    assert good_count >= 4, (
        f"Expected at least 4 points with >50 unique values, got {good_count}"
    )

    # Check distinct waveforms for interior-domain points
    inside_waveforms = [
        data[i, 0, :] for i in range(5) if not np.all(np.isnan(data[i, 0, :]))
    ]
    if len(inside_waveforms) >= 2:
        # At least one pair of waveforms must differ
        any_distinct = any(
            not np.allclose(inside_waveforms[j], inside_waveforms[k], atol=1e-10)
            for j in range(len(inside_waveforms))
            for k in range(j + 1, len(inside_waveforms))
        )
        assert any_distinct, "All inside-domain waveforms are identical (unexpected)"


# ---------------------------------------------------------------------------
# Test 5 – stream() returns valid obspy.Stream
# ---------------------------------------------------------------------------

def test_stream_produces_valid_obspy(demo_eo):
    """stream() at a surface point produces 3 traces with correct metadata."""
    points = np.array([[6371000.0, 30.0, 0.0]])  # degrees, coord_in_deg=True
    channels = ['UR', 'UT', 'UZ']
    st = demo_eo.stream(points=points, channels=channels, coord_in_deg=True)

    assert isinstance(st, obspy.Stream), "stream() did not return an obspy.Stream"
    assert len(st) == 3, f"Expected 3 traces, got {len(st)}"

    returned_channels = [tr.stats.channel for tr in st]
    assert returned_channels == channels, (
        f"Expected channels {channels}, got {returned_channels}"
    )

    npts = _expected_npts(demo_eo)
    for tr in st:
        assert tr.stats.npts == npts, (
            f"Expected npts={npts}, got {tr.stats.npts} for channel {tr.stats.channel}"
        )
        assert tr.stats.delta > 0, (
            f"delta must be positive, got {tr.stats.delta}"
        )
        assert not np.all(tr.data == 0), (
            f"Channel {tr.stats.channel} data is all zeros"
        )


# ---------------------------------------------------------------------------
# Test 6 – create_inventory() returns valid obspy.Inventory
# ---------------------------------------------------------------------------

def test_create_inventory_produces_valid_inventory(demo_eo):
    """create_inventory() returns a well-formed inventory; source coords are numbers."""
    pytest.importorskip("obspy")

    if not os.path.isfile(_DEMO_STATIONS):
        pytest.skip("GSN_small.txt station file not found")

    inv = demo_eo.create_inventory(_DEMO_STATIONS)

    assert isinstance(inv, obspy.core.inventory.Inventory), (
        "create_inventory() did not return an obspy.Inventory"
    )
    assert len(inv.networks) >= 1, "Inventory has no networks"
    total_stations = sum(len(net.stations) for net in inv.networks)
    assert total_stations >= 1, "Inventory has no stations"

    # Source coordinates are numeric attributes on the object
    assert isinstance(demo_eo.source_lat, (int, float)), (
        f"source_lat is not a number: {demo_eo.source_lat!r}"
    )
    assert isinstance(demo_eo.source_lon, (int, float)), (
        f"source_lon is not a number: {demo_eo.source_lon!r}"
    )


# ---------------------------------------------------------------------------
# Test 7 – point outside domain returns NaN
# ---------------------------------------------------------------------------

def test_load_data_outside_domain_returns_nan(demo_eo):
    """A point above the surface of the earth (outside all domains) returns all-NaN."""
    # 7000 km radius is above Domain_Radius=6371km — no mesh elements exist there
    point = np.array([[7000000.0, radians(5), 0.0]])
    data = demo_eo.load_data(points=point, channels=['UZ'], in_deg=False)

    npts = _expected_npts(demo_eo)
    assert data.shape == (1, 1, npts), f"Expected shape (1, 1, {npts}), got {data.shape}"
    assert np.all(np.isnan(data)), (
        "Expected all-NaN for a point outside the simulation domain"
    )


# ---------------------------------------------------------------------------
# Test 8 – metadata self-consistency
# ---------------------------------------------------------------------------

def test_metadata_consistency(demo_eo):
    """Element group metadata is internally self-consistent."""
    # Domain radius
    assert abs(demo_eo.Domain_Radius - 6371000.0) < 1.0, (
        f"Domain_Radius expected 6371000.0, got {demo_eo.Domain_Radius}"
    )

    # Source coordinates are numbers
    assert isinstance(demo_eo.source_lat, (int, float))
    assert isinstance(demo_eo.source_lon, (int, float))

    # All element groups have the same time axis length
    time_lengths = [
        len(demo_eo.element_groups_info[g]['metadata']['data_time'])
        for g in demo_eo.element_groups_info
    ]
    assert len(set(time_lengths)) == 1, (
        f"Element groups have inconsistent time axis lengths: {time_lengths}"
    )
    assert time_lengths[0] > 0, (
        f"Expected positive time axis length, got {time_lengths[0]}"
    )

    # Vertical ranges don't overlap: mantle > outer_core > inner_core
    # Each group has vertical_range [min, max] in meters
    ranges = {}
    for g in demo_eo.element_groups_info:
        elems = demo_eo.element_groups_info[g].get('elements', {})
        vr = elems.get('vertical_range', None)
        if vr is not None:
            ranges[g] = vr

    if 'mantle' in ranges and 'outer_core' in ranges:
        assert ranges['mantle'][0] >= ranges['outer_core'][1], (
            f"Mantle and outer_core overlap: mantle={ranges['mantle']}, "
            f"outer_core={ranges['outer_core']}"
        )
    if 'outer_core' in ranges and 'inner_core' in ranges:
        assert ranges['outer_core'][0] >= ranges['inner_core'][1], (
            f"Outer core and inner core overlap: outer_core={ranges['outer_core']}, "
            f"inner_core={ranges['inner_core']}"
        )
