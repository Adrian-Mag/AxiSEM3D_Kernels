"""
test_obspy_workflows.py – TDD tests for AxiSEM3D ObsPy workflows.

Tests are organised in three classes:
  1. TestStationOutputCore      – unit tests for StationOutput internals
  2. TestStationOutputObspy     – integration tests: stream, inventory, obspyfy
  3. TestObspyfiedOutput        – loading previously-obspyfied data via
                                  ObspyfiedOutput

All tests run against the ``mini_sim`` pytest fixture (synthetic data generated
by conftest.py) so that no real simulation output needs to be committed or run.

Element output is exercised only at the import / instantiation level for Phase 1
(real element-group output would require a much larger fixture).
"""
import os
import shutil

import numpy as np
import obspy
import pytest
from obspy.core.event import Catalog
from obspy.core.inventory import Inventory

from axikernels.core.handlers.station_output import StationOutput
from axikernels.core.handlers.obspy_output import ObspyfiedOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sta_output_path(sim_dir: str) -> str:
    return os.path.join(sim_dir, "output", "stations", "Station_grid")


# ===========================================================================
# 1. Core StationOutput behaviour
# ===========================================================================

class TestStationOutputCore:
    """Unit tests for StationOutput loading and metadata."""

    def test_init_station_output(self, mini_sim):
        """StationOutput initialises without error from a valid output dir."""
        so = StationOutput(_sta_output_path(mini_sim))
        assert so is not None

    def test_simulation_path_resolved(self, mini_sim):
        """The simulation root is correctly inferred from the output path."""
        so = StationOutput(_sta_output_path(mini_sim))
        assert os.path.samefile(so.path_to_simulation, mini_sim)

    def test_station_group_name(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        assert so.station_group_name == "Station_grid"

    def test_coordinate_frame_rtz(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        assert so.coordinate_frame == "RTZ"

    def test_channels_derived_from_inparam(self, mini_sim):
        """channels list matches 'U' → ['U'] from inparam.output.yaml."""
        so = StationOutput(_sta_output_path(mini_sim))
        assert "U" in so.channels

    def test_detailed_channels_rtz_expansion(self, mini_sim):
        """channels=[U] with RTZ frame → ['UR', 'UT', 'UZ']."""
        so = StationOutput(_sta_output_path(mini_sim))
        assert so.detailed_channels == ["UR", "UT", "UZ"]

    def test_time_axis_shape(self, mini_sim):
        """The loaded time axis contains 400 samples (fixture parameter)."""
        so = StationOutput(_sta_output_path(mini_sim))
        assert len(so.data_time) == 400

    def test_time_axis_monotone(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        assert np.all(np.diff(so.data_time) > 0)

    def test_rank_list_loaded(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        assert len(so._rank_list) == 5  # 5 stations in fixture

    def test_load_data_at_station(self, mini_sim):
        """load_data_at_station returns (n_channels, n_times) array."""
        so = StationOutput(_sta_output_path(mini_sim))
        data = so.load_data_at_station("A", "ST1")
        assert data.ndim == 2
        assert data.shape[0] == 3    # U1 U2 U3
        assert data.shape[1] == 400

    def test_load_data_unknown_station_raises(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        with pytest.raises(ValueError, match="does not exist"):
            so.load_data_at_station("XX", "BOGUS")

    # -----------------------------------------------------------------------
    # Phase 2C new tests
    # -----------------------------------------------------------------------

    def test_time_limits_exact_endpoints(self, mini_sim):
        """time_limits=[data_min, data_max] must succeed (not raise).

        The old strict-inequality guard (< and >) incorrectly rejects a
        request whose time limits exactly match the data range.
        """
        so = StationOutput(_sta_output_path(mini_sim))
        t_min = float(np.min(so.data_time))
        t_max = float(np.max(so.data_time))
        # Should not raise – all 400 samples lie within [t_min, t_max]
        data = so.load_data_at_station("A", "ST1", time_limits=[t_min, t_max])
        assert data is not None
        assert data.shape[1] == 400

    def test_empty_station_dir_raises(self, mini_sim, tmp_path):
        """Constructing StationOutput on a dir with no .nc.rank* files raises
        FileNotFoundError with a descriptive message.

        Previously the empty _nc_data dict caused a bare StopIteration from
        _get_time(), which was hard to debug.
        """
        empty_group = os.path.join(
            mini_sim, "output", "stations", "EmptyGroup"
        )
        os.makedirs(empty_group, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="No station output files"):
            StationOutput(empty_group)


# ===========================================================================
# 2. ObsPy stream and inventory
# ===========================================================================

class TestStationOutputObspy:
    """Integration tests: stream extraction and inventory building."""

    def test_stream_returns_obspy_stream(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        st = so.stream(["A", "A"], ["ST2", "ST3"])
        assert isinstance(st, obspy.Stream)

    def test_stream_trace_count(self, mini_sim):
        """Two stations × 3 channels → 6 traces."""
        so = StationOutput(_sta_output_path(mini_sim))
        st = so.stream(["A", "A"], ["ST2", "ST3"])
        assert len(st) == 6

    def test_stream_channel_names(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        st = so.stream(["A"], ["ST1"])
        channel_codes = {tr.stats.channel for tr in st}
        assert channel_codes == {"UR", "UT", "UZ"}

    def test_stream_trace_npts(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        st = so.stream(["A"], ["ST1"])
        for tr in st:
            assert tr.stats.npts == 400

    def test_stream_trace_starttime_type(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        st = so.stream(["A"], ["ST1"])
        for tr in st:
            assert isinstance(tr.stats.starttime, obspy.UTCDateTime)

    def test_stream_channel_filter(self, mini_sim):
        """Requesting only the radial component returns 1 trace per station."""
        so = StationOutput(_sta_output_path(mini_sim))
        st = so.stream(["A"], ["ST1"], channels=["UR"])
        assert len(st) == 1
        assert st[0].stats.channel == "UR"

    def test_inventory_returns_inventory(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        inv = so.inventory
        assert isinstance(inv, Inventory)

    def test_inventory_contains_all_stations(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        inv = so.inventory
        sta_codes = {sta.code for net in inv for sta in net.stations}
        assert {"ST1", "ST2", "ST3", "ST4", "ST5"}.issubset(sta_codes)

    def test_inventory_channels_per_station(self, mini_sim):
        """Each station should have 3 channels (UR, UT, UZ)."""
        so = StationOutput(_sta_output_path(mini_sim))
        inv = so.inventory
        for net in inv:
            for sta in net.stations:
                assert len(sta.channels) == 3

    def test_catalogue_type(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        cat = so.catalogue
        assert isinstance(cat, Catalog)
        assert len(cat) == 1  # single point source in fixture

    def test_parse_to_mseed_returns_stream(self, mini_sim):
        so = StationOutput(_sta_output_path(mini_sim))
        st = so.parse_to_mseed()
        assert isinstance(st, obspy.Stream)
        assert len(st) > 0

    def test_parse_to_mseed_all_stations(self, mini_sim):
        """parse_to_mseed() must include every ranked station, not drop first/last."""
        so = StationOutput(_sta_output_path(mini_sim))
        st = so.parse_to_mseed()
        # 5 stations × 3 channels (UR, UT, UZ) = 15 traces
        assert len(st) == 15, (
            f"Expected 15 traces (5 stations × 3 channels) but got {len(st)}. "
            "First/last stations may have been silently dropped."
        )

    def test_inventory_coordinates_match_station_file(self, mini_sim):
        """Inventory lat/lon values must match those in the source STA_MINI.txt.

        The fixture defines (name, lat, lon):
          ST1: 10, 20  |  ST2: 20, 40  |  ST3: 30, 60
          ST4: 40, 80  |  ST5: 50, 100
        """
        so = StationOutput(_sta_output_path(mini_sim))
        inv = so.inventory
        expected = {
            "ST1": (10.0, 20.0),
            "ST2": (20.0, 40.0),
            "ST3": (30.0, 60.0),
            "ST4": (40.0, 80.0),
            "ST5": (50.0, 100.0),
        }
        found = {
            sta.code: (sta.latitude, sta.longitude)
            for net in inv
            for sta in net.stations
        }
        for code, (exp_lat, exp_lon) in expected.items():
            assert code in found, f"Station {code} missing from inventory"
            got_lat, got_lon = found[code]
            assert got_lat == pytest.approx(exp_lat, abs=1e-6), (
                f"{code} latitude: expected {exp_lat}, got {got_lat}"
            )
            assert got_lon == pytest.approx(exp_lon, abs=1e-6), (
                f"{code} longitude: expected {exp_lon}, got {got_lon}"
            )


# ===========================================================================
# 3. obspyfy round-trip
# ===========================================================================

class TestObspyfyRoundTrip:
    """Tests the full obspyfy → ObspyfiedOutput load cycle."""

    def test_obspyfy_creates_output_files(self, mini_sim, tmp_path):
        """obspyfy() writes mseed, inventory, and catalogue files."""
        so = StationOutput(_sta_output_path(mini_sim))
        so.obspyfy()

        obspy_dir = os.path.join(_sta_output_path(mini_sim), "obspyfied")
        assert os.path.isdir(obspy_dir)

        mseed_files = [f for f in os.listdir(obspy_dir) if f.endswith(".mseed")]
        inv_files   = [f for f in os.listdir(obspy_dir) if f.endswith("inv.xml")]
        cat_files   = [f for f in os.listdir(obspy_dir) if f.endswith("cat.xml")]

        assert len(mseed_files) == 1, "Expected exactly one MiniSEED file"
        assert len(inv_files)   == 1, "Expected exactly one inventory XML"
        assert len(cat_files)   == 1, "Expected exactly one catalogue XML"

    def test_obspyfied_output_loads(self, mini_sim):
        """ObspyfiedOutput can be constructed from the obspyfied folder."""
        obspy_dir = os.path.join(_sta_output_path(mini_sim), "obspyfied")
        # obspyfy() was already called in the test above (same module scope
        # fixture), so the files should exist; if not, call it again.
        if not os.path.isdir(obspy_dir):
            so = StationOutput(_sta_output_path(mini_sim))
            so.obspyfy()

        oo = ObspyfiedOutput(obspyfied_path=obspy_dir)
        assert isinstance(oo.stream, obspy.Stream)
        assert isinstance(oo.cat, Catalog)
        assert isinstance(oo.inv, Inventory)

    def test_obspyfied_stream_traces_nonzero(self, mini_sim):
        """Loaded traces contain non-trivial data (not all zeros)."""
        obspy_dir = os.path.join(_sta_output_path(mini_sim), "obspyfied")
        if not os.path.isdir(obspy_dir):
            so = StationOutput(_sta_output_path(mini_sim))
            so.obspyfy()

        oo = ObspyfiedOutput(obspyfied_path=obspy_dir)
        norms = [np.linalg.norm(tr.data) for tr in oo.stream]
        assert any(n > 0 for n in norms), "All traces are zero – suspicious"

    # -----------------------------------------------------------------------
    # Phase 2B new tests
    # -----------------------------------------------------------------------

    def _ensure_obspyfied(self, mini_sim):
        """Helper: run obspyfy() if the obspyfied folder does not yet exist."""
        obspy_dir = os.path.join(_sta_output_path(mini_sim), "obspyfied")
        if not os.path.isdir(obspy_dir):
            StationOutput(_sta_output_path(mini_sim)).obspyfy()
        return obspy_dir

    def test_missing_inv_file_raises(self, mini_sim, tmp_path):
        """FileNotFoundError with 'inv.xml' in message when inv.xml is absent."""
        obspy_dir = self._ensure_obspyfied(mini_sim)
        test_dir = tmp_path / "obspyfied_no_inv"
        shutil.copytree(obspy_dir, str(test_dir))
        # Remove any file whose name contains 'inv.xml'
        for fname in os.listdir(str(test_dir)):
            if "inv.xml" in fname:
                os.unlink(os.path.join(str(test_dir), fname))
        with pytest.raises(FileNotFoundError, match="inv.xml"):
            ObspyfiedOutput(obspyfied_path=str(test_dir))

    def test_missing_cat_file_raises(self, mini_sim, tmp_path):
        """FileNotFoundError with 'cat.xml' in message when cat.xml is absent."""
        obspy_dir = self._ensure_obspyfied(mini_sim)
        test_dir = tmp_path / "obspyfied_no_cat"
        shutil.copytree(obspy_dir, str(test_dir))
        for fname in os.listdir(str(test_dir)):
            if "cat.xml" in fname:
                os.unlink(os.path.join(str(test_dir), fname))
        with pytest.raises(FileNotFoundError, match="cat.xml"):
            ObspyfiedOutput(obspyfied_path=str(test_dir))

    def test_missing_mseed_file_raises(self, mini_sim, tmp_path):
        """FileNotFoundError with 'mseed' in message when .mseed is absent."""
        obspy_dir = self._ensure_obspyfied(mini_sim)
        test_dir = tmp_path / "obspyfied_no_mseed"
        shutil.copytree(obspy_dir, str(test_dir))
        for fname in os.listdir(str(test_dir)):
            if ".mseed" in fname:
                os.unlink(os.path.join(str(test_dir), fname))
        with pytest.raises(FileNotFoundError, match="mseed"):
            ObspyfiedOutput(obspyfied_path=str(test_dir))

    def test_duplicate_inv_file_raises(self, mini_sim, tmp_path):
        """FileExistsError with 'inv.xml' in message when multiple inv.xml files
        exist.
        """
        obspy_dir = self._ensure_obspyfied(mini_sim)
        test_dir = tmp_path / "obspyfied_dup_inv"
        shutil.copytree(obspy_dir, str(test_dir))
        # Find the existing inv file and copy it with a different name that
        # still contains 'inv.xml' so the search_files glob matches both.
        inv_file = next(
            f for f in os.listdir(str(test_dir)) if "inv.xml" in f
        )
        shutil.copy(
            os.path.join(str(test_dir), inv_file),
            os.path.join(str(test_dir), "extra_inv.xml"),
        )
        with pytest.raises(FileExistsError, match="inv.xml"):
            ObspyfiedOutput(obspyfied_path=str(test_dir))

    # -----------------------------------------------------------------------
    # Phase 3 new tests
    # -----------------------------------------------------------------------

    def test_obspyfy_file_naming(self, mini_sim):
        """obspyfy() must write files with the exact expected names.

        Contracts:
          - MiniSEED:  <station_group_name>.mseed  → ``Station_grid.mseed``
          - Inventory: <stations_file_stem>_inv.xml → ``STA_MINI_inv.xml``
          - Catalogue: ``cat.xml``
        """
        obspy_dir = self._ensure_obspyfied(mini_sim)
        expected = {"Station_grid.mseed", "STA_MINI_inv.xml", "cat.xml"}
        actual = set(os.listdir(obspy_dir))
        assert actual == expected, (
            f"obspyfied directory contents mismatch.\n"
            f"  Expected: {sorted(expected)}\n"
            f"  Actual:   {sorted(actual)}"
        )

    def test_obspyfied_trace_count_matches_original(self, mini_sim):
        """Round-trip preserves the total trace count (5 stations × 3 channels)."""
        so = StationOutput(_sta_output_path(mini_sim))
        original_count = len(so.parse_to_mseed())
        obspy_dir = self._ensure_obspyfied(mini_sim)
        oo = ObspyfiedOutput(obspyfied_path=obspy_dir)
        assert len(oo.stream) == original_count, (
            f"Trace count changed on round-trip: "
            f"original={original_count}, reloaded={len(oo.stream)}"
        )

    def test_obspyfied_inventory_station_count(self, mini_sim):
        """The reloaded inventory must contain exactly 5 stations."""
        obspy_dir = self._ensure_obspyfied(mini_sim)
        oo = ObspyfiedOutput(obspyfied_path=obspy_dir)
        station_count = sum(len(net.stations) for net in oo.inv)
        assert station_count == 5, (
            f"Expected 5 stations in reloaded inventory, got {station_count}"
        )


# ===========================================================================
# 4. Element output – import smoke test
# ===========================================================================

class TestElementOutputImport:
    """Phase 1: verify the ElementOutput class is importable and documented."""

    def test_element_output_importable(self):
        from axikernels.core.handlers.element_output import ElementOutput  # noqa: F401
        assert ElementOutput is not None

    def test_element_output_has_obspyfy(self):
        from axikernels.core.handlers.element_output import ElementOutput
        assert callable(getattr(ElementOutput, "obspyfy", None))

    def test_element_output_has_stream_sta(self):
        from axikernels.core.handlers.element_output import ElementOutput
        assert callable(getattr(ElementOutput, "stream_STA", None))
