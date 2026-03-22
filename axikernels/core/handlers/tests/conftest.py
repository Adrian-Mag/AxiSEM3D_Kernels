"""
conftest.py – pytest fixtures for handler ObsPy workflow tests.

Generates a minimal synthetic AxiSEM3D simulation output tree underneath a
temporary directory.  The fixture creates:

    <tmp>/MINI_SIM/
        input/
            inparam.model.yaml
            inparam.output.yaml
            inparam.source.yaml
            inparam.advanced.yaml
            inparam.nr.yaml
            STA_MINI.txt          (5 surface stations)
            prem_iso_elastic.bm   (symlink to examples/data copy)
        output/
            stations/
                Station_grid/
                    rank_station.info
                    axisem3d_synthetics.nc.rank0

The synthetic waveform data are purely random floats – they are sufficient to
exercise the full ObsPy pipeline (stream construction, inventory building,
obspyfy), but carry no physical meaning.
"""
from __future__ import annotations

import os
import shutil
import textwrap

import netCDF4 as nc
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths to the committed fixture data that we re-use for input files
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_COMMITS_INPUT = os.path.join(_HERE, "NORMAL_FAULT_100KM", "input")

# We borrow the .bm model file from the examples tree (closest committed copy).
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_BM_SRC = os.path.join(
    _PKG_ROOT,
    "examples", "data", "1D_KERNEL_EXAMPLE", "input", "prem_iso_elastic.bm",
)

# ---------------------------------------------------------------------------
# Tiny station list content (5 surface stations, network A)
# ---------------------------------------------------------------------------
_STA_MINI = textwrap.dedent("""\
    #name network latitude longitude useless depth
    ST1 A 10 20 0 0
    ST2 A 20 40 0 0
    ST3 A 30 60 0 0
    ST4 A 40 80 0 0
    ST5 A 50 100 0 0
""")

# ---------------------------------------------------------------------------
# inparam.output.yaml – hard-coded minimal version referencing STA_MINI.txt
# ---------------------------------------------------------------------------
_INPARAM_OUTPUT = textwrap.dedent("""\
    list_of_station_groups:
        - Station_grid:
            locations:
                station_file: STA_MINI.txt
                horizontal_x1_x2: LATITUDE_LONGITUDE
                vertical_x3: DEPTH
                ellipticity: true
                depth_below_solid_surface: true
                undulated_geometry: true
            wavefields:
                coordinate_frame: RTZ
                medium: SOLID
                channels: [U]
            temporal:
                sampling_period: DT
                time_window: FULL
            file_options:
                format: NETCDF
                buffer_size: 50000
                flush: true
    list_of_element_groups: []
""")


# ---------------------------------------------------------------------------
# Helper: create the synthetic netCDF file
# ---------------------------------------------------------------------------

def _write_nc_rank(path: str, station_keys: list[str],
                   n_times: int = 200, dt: float = 0.5) -> None:
    """Write a minimal axisem3d_synthetics.nc file for *station_keys*."""
    n_sta = len(station_keys)
    channels = ["U1", "U2", "U3"]
    n_chan = len(channels)

    rng = np.random.default_rng(seed=42)

    ds = nc.Dataset(path, "w", format="NETCDF4")

    # Dimensions
    ds.createDimension("dim_time", n_times)
    ds.createDimension("dim_station", n_sta)
    ds.createDimension("dim_channel", n_chan)
    # string length dims – long enough for keys like "A.ST1"
    max_sta_len = max(len(k) for k in station_keys)
    max_ch_len  = max(len(c) for c in channels)
    ds.createDimension("dim_station_str_length", max_sta_len)
    ds.createDimension("dim_channel_str_length", max_ch_len)

    # data_time
    t_var = ds.createVariable("data_time", "f8", ("dim_time",))
    t_var[:] = np.arange(n_times, dtype=np.float64) * dt

    # data_wave  (station × channel × time)
    w_var = ds.createVariable("data_wave", "f4",
                               ("dim_station", "dim_channel", "dim_time"))
    # simple synthetic signal: sine wave with station/channel offsets
    for i in range(n_sta):
        for j in range(n_chan):
            freq = 0.05 * (i + 1) + 0.01 * j
            w_var[i, j, :] = np.sin(
                2 * np.pi * freq * np.arange(n_times) * dt
            ).astype(np.float32)

    # list_channel
    ch_var = ds.createVariable(
        "list_channel", "S1",
        ("dim_channel", "dim_channel_str_length"),
    )
    for j, ch in enumerate(channels):
        padded = ch.ljust(max_ch_len)
        ch_var[j, :] = np.array(list(padded), dtype="S1")

    # list_station
    st_var = ds.createVariable(
        "list_station", "S1",
        ("dim_station", "dim_station_str_length"),
    )
    for i, key in enumerate(station_keys):
        padded = key.ljust(max_sta_len)
        st_var[i, :] = np.array(list(padded), dtype="S1")

    ds.close()


# ---------------------------------------------------------------------------
# Pytest fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini_sim(tmp_path_factory):
    """Return the path to a minimal synthetic AxiSEM3D simulation directory.

    Layout::

        <tmp>/MINI_SIM/
            input/  ...
            output/stations/Station_grid/  ...
    """
    base = tmp_path_factory.mktemp("MINI_SIM", numbered=False)
    sim_dir = base / "MINI_SIM"
    sim_dir.mkdir()

    # --- input/ -------------------------------------------------------
    inp = sim_dir / "input"
    inp.mkdir()

    # Copy immutable input files from the committed fixture
    for fname in (
        "inparam.model.yaml",
        "inparam.advanced.yaml",
        "inparam.nr.yaml",
        "inparam.source.yaml",
    ):
        shutil.copy(os.path.join(_COMMITS_INPUT, fname), inp / fname)

    # Our custom inparam.output.yaml (references STA_MINI.txt)
    (inp / "inparam.output.yaml").write_text(_INPARAM_OUTPUT)

    # Mini station file
    (inp / "STA_MINI.txt").write_text(_STA_MINI)

    # .bm model file
    shutil.copy(_BM_SRC, inp / "prem_iso_elastic.bm")

    # --- output/ ------------------------------------------------------
    sta_dir = sim_dir / "output" / "stations" / "Station_grid"
    sta_dir.mkdir(parents=True)

    # Station keys: network.name for all 5 stations
    station_keys = ["A.ST1", "A.ST2", "A.ST3", "A.ST4", "A.ST5"]

    # rank_station.info
    rank_info_lines = ["MPI_RANK STATION_KEY STATION_INDEX_IN_RANK"]
    for idx, key in enumerate(station_keys):
        rank_info_lines.append(f"0 {key} {idx}")
    (sta_dir / "rank_station.info").write_text("\n".join(rank_info_lines) + "\n")

    # Synthetic netCDF file
    nc_path = sta_dir / "axisem3d_synthetics.nc.rank0"
    _write_nc_rank(str(nc_path), station_keys, n_times=400, dt=0.5)

    return str(sim_dir)
