"""
example_obspy_workflows.py
==========================
Script-first walkthrough of the axikernels ObsPy workflows.

This script demonstrates:
  1. Station-output path   – loading netCDF waveforms → ObsPy Stream/Inventory
  2. Element-output path   – overview of ElementOutput API (no output data
     required for the walkthrough; all API calls are shown as comments when
     actual output files would be needed)

Run with the synthetic mini-simulation fixture so that the script is fully
self-contained and needs no external simulation output::

    python examples/example_obspy_workflows.py

The script creates a temporary simulation directory, fills it with minimal
synthetic data, and then runs the full pipeline all the way through
``obspyfy()``.  The optional matplotlib plots are generated when
``SHOW_PLOTS=1`` is set in the environment.

Usage
-----
    # Plain run (no plots):
    python examples/example_obspy_workflows.py

    # With waveform plots:
    SHOW_PLOTS=1 python examples/example_obspy_workflows.py
"""
import os
import shutil
import sys
import tempfile
import textwrap

import matplotlib
import numpy as np

# --------------------------------------------------------------------------
# Determine whether to show interactive plots
# --------------------------------------------------------------------------
_SHOW = os.environ.get("SHOW_PLOTS", "0").strip() not in ("0", "", "false")
if not _SHOW:
    matplotlib.use("Agg")  # non-interactive backend when plots not requested
import matplotlib.pyplot as plt  # noqa: E402 (after backend set)

# --------------------------------------------------------------------------
# Locate repo root so we can find committed fixture data
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_COMMITTED_INPUT = os.path.join(
    _REPO_ROOT,
    "axikernels", "core", "handlers", "tests",
    "NORMAL_FAULT_100KM", "input",
)
_BM_SRC = os.path.join(
    _REPO_ROOT,
    "examples", "data", "1D_KERNEL_EXAMPLE", "input", "prem_iso_elastic.bm",
)

# --------------------------------------------------------------------------
# Minimal inparam.output.yaml (references STA_MINI.txt)
# --------------------------------------------------------------------------
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

_STA_MINI = textwrap.dedent("""\
    #name network latitude longitude useless depth
    ST1 A  10  20 0 0
    ST2 A  20  40 0 0
    ST3 A  30  60 0 0
    ST4 A  40  80 0 0
    ST5 A  50 100 0 0
""")


# ==========================================================================
# SECTION 0 – Build the synthetic fixture (skip this in a real workflow)
# ==========================================================================

def _create_synthetic_fixture(sim_dir: str) -> None:
    """Populate *sim_dir* with a minimal AxiSEM3D simulation tree.

    In a real workflow, *sim_dir* would be a directory produced by running
    AxiSEM3D.  Here we create synthetic netCDF outputs so that the example
    is self-contained.
    """
    import netCDF4 as nc

    # ------------------------------------------------------------------
    # input/
    # ------------------------------------------------------------------
    inp = os.path.join(sim_dir, "input")
    os.makedirs(inp, exist_ok=True)

    for fname in (
        "inparam.model.yaml",
        "inparam.advanced.yaml",
        "inparam.nr.yaml",
        "inparam.source.yaml",
    ):
        shutil.copy(os.path.join(_COMMITTED_INPUT, fname),
                    os.path.join(inp, fname))

    with open(os.path.join(inp, "inparam.output.yaml"), "w") as fh:
        fh.write(_INPARAM_OUTPUT)

    with open(os.path.join(inp, "STA_MINI.txt"), "w") as fh:
        fh.write(_STA_MINI)

    shutil.copy(_BM_SRC, os.path.join(inp, "prem_iso_elastic.bm"))

    # ------------------------------------------------------------------
    # output/stations/Station_grid/
    # ------------------------------------------------------------------
    sta_dir = os.path.join(sim_dir, "output", "stations", "Station_grid")
    os.makedirs(sta_dir, exist_ok=True)

    station_keys = ["A.ST1", "A.ST2", "A.ST3", "A.ST4", "A.ST5"]
    n_sta   = len(station_keys)
    n_chan  = 3      # U1 U2 U3
    n_times = 400
    dt      = 0.5   # seconds

    # rank_station.info
    lines = ["MPI_RANK STATION_KEY STATION_INDEX_IN_RANK"]
    for idx, key in enumerate(station_keys):
        lines.append(f"0 {key} {idx}")
    with open(os.path.join(sta_dir, "rank_station.info"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # Synthetic netCDF: simple sine-wave per channel
    nc_path = os.path.join(sta_dir, "axisem3d_synthetics.nc.rank0")
    max_sta_len = max(len(k) for k in station_keys)
    channels    = ["U1", "U2", "U3"]
    max_ch_len  = max(len(c) for c in channels)

    ds = nc.Dataset(nc_path, "w", format="NETCDF4")
    ds.createDimension("dim_time",                 n_times)
    ds.createDimension("dim_station",              n_sta)
    ds.createDimension("dim_channel",              n_chan)
    ds.createDimension("dim_station_str_length",   max_sta_len)
    ds.createDimension("dim_channel_str_length",   max_ch_len)

    t_var = ds.createVariable("data_time", "f8", ("dim_time",))
    t_var[:] = np.arange(n_times) * dt

    w_var = ds.createVariable("data_wave", "f4",
                               ("dim_station", "dim_channel", "dim_time"))
    rng = np.random.default_rng(seed=0)
    t   = np.arange(n_times) * dt
    for i in range(n_sta):
        for j in range(n_chan):
            amp  = rng.uniform(0.5, 2.0)
            freq = 0.02 * (i + 1) + 0.005 * j
            phase = rng.uniform(0, 2 * np.pi)
            w_var[i, j, :] = (amp * np.sin(2 * np.pi * freq * t + phase)
                               * np.exp(-0.002 * t)).astype(np.float32)

    ch_var = ds.createVariable("list_channel", "S1",
                                ("dim_channel", "dim_channel_str_length"))
    for j, ch in enumerate(channels):
        ch_var[j, :] = np.array(list(ch.ljust(max_ch_len)), dtype="S1")

    st_var = ds.createVariable("list_station", "S1",
                                ("dim_station", "dim_station_str_length"))
    for i, key in enumerate(station_keys):
        st_var[i, :] = np.array(list(key.ljust(max_sta_len)), dtype="S1")

    ds.close()
    print(f"  Synthetic fixture written to: {sim_dir}")


# ==========================================================================
# SECTION 1 – Station-output workflow
# ==========================================================================

def demo_station_output(sim_dir: str) -> None:
    """Full station-output ObsPy workflow."""
    from axikernels.core.handlers.station_output import StationOutput

    print("\n" + "=" * 60)
    print("SECTION 1 – Station-output ObsPy workflow")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1a. Initialise StationOutput from the station-group directory.
    # In a real workflow this path points to AxiSEM3D's output tree:
    #   <simulation_dir>/output/stations/<station_group_name>
    # ------------------------------------------------------------------
    path_to_station_output = os.path.join(
        sim_dir, "output", "stations", "Station_grid"
    )

    print(f"\n[1a] Loading StationOutput from:\n     {path_to_station_output}")
    so = StationOutput(path_to_station_output)

    print(f"     Simulation name  : {so.simulation_name}")
    print(f"     Station group    : {so.station_group_name}")
    print(f"     Coordinate frame : {so.coordinate_frame}")
    print(f"     Channels         : {so.channels}  →  {so.detailed_channels}")
    print(f"     Time axis        : {len(so.data_time)} samples, "
          f"Δt = {so.data_time[1] - so.data_time[0]:.2f} s")

    # ------------------------------------------------------------------
    # 1b. Load data at a single station as a NumPy array
    # ------------------------------------------------------------------
    print("\n[1b] Raw waveform data at station A.ST3 ...")
    data = so.load_data_at_station("A", "ST3")
    print(f"     Shape: {data.shape}  (n_channels × n_times)")
    print(f"     Max amplitude: {np.abs(data).max():.4f}")

    # ------------------------------------------------------------------
    # 1c. Build an ObsPy Stream for two stations, all channels
    # ------------------------------------------------------------------
    print("\n[1c] Building ObsPy Stream for stations ST2 and ST4 ...")
    stream = so.stream(["A", "A"], ["ST2", "ST4"])
    print(f"     Traces : {len(stream)}")
    for tr in stream[:3]:
        print(f"       {tr.id}  |  npts={tr.stats.npts}  "
              f"|  δt={tr.stats.delta:.2f}s"
              f"|  start={tr.stats.starttime}")

    # ------------------------------------------------------------------
    # 1d. Build ObsPy Inventory from the station file
    # ------------------------------------------------------------------
    print("\n[1d] Building ObsPy Inventory ...")
    inv = so.inventory
    n_sta = sum(len(net.stations) for net in inv)
    n_cha = sum(len(sta.channels) for net in inv for sta in net.stations)
    print(f"     Networks : {len(inv.networks)}")
    print(f"     Stations : {n_sta}")
    print(f"     Channels : {n_cha}  (= {n_sta} stations × "
          f"{len(so.detailed_channels)} channels each)")

    # ------------------------------------------------------------------
    # 1e. Access / create the event catalogue
    # ------------------------------------------------------------------
    print("\n[1e] Event catalogue ...")
    cat = so.catalogue
    print(f"     Events: {len(cat)}")
    for ev in cat:
        orig = ev.preferred_origin() or ev.origins[0]
        print(f"       lat={orig.latitude:.1f}°  "
              f"lon={orig.longitude:.1f}°  "
              f"depth={orig.depth / 1e3:.0f} km")

    # ------------------------------------------------------------------
    # 1f. full ObsPy stream via parse_to_mseed (uses all rank-listed stations)
    # ------------------------------------------------------------------
    print("\n[1f] Parse all ranked stations to MiniSEED stream ...")
    mseed_stream = so.parse_to_mseed()
    print(f"     Traces: {len(mseed_stream)}")

    # ------------------------------------------------------------------
    # 1g. obspyfy() – write MiniSEED + StationXML + QuakeML to disk
    # ------------------------------------------------------------------
    print("\n[1g] Running obspyfy() ...")
    so.obspyfy()
    obspy_dir = os.path.join(path_to_station_output, "obspyfied")
    written = sorted(os.listdir(obspy_dir))
    print(f"     Files written to {obspy_dir}:")
    for f in written:
        size = os.path.getsize(os.path.join(obspy_dir, f))
        print(f"       {f}  ({size} bytes)")

    # ------------------------------------------------------------------
    # 1h. Load the obspyfied output via ObspyfiedOutput
    # ------------------------------------------------------------------
    print("\n[1h] Loading obspyfied output via ObspyfiedOutput ...")
    from axikernels.core.handlers.obspy_output import ObspyfiedOutput
    oo = ObspyfiedOutput(obspyfied_path=obspy_dir)
    print(f"     Stream  : {oo.stream}")
    print(f"     Catalog : {oo.cat}")
    inv_loaded = oo.inv
    print(f"     Inventory: {sum(len(n.stations) for n in inv_loaded.networks)} stations")

    # ------------------------------------------------------------------
    # Optional: plot the first 3 traces
    # ------------------------------------------------------------------
    if _SHOW or os.environ.get("SAVE_PLOTS", "0") != "0":
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        fig.suptitle("Synthetic waveforms – Station A.ST2, RTZ channels")
        t = so.data_time
        for ax, tr, lbl in zip(axes, stream[:3],
                               so.detailed_channels):
            ax.plot(t[:len(tr.data)], tr.data, lw=0.8)
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        if _SHOW:
            plt.show()
        else:
            out_png = os.path.join(_HERE, "obspy_workflow_station.png")
            fig.savefig(out_png, dpi=150)
            print(f"     Plot saved: {out_png}")
        plt.close(fig)


# ==========================================================================
# SECTION 2 – Element-output API overview
# ==========================================================================

def demo_element_output_overview() -> None:
    """Print an API overview of ElementOutput.

    ElementOutput requires actual simulation output files (element netCDF
    data from AxiSEM3D) that are too large to commit.  This section describes
    the workflow and shows example code; all executable lines are clearly
    marked as requiring real output data.
    """
    print("\n" + "=" * 60)
    print("SECTION 2 – Element-output API overview")
    print("=" * 60)

    from axikernels.core.handlers.element_output import ElementOutput  # noqa

    print("""
ElementOutput inherits from AxiSEM3DOutput and works with the element-wise
output produced by AxiSEM3D (output/elements/<group_name>/).

Typical workflow (requires real AxiSEM3D element-output files)
--------------------------------------------------------------

    # 1. Load the element output
    from axikernels.core.handlers.element_output import ElementOutput

    path = "<simulation_dir>/output/elements/<group_name>"
    eo = ElementOutput(path)

    print("Source:", eo.source_lat, eo.source_lon, eo.source_depth)
    print("Groups:", eo.element_groups)

    # 2. Build an ObsPy Stream at stations listed in a station file
    #    (the stations are interpolated onto the element mesh)
    stream = eo.stream_STA(
        path_to_station_file="<simulation_dir>/input/GSN_stations.txt",
        channels=["UR", "UT", "UZ"],
    )
    print(stream)

    # 3. Build an Inventory for those stations
    inv = eo.create_inventory(
        path_to_station_file="<simulation_dir>/input/GSN_stations.txt"
    )

    # 4. Full obspyfy: writes MiniSEED + StationXML + QuakeML
    eo.obspyfy(
        path_to_station_file="<simulation_dir>/input/GSN_stations.txt",
        channels=["UR", "UT", "UZ"],
    )

Available public attributes and methods
----------------------------------------
  eo.path_to_elements_output   – path to the elements group folder
  eo.element_groups            – list of element group names
  eo.element_groups_info       – metadata dict for each element group
  eo.source_lat / .source_lon / .source_depth  – event location
  eo.stream_STA(sta_file, channels)   – ObsPy Stream from station list
  eo.create_inventory(sta_file)       – ObsPy Inventory from station list
  eo.obspyfy(sta_file, channels)      – write MiniSEED + XML to disk
""")

    print("ElementOutput class confirmed importable:", ElementOutput.__name__)
    print("  obspyfy      :", callable(getattr(ElementOutput, "obspyfy", None)))
    print("  stream_STA   :", callable(getattr(ElementOutput, "stream_STA", None)))
    print("  create_inventory:",
          callable(getattr(ElementOutput, "create_inventory", None)))


# ==========================================================================
# Entry point
# ==========================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("axikernels – ObsPy workflow example")
    print("=" * 60)

    # Create a temporary directory to hold the synthetic simulation
    tmp = tempfile.mkdtemp(prefix="axikernels_obspy_example_")
    sim_dir = os.path.join(tmp, "MINI_SIM")
    os.makedirs(sim_dir)

    print(f"\n[setup] Creating synthetic fixture in: {sim_dir}")
    _create_synthetic_fixture(sim_dir)

    try:
        demo_station_output(sim_dir)
        demo_element_output_overview()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n[cleanup] Removed temporary directory: {tmp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
