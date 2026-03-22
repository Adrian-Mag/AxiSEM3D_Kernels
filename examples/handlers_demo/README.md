# Handlers Demo

A self-contained mini-simulation that demonstrates the two main output handlers provided by `axikernels`:

- **`StationOutput`** — reads per-rank NetCDF waveforms, exposes them as ObsPy streams, and can *obspyfy* the output (miniSEED + StationXML + QuakeML).
- **`ElementOutput`** — reads the full wavefield at GLL points and interpolates at any `[radius, lat, lon]` coordinate.

## Prerequisites

- An MPI implementation (`mpirun` / `mpiexec`) available in your environment.
- The `axikernels` package installed (`pip install -e .` from the repo root).

## Quick start

```bash
# 1. Run the tiny AxiSEM3D simulation (takes ~1 min on 2 cores)
./run_demo.sh

# 2. Open the notebook
jupyter notebook handlers_demo.ipynb
```

The simulation writes output to `sim_run/output/` (gitignored).
Use `./clean_demo.sh` to remove the run directory when done.

## Files

| File | Purpose |
|---|---|
| `run_demo.sh` | Runs the simulation and populates `sim_run/output/` |
| `clean_demo.sh` | Removes `sim_run/` (with confirmation) |
| `handlers_demo.ipynb` | Notebook walking through `StationOutput`, `ObspyfiedOutput`, and `ElementOutput` |
| `axisem3d` | Pre-compiled AxiSEM3D binary (Linux x86-64) |
| `input/` | AxiSEM3D input parameter files and mesh |
