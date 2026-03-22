# axisem3d_output

Package for handling the stations and elements outputs of AxiSEM3D. Comes with visualzation tools and classes for creating sensitivity kernels based on AxiSEM3D outputs.


## Requirements
- python (tested on 3.12)
- pyyaml (to load and edit inparam files)
- pandas (to handle h5 files for kernels)
- obspy
- mpmath (for some high accuracy calculations)
- mayavi (for 3D visualization on the GPU)
- xarray (for lazy loading from large netcdf files)
- tqdm (for some loading bars)
- plotly (needed some functionalities for plotting kernels on slice meshes)
- netCDF4 (to handle netcdf files)
- basemap (for plotting)
- ruamel.yaml (needed to modify the inparam files from python)
- tables (for saving h5 metadata)


## ObsPy Workflow Examples

This branch adds a script-first ObsPy walkthrough and a companion notebook under `examples/`:

- `examples/example_obspy_workflows.py` runs a fully self-contained station-output workflow using a tiny synthetic fixture built from committed inputs.
- `examples/example_obspy_workflows.ipynb` mirrors the same station-output path and includes an ElementOutput API template section for real element-output runs.

Recommended environment for the full workflow:

- `conda activate axikernels_env`
- `python examples/example_obspy_workflows.py`

Notes:

- The script was re-verified in Phase 4 and completes end to end.
- The notebook now adds the repository root to `sys.path` automatically so it can import the local package even when the kernel starts outside the repo root.
- The notebook keeps the station-output workflow fully runnable in a lighter kernel and defers live ElementOutput import to the later API-overview section. For live element-output execution, use the full `axikernels_env` dependency stack.
