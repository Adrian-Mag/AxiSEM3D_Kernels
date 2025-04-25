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
