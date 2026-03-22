# axikernels Handlers — Living Reference

_Last updated: interpolation-repair Phase 8_

---

## Package root
`axikernels/core/handlers/`

---

## Classes

### `AxiSEM3DOutput` (`axisem3d_output.py`)
Base class for all simulation output handlers.

**Constructor:**
```python
AxiSEM3DOutput(path_to_simulation: str, path_to_base_model: str = None)
```
- Walks up from `path_to_simulation` to find the simulation root directory
- Sets `self.simulation_name = os.path.basename(path_to_simulation)`
- Auto-finds the `.bm` file in `input/` if `path_to_base_model` is None (raises `ValueError` if none found)
- Reads `inparam.output.yaml` for output metadata
- Sets `self.Domain_Radius` from the `.bm` model file

**Key properties/methods:**

| Name | Returns | Notes |
|------|---------|-------|
| `catalogue` | `obspy.Catalog` | Reads `inparam.source.yaml`. On first call writes `<sim>/input/<name>_cat.xml` and caches the Catalog. Subsequent calls re-use the cached value (even across new instances if the XML file exists). |
| `_find_catalogue()` | `Catalog \| None` | Returns `Catalog` if exactly one `*cat*.xml` in `input/`, else `None`. Previously returned `(None, 1)`—**FIXED Phase 1** (the old `[0]` subscript caused the second instance to cache the first Event, not the Catalog). |
| `inventory` | `obspy.Inventory` | Reads station file; builds Inventory. Bug fixed Phase 1: `header=None` instead of `header=1` (previously skipped first 2 stations). |
| `_find_outputs()` | `dict` | Returns `{'elements': {...}, 'stations': {...}}` |

---

### `StationOutput` (`station_output.py`)
Subclass of `AxiSEM3DOutput`. Handles point-station netCDF output.

**Constructor:**
```python
StationOutput(path_to_station_group: str)
```
- `path_to_station_group` must be `<sim>/output/stations/<group_name>/`
- Parses `rank_station.info` to map station keys to MPI ranks

**Key properties/methods:**

| Name | Returns | Notes |
|------|---------|-------|
| `station_group_name` | `str` | Name of the station group (`output/stations/<this>`) |
| `coordinate_frame` | `str` | `RTZ`, `ENZ`, etc. from `inparam.output.yaml` |
| `channels` | `list[str]` | e.g. `['U']` |
| `detailed_channels` | `list[str]` | Expanded: `['UR', 'UT', 'UZ']` |
| `data_time` | `np.ndarray` | 1-D time axis from netCDF `data_time` |
| `load_data_at_station(network, name, channels, time_limits)` | `np.ndarray (n_ch, n_t)` | Raw float32 waveform data. Phase 2: `time_limits` now uses `<=`/`>=` (was `<`/`>`), and `np.where(...)[0]` unpacking fixed. |
| `stream(networks, stations, ...)` | `obspy.Stream` | Wraps waveforms in Trace objects. Phase 2: bare `except` retry removed. |
| `inventory` | `obspy.Inventory` | From station file; `header=None` fix Phase 1 |
| `catalogue` | `obspy.Catalog` | Inherited from base class |
| `obspyfy()` | None | Writes `obspyfied/<group>.mseed`, `<sta>_inv.xml`, `cat.xml` |
| `_rank_list` | `pd.DataFrame` | Parsed `rank_station.info` |

**Constructor notes (Phase 2):** Raises `FileNotFoundError` immediately after `_load_files()` returns empty list (no `.nc.rank*` files in the directory) instead of letting `_get_time()` crash with `StopIteration`.

**netCDF output format:**
```
data_time   (dim_time,)               float64
data_wave   (dim_station, dim_channel, dim_time)  float32 / masked
list_channel (dim_channel, dim_channel_str_length)  S1
list_station (dim_station, dim_station_str_length)  S1
```

---

### `ElementOutput` (`element_output.py`)
Subclass of `AxiSEM3DOutput`. Handles 2-D mesh-slice element output.

**Constructor:**
```python
ElementOutput(path_to_element_group: str)
```
- `path_to_element_group` must be `<sim>/output/elements/<group_name>/`
- **h5netcdf fallback (post Phase 8):** `_read_element_metadata` now catches `OSError` from `xr.open_dataset()` and retries with `engine='h5netcdf'`. This handles HDF5 library version mismatches between the `netcdf4` C library and the file format.

**Key methods:**

| Name | Notes |
|------|-------|
| `stream_STA(path_to_station_file, channels, ...)` | Interpolates wavefield at station locations. Phase 2: `header=None` fix, `trace.stats.npts` typo fixed. **Phase 7:** `self.data_time` bug fixed (uses metadata path); `np.where` → `np.flatnonzero`; npts/starttime computed from selected time subset. |
| `create_inventory(path_to_station_file)` | Builds `obspy.Inventory`. Phase 2: `header=None` fix. **Phase 4:** theta computed via `arccos(z/r)` (not `arctan2+π/2`); `-1` domain guard added (skips station with warning if not in any group). |
| `obspyfy(path_to_station_file, channels)` | Writes MiniSEED + XML |
| `load_data(points, channels, time_slices, ...)` | Phase 3: sentinel changed from `np.ones` to `np.full(..., np.nan)`. |
| `load_data_from_element_group(points, group, channel_slices, time_slices, pbar)` | Phase 3: complete rewrite of interpolation pipeline. KDTree + Newton element search; isoparametric weights via `interpolation_weights_9node`; NaN sentinel; `_lagrange` method removed; polar-coord / problematic-elements code removed. |
| `stream(points, channels, ...)` | **Phase 7:** `np.where` → `np.flatnonzero`; npts/starttime computed from selected time subset. |
| `animation(...)` | **Phase 7:** Replaced `sys.exit()` with `raise ValueError(...)`. |
| `_project_on_inplane(points, degrees, frame, coords)` | **Phase 4:** theta now computed as `arccos(z/r)` (colatitude). |
| `_group_by_material(points)` | **Phase 4:** `-1` guard added — unmatched points (domain index `-1`) map to material `-1` instead of silently wrapping to the last group. |

---

### `ObspyfiedOutput` (`obspy_output.py`)
Reloads a previously-obspyfied directory.

**Constructor:**
```python
ObspyfiedOutput(obspyfied_path: str)
```

**Key attributes:** `.stream` (`obspy.Stream`), `.inv` (`obspy.Inventory`), `.cat` (`obspy.Catalog`)

**Error handling (Phase 2):**
- `_find_inv_files`: raises `FileNotFoundError('No inv.xml files were found')` / `FileExistsError('Multiple inv.xml files were found')`
- `_find_cat_files`: raises `FileNotFoundError('No cat.xml files were found')` / `FileExistsError('Multiple cat.xml files were found')`
- `_find_mseed_files`: messages unchanged (already correct for mseed)
- All unreachable `sys.exit(1)` calls after `raise` removed

---

## Test infrastructure

### `axikernels/core/handlers/tests/`

| File | Purpose |
|------|---------|
| `NORMAL_FAULT_100KM/input/` | Committed input fixture (no output data) |
| `conftest.py` | **NEW Phase 1** — Module-scoped `mini_sim` fixture: generates synthetic netCDF from committed input files in a tempdir |
| `test_obspy_workflows.py` | Phase 1: 27 tests. Phase 2: +8 tests → 35. **Phase 3: +4 tests** → `TestStationOutputCore` (13), `TestStationOutputObspy` (13), `TestObspyfyRoundTrip` (10), `TestElementOutputImport` (3). Total: **39 tests**. |
| `test_element_output.py` | **NEW Phase 2** — 5 tests (Phase 2). **Phase 3: +5 tests** → `TestPhase3SourceGuards` class. **Phase 4: +5 tests** → `TestPhase4ThetaConvention` class. **Phase 7: +9 tests** → `TestPhase7SourceGuards` (5) + `TestPhase7FourierReconstruction` (4). Total: **24 tests**. Phase 7 source guards: `test_stream_sta_no_self_data_time`, `test_no_sys_exit`, `test_no_point_not_in_output_domain`, `test_no_bare_np_where_in_stream`, `test_no_bare_np_where_in_stream_sta`. Phase 7 Fourier tests: `test_fourier_reconstruction_order0`, `test_fourier_reconstruction_order1`, `test_fourier_reconstruction_nyquist`, `test_fourier_reconstruction_known_signal`. |
| `test_integration.py` | **NEW Phase 8** — 8 end-to-end integration tests against real handlers_demo PREM simulation data. Module-scoped `demo_eo` fixture (ElementOutput loaded once). Tests: `test_load_data_mantle_nontrivial`, `test_load_data_outer_core_nontrivial`, `test_load_data_multi_channel`, `test_load_data_batch_points`, `test_stream_produces_valid_obspy`, `test_create_inventory_produces_valid_inventory`, `test_load_data_outside_domain_returns_nan` (uses r=7000km above surface), `test_metadata_consistency`. All skipped if `examples/handlers_demo/sim_run/output/elements/` is absent. |
| `test_example_script.py` | **NEW Phase 1** — 1 test: `test_example_obspy_workflows_end_to_end` — runs `examples/example_obspy_workflows.py` as a subprocess and asserts key stdout markers |
| `test_axisem3d_output.py` | Pre-existing tests refreshed in Phase 1 — uses an absolute fixture path and transiently bootstraps the `.bm` model file so `AxiSEM3DOutput` can initialize against the committed fixture |

**Fixture constants (conftest.py):**
- Station group: `Station_grid`, frame: `RTZ`, channels: `[U]` → `['UR','UT','UZ']`
- 5 stations: `A.ST1`…`A.ST5`, surface, uniform spacing
- n_times: **400**, dt: 0.5 s  _(note: `_write_nc_rank` default arg is 200, but `mini_sim` passes `n_times=400`)_
- Copies `prem_iso_elastic.bm` from `examples/data/1D_KERNEL_EXAMPLE/input/`

---

## Examples

| File | Description |
|------|-------------|
| `examples/handlers_demo/run_demo.sh` | **NEW handlers-demo Phase 1** — self-contained local runner inside `examples/handlers_demo/`; uses the copied binary and input bundle in the same folder, validates the bundle with `--dry-run`, and writes station and element outputs under `examples/handlers_demo/sim_run/` |
| `examples/handlers_demo/handlers_demo.ipynb` | **NEW handlers-demo Phase 3** — minimal demo notebook; loads `StationOutput` from `sim_run/output/stations/GSN_Station_Grid` and `ElementOutput` from `sim_run/output/elements`; demonstrates `.outputs`, `.stream()`, and `.load_data()` with inline plots |
| `examples/run_handlers_demo.sh` | **NEW realistic-demo Phase 1** — lightweight Linux/MPI runner that copies `examples/data/HANDLERS_EXAMPLE/` into `examples/demo_runs/HANDLERS_EXAMPLE_RUN/`, runs AxiSEM3D, and prints the generated station and element output paths for realistic handler post-processing |
| `examples/example_obspy_workflows.py` | **NEW Phase 1** — Runnable script, full station-output workflow + element API overview; validated by `test_example_script.py` |
| `examples/example_obspy_workflows.ipynb` | **Rewritten realistic-demo Phase 2** — guided post-processing notebook for a user who has already run the small HANDLERS_EXAMPLE simulation; defaults to `examples/demo_runs/HANDLERS_EXAMPLE_RUN/`, walks through real station and element outputs, and reuses the existing run station file for `ElementOutput` so the workflow stays read-only |

---

## Bug fixes (Phase 1)

1. **`axisem3d_output.py` `_find_catalogue()`**: Was returning `(None, 1)` / `(None, 2)` on empty/multiple. The `[0]` subscript in `__init__` caused the second instance to store `Event` instead of `Catalog`. Fixed: returns `None` directly; init uses `self._find_catalogue()` without subscript.

2. **`station_output.py` `inventory` property**: `pd.read_csv(..., header=1)` silently skipped the first 2 data rows. Fixed: `header=None`.

3. **`test_axisem3d_output.py`**: Wrong import path `from axisem3d_output.core.handlers...` → `from axikernels.core.handlers...`.

---

## Bug fixes (Phase 2)

1. **`element_output.py` `create_inventory` + `stream_STA`**: `pd.read_csv(..., header=0, comment='#')` consumed the first data row as a discarded header, silently dropping one station. Fixed: `header=None` in both methods.

2. **`element_output.py` `stream_STA` + `stream`**: `trace.stats.ntps = npts` typo — `ntps` is not a valid ObsPy attribute. Fixed: `trace.stats.npts = npts`.

3. **`obspy_output.py` `_find_inv_files` + `_find_cat_files`**: Error messages said "mseed files" regardless of context. Fixed: inv methods say "inv.xml", cat methods say "cat.xml". Also removed 6 unreachable `sys.exit(1)` calls after `raise` statements; removed now-unused `import sys`.

4. **`station_output.py` `load_data_at_station`**: `time_limits` guard used strict `<`/`>`, rejecting requests where limits exactly matched the data range. Fixed: `<=`/`>=`. Also fixed `np.where(...)[0]` unpacking so the returned 1-D array index is consistent with the `time_limits=None` path which uses `np.arange()`.

5. **`station_output.py` `stream`**: Bare `except Exception` retry block called `load_data_at_station` twice with identical arguments, guaranteed to fail the same way. Fixed: call directly without try/except.

6. **`station_output.py` `__init__`**: Empty `_load_files()` result caused `StopIteration` deep in `_get_time()`. Fixed: raises `FileNotFoundError("No station output files found in ...")` immediately after file discovery.

---

## Bug fixes / rewrites (Phase 3)

**`element_output.py` `load_data_from_element_group` — isoparametric interpolation rewrite:**

1. **Broken polar-coordinate Lagrange interpolation removed.** The old code converted physical (s,z) coordinates to polar (r,θ), then attempted to invert a tensor-product Lagrange interpolation over (r,θ). This failed for non-separable ("problematic") elements and produced wrong results for curved elements near the axis. Removed: `cart2polar` calls, `_lagrange` method calls, `problematic_elements` tracking, `map_of_problematique`, `expanded_map_of_problematique`, `points_of_interest` expansion, `GLL_rads/GLL_thetas`, and the `[0,0,0,0,1,0,0,0,0]` fallback hack.

2. **New element search via KDTree + Newton containment.** `build_element_kdtree` + `find_containing_elements_batch` from `isoparametric.py` replace the old nearest-center `argmin`. Returns reference coordinates (ξ,η) alongside element indices. Unmatched points (element_index==-1) are silently skipped — they remain NaN in the output.

3. **New isoparametric interpolation weights.** `interpolation_weights_9node(xi, eta, xi_nodes, eta_nodes)` replaces the outer Lagrange product. `detect_axial` + `reference_abscissae` correctly select GLL×GLL vs GLJ×GLL abscissae per element.

---

## Bug fixes (Phase 4)

1. **`element_output.py` `_project_on_inplane` — theta convention corrected.**
   `cart2polar_mpmath(s, z)` returns `arctan2(z, s)` (angle from s-axis). Adding `π/2` gives `arctan2(z,s) + π/2 = π/2 - arctan2(s,z)` which equals `π − colatitude`, not colatitude. Replaced with direct computation: `theta = arccos(clip(z/r, -1, 1))`, which gives correct colatitude (0 at north pole, π at south pole). The `cart2polar_mpmath` call and the `+= np.pi/2` line are both removed.

2. **`element_output.py` `create_inventory` — same theta fix + `-1` domain guard.**
   The same `cart2polar + += np.pi/2` pattern appeared in `create_inventory`; replaced with the same `arccos(z/r)` computation. Added guard: if `_separate_by_inplane_domain` returns `-1` for a station, it is skipped with a `logging.warning` instead of silently wrapping to `self.element_groups[-1]` (the last group).

3. **`element_output.py` `_group_by_material` — `-1` index-wrapping guard.**
   `group_mapping_to_material[group]` with `group == -1` silently returned the last material (Python list negative indexing). Added explicit guard: unmatched points (`group < 0`) map to material value `-1` instead.


4. **NaN sentinels** replace `np.ones(...)` in both `load_data` and `load_data_from_element_group`. Uninterpolated points yield `np.nan` instead of `1.0`, making missing data detectable.

5. **`_lagrange` instance method deleted** (was lines 1211-1228). Replaced by `interpolation_weights_9node` from `isoparametric.py`.

---

## Bug fixes (Phase 7)

1. **`element_output.py` `stream_STA`**: Used nonexistent `self.data_time`; replaced with proper metadata lookup `self.element_groups_info[first_key]['metadata']['data_time']`. Moved before time-slice computation. `np.where` → `np.flatnonzero`. `npts`/`starttime` now computed from the selected time subset (`selected_time`), not the full axis.

2. **`element_output.py` `stream`**: Same `np.where` → `np.flatnonzero` fix. `npts`/`starttime` computed from `selected_time`.

3. **`element_output.py` unused imports removed**: `concurrent.futures`, `time`, `warnings`, `matplotlib` (bare import — `matplotlib.pyplot` and `matplotlib.animation` remain). `cart2polar_mpmath` removed from coordinate_transforms import (not used in this file).

4. **`element_output.py` `animation`**: Replaced `logging.error(...); sys.exit()` with `raise ValueError('Not all element groups have the same time axis.')`.

5. **`element_output.py` `_point_not_in_output_domain`**: Dead method removed. It had no callers and referenced undefined attributes `self.vertical_range` / `self.horizontal_range`.

6. **`aux/coordinate_transforms.py` `cart2cyl` + `cart2cyl_mpmath`**: Docstrings said return order `[s, phi, z]` but actual code returns `[s, z, phi]`. Docstrings corrected to `[s, z, phi]`.

**Preserved unchanged:**
- Unique-points deduplication logic
- File/nag grouping structure and `main_dict`
- Data loading from netCDF
- Data expansion loop (simplified: `map_of_problematique` removed)
- Point-level expansion (`inplane_point_repetitions`)
- Fourier reconstruction and result writing

---

## Module: `isoparametric.py` (NEW — interpolation-repair Phase 1; extended Phase 2)

**Purpose:** Pure-function reference-space isoparametric interpolation for AxiSEM3D
element output.  No classes, no side effects, float64 throughout.

**Imports added Phase 2:** `from scipy.spatial import KDTree`

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `GLL_SUBSET` | `[-1.0, 0.0, 1.0]` | 3-point GLL subset (indices [0,2,4] of nPol=4 set) |
| `GLJ_SUBSET` | `[-1.0, 0.132300820777, 1.0]` | 3-point GLJ subset for axial elements |

### Functions

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `lagrange_1d` | `(t, nodes, i)` | `float` | i-th Lagrange basis at t |
| `lagrange_weights_1d` | `(t, nodes)` | `ndarray (n,)` | All n basis weights; sums to 1 |
| `compute_min_edge_length` | `(element_coords_9)` | `float` | Min of 4 corner edge lengths |
| `detect_axial` | `(element_coords_9, tol=1e-3)` | `bool` | True if LEFT edge (indices 0,1,2) has \|s\| < tol×min_edge |
| `reference_abscissae` | `(axial)` | `(xi_nodes, eta_nodes)` | Returns (GLJ,GLL) if axial else (GLL,GLL) |
| `forward_map_9node` | `(xi, eta, coords9, xi_nodes, eta_nodes)` | `(s, z)` | F(ξ,η) = Σᵢ Σⱼ Lᵢ(ξ)Lⱼ(η) X_ij |
| `jacobian_9node` | `(xi, eta, coords9, xi_nodes, eta_nodes)` | `ndarray (2,2)` | ∂(s,z)/∂(ξ,η) |
| `newton_inverse` | `(s, z, coords9, xi_nodes, eta_nodes, max_iter=10, tolerance=1e-9)` | `(xi, eta, converged, inside)` | Newton solve F(ξ,η)=(s,z) |
| `interpolation_weights_9node` | `(xi, eta, xi_nodes, eta_nodes)` | `ndarray (9,)` | Outer product Lᵢ(ξ)⊗Lⱼ(η), ipnt=ipol×3+jpol order |
| `build_element_kdtree` | `(all_element_coords)` | `(KDTree, centers ndarray (n,2))` | Builds scipy KDTree on element centres (mean of 9 nodes) |
| `find_containing_element` | `(s, z, all_element_coords, kdtree, k=20)` | `(element_index, xi, eta)` | k-NN search + Newton containment test; returns (-1, nan, nan) if not found |
| `find_containing_elements_batch` | `(points_sz, all_element_coords, kdtree, k=20)` | `(indices ndarray, xi_arr, eta_arr)` | Loops over rows of points_sz calling find_containing_element |

### Storage order

`ipnt = ipol × 3 + jpol` where ipol→ξ (outer), jpol→η (inner).
Corner indices: `[0, 2, 6, 8]`.  LEFT edge (ξ=-1): `[0, 1, 2]`.

### Newton algorithm (matching `Mapping.hpp`)

- Init: (ξ,η) = (0,0)
- ε_sz = tolerance × min_edge_length
- Loop max_iter: Δ = F(ξ,η)-target; if ‖Δ‖ < ε_sz break; (ξ,η) += J⁻¹Δ
- inside: |ξ|,|η| < 1 + 20×tolerance

### Tests

`axikernels/core/handlers/tests/test_isoparametric.py` — **48 tests**, all passing.

| Class | Tests |
|-------|-------|
| `TestLagrange1D` | Kronecker delta (GLL+GLJ), sum-to-1, exact quadratic (6 tests) |
| `TestAxialDetection` | True/False/tolerance/just-above/spherical (5 tests) |
| `TestReferenceAbscissae` | non-axial→GLL, axial→GLJ×GLL (2 tests) |
| `TestForwardMap` | Recovers nodes (linear/spherical/axial), interior bilinear (4 tests) |
| `TestJacobian` | Exact linear, finite-difference spherical (2 tests) |
| `TestNewtonInverse` | Nodes (linear/spherical/axial), interior, outside, interior spherical (6 tests) |
| `TestNewtonFailure` | Singular Jacobian, max-iter exhaustion, strict boundary rule (3 tests) |
| `TestInterpolationWeights` | Sum-to-1 (GLL+GLJ), Kronecker, bilinear, shape, corners (9 tests) |
| `TestElementSearch` | KDTree construction, centre means, find at centre/boundary/outside/correct element/each quadrant, batch consistency, batch shapes, batch NaN, k-clamp (11 tests) |
