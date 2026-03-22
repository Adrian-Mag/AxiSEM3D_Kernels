# axikernels Handlers — Living Reference

_Last updated: realistic-demo Phase 2_

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

**Key methods:**

| Name | Notes |
|------|-------|
| `stream_STA(path_to_station_file, channels, ...)` | Interpolates wavefield at station locations. Phase 2: `header=None` fix, `trace.stats.npts` typo fixed. |
| `create_inventory(path_to_station_file)` | Builds `obspy.Inventory`. Phase 2: `header=None` fix (was `header=0`). |
| `obspyfy(path_to_station_file, channels)` | Writes MiniSEED + XML |

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
| `test_element_output.py` | **NEW Phase 2** — 5 tests: `test_source_no_header_zero` (source guard), `test_station_csv_all_rows` (behavioral), `test_station_csv_header_zero_drops` (behavioral), `test_source_no_ntps_typo` (source guard), `test_trace_npts_via_obspy` (behavioral). Element tests are minimal by design — full element testing deferred to a future phase with synthetic mesh fixture. |
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
