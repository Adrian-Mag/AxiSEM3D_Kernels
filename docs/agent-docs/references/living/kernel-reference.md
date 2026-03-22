# axikernels Kernel — Living Reference

_Last updated: 3D Kernel Example Phase 1_

---

## Package root
`axikernels/core/kernels/`

---

## Classes

### `ObjectiveFunction` (`objective_function.py`, abstract)
Abstract base class for adjoint-based objective functions.

**Constructor:**
```python
ObjectiveFunction(forward_data: ElementOutput,
                  real_data: ObspyfiedOutput = None,
                  backward_simulation: ElementOutput = None,
                  interactive: bool = True)
```

**Instance attributes (Phase 6 fix – `self.real_data` now stored):**

| Attribute | Notes |
|-----------|-------|
| `forward_data` | `ElementOutput` of forward simulation. |
| `real_data` | `ObspyfiedOutput` of observed seismograms. **Bug 1 fixed Phase 6.** |
| `backward_simulation` | `ElementOutput` of adjoint simulation, or `None`. |
| `interactive` | If `True`, uses `input()` prompts at key steps. |
| `kernel` | `Kernel` instance, populated by `initialize_kernels()`. |

**Important conventions fixed in Phase 6:**
- Use `self.forward_data.Domain_Radius` (not `Earth_Radius`) to get the domain radius.
- Access time via metadata: `element_groups_info[first_group]['metadata']['data_time']`.
- Access coordinate frame via: `element_groups_info[first_group]['wavefields']['coordinate_frame']`.
- `ElementOutput.stream(point, coord_in_deg=True)` — no `channels` kwarg.
- Rotation method is `_compute_RT_rotation_matrix` (typo `totation` fixed Phase 6).

### `XObjectiveFunction` (`objective_function.py`)
Concrete `ObjectiveFunction` subclass implementing cross-correlation adjoint method.

**Key method:** `_compute_RT_rotation_matrix(receiver_point)` — renamed from `_compute_RT_totation_matrix` in Phase 6.

### `L2ObjectiveFunction` (`objective_function.py`)
Concrete `ObjectiveFunction` subclass implementing L2 waveform-difference adjoint method.

**Key methods:**
- `_compute_adjoint_STF(...)` — computes adjoint source-time function from data residual.
- `evaluate_objective_function(network, station, location, plot_residue)` → `float`.

---

### `Kernel` (`kernel.py`)
Computes sensitivity kernels by cross-correlating forward and adjoint wavefields.

**Constructor:**
```python
Kernel(forward_obj: ElementOutput, backward_obj: ElementOutput)
```
- Stores forward/adjoint `ElementOutput` objects.
- Calls `_compute_times()` to build `master_time`, `fw_time`, `bw_time`.

**Class attributes:**

| Attribute | Type | Notes |
|-----------|------|-------|
| `_discontinuity_kernels` | `set[str]` | `{'Kd', 'K_dn', 'dV', 'SS', 'CMB_solid', 'CMB_fluid', 'geometric'}` — kernels that require a `radius` argument. **Added Phase 5.** |

**Instance attributes set by `__init__`:**

| Attribute | Notes |
|-----------|-------|
| `forward_data` | The `ElementOutput` for the forward simulation. |
| `backward_data` | The `ElementOutput` for the adjoint simulation. |
| `fw_time`, `bw_time` | Raw time axes from the first element group. `bw_time` is time-reversed: `np.flip(T_max - bw_time_raw)`. |
| `fw_dt`, `bw_dt` | Time steps of forward/backward. |
| `master_time` | Common time axis at `dt = max(fw_dt, bw_dt)` over the `[max(t_min), min(t_max)]` overlap. |
| `kernel_types` | `dict[str, callable]` mapping kernel name → evaluation method. **Phase 5: added** `'Kd'`, `'K_dn'`, `'SS'`, `'CMB_solid'`, `'CMB_fluid'`, `'geometric'`. |

**`kernel_types` dispatch table** (complete as of Phase 5):

| Key | Method | Signature | Type |
|-----|--------|-----------|------|
| `'rho_0'` | `evaluate_rho_0` | `(points)` | volumetric |
| `'lambda'` | `evaluate_lambda` | `(points)` | volumetric |
| `'mu'` | `evaluate_mu` | `(points)` | volumetric |
| `'rho'` | `evaluate_rho` | `(points)` | volumetric |
| `'vp'` | `evaluate_vp` | `(points)` | volumetric |
| `'vs'` | `evaluate_vs` | `(points)` | volumetric |
| `'dV'` | `evaluate_K_dv` | `(points, radius)` | discontinuity |
| `'Kd'` | `evaluate_Kd` | `(points, radius)` | discontinuity |
| `'K_dn'` | `evaluate_K_dn` | `(points, radius)` | discontinuity |
| `'SS'` | `evaluate_SS` | `(points, radius)` | discontinuity |
| `'CMB_solid'` | `evaluate_CMB_solid` | `(points, radius)` | discontinuity |
| `'CMB_fluid'` | `evaluate_CMB_fluid` | `(points, radius)` | discontinuity |
| `'geometric'` | `evaluate_geometric` | `(points, radius)` | discontinuity |

---

## Public evaluation methods

### Volumetric kernels (take `points` only)

| Method | Description |
|--------|-------------|
| `evaluate_rho_0(points)` | Density kernel (kinetic energy term `∫ ∂ₜu·∂ₜuᵀ dt`). |
| `evaluate_lambda(points)` | λ kernel. Handles solid (trace-strain) and fluid (P) branches. |
| `evaluate_mu(points)` | μ kernel. Handles solid (full strain tensor) and fluid (P) branches. **Phase 5 bug fix:** `dt` moved before the `if len(solid_points)` block so it is defined even when `solid_points` is empty. |
| `evaluate_rho(points)` | Total density kernel = `K_rho_0 + (vp²−2vs²)K_λ + vs² K_μ`. |
| `evaluate_vp(points)` | Vp kernel = `2ρ·vp·K_λ`. |
| `evaluate_vs(points)` | Vs kernel = `2ρ·vs·(K_μ − 2K_λ)`. |

### Discontinuity kernels (take `points` AND `radius`)

| Method | Description |
|--------|-------------|
| `evaluate_K_dv(points, radius)` | Volumetric-geometry kernel at `radius`. Dispatches on `SS`/`FS`/`SF` discontinuity type. |
| `evaluate_K_dn(points, radius)` | Normal-displacement kernel at `radius`. **Phase 5 bug fix:** FS and SF branches now squeeze P arrays to `(N, T)` → correct flip axis (1 not 2), correct `np.interp` sources, correct integrand `factor[:, np.newaxis] * P_fwd * P_bwd`. **Channel-order bug fix (post Phase 6):** All three branches (SS, FS, SF) now load forward gradient channels as `['GZR', 'GZZ', 'GZT']`, matching backward loads. Previously forward used `['GZR', 'GZT', 'GZZ']` (T and Z swapped), corrupting the traction-gradient cross-products and producing near-zero kernels. |
| `evaluate_Kd(points, radius)` | Total discontinuity kernel = `K_dn + K_dv`. |
| `evaluate_SS(points, radius)` | Solid–solid discontinuity kernel. |
| `evaluate_CMB_solid(points, radius)` | CMB kernel contribution from the solid side. |
| `evaluate_CMB_fluid(points, radius)` | CMB kernel contribution from the fluid side. |
| `evaluate_geometric(points, radius)` | Geometric kernel (stub — not yet implemented). |

---

## Dispatch / evaluation entry points

### `evaluate_on_sphere_2(n, radius, parameter)`
Evaluates a kernel on a spherical mesh of resolution `n` at `radius`.

**Phase 5 fix:** Previously passed `radius` unconditionally to all kernels, crashing for volumetric kernels with `TypeError`. Now:
```python
if parameter in self._discontinuity_kernels:
    data = self.kernel_types[parameter](mesh.points, radius)
else:
    data = self.kernel_types[parameter](mesh.points)
```

### `evaluate_on_slice(parameter, ..., radius=None)`
Evaluates a kernel on a 2-D great-circle slice mesh.

**Phase 5 fix:** Added `radius: float = None` parameter. Dispatch:
```python
if parameter in self._discontinuity_kernels:
    if radius is None:
        raise ValueError(f"radius is required for discontinuity kernel '{parameter}'")
    data = self.kernel_types[parameter](mesh.points, radius)
else:
    data = self.kernel_types[parameter](mesh.points)
```

---

## Private helper methods

### `_compute_times()`
Sets `fw_time`, `bw_time`, `fw_dt`, `bw_dt`, `master_time`.

### `_find_discontinuity_type(radius) → str`
Returns `'SS'`, `'FS'`, or `'SF'` for the discontinuity at `radius`.
Raises `ValueError` if `radius` is not in `base_model['DISCONTINUITIES']`.

### `_form_limit_points(points, radius) → (upper_points, lower_points)`
Constructs `[radius ± 1000 m, lat, lon]` arrays for evaluating wavefields just above/below a discontinuity.

### `_find_material_property(points, material_property) → np.ndarray`
Looks up a 1-D model property at given radii using `searchsorted`.

**Phase 5 fixes:**
- **Bug 2:** Changed `np.logical_or` → `np.logical_and` in the boundary mask so only points strictly inside the model are kept.
- **Bug 3:** Added `np.clip(filtered_index, 1, len(radii) - 1)` before the `- 1` subtraction to prevent `index=0` from wrapping to `-1` (last element).

---

## Tests

`axikernels/core/kernels/tests/test_kernel.py` — 18 source-guard tests added in Phase 5.

`axikernels/core/kernels/tests/test_objective_function.py` — 7 source-guard tests added in Phase 6.

All tests are pure `inspect.getsource` checks — no real simulation data required.

### test_kernel.py

| Test class | Bugs covered |
|------------|--------------|
| `TestEvaluateMuDtScoping` | Bug 1 (dt indentation / scoping) |
| `TestFindMaterialProperty` | Bugs 2+3 (logical_and, np.clip) |
| `TestKernelTypesComplete` | Bug 4 (missing kernel_types entries) |
| `TestEvaluateOnSphere2Dispatch` | Bug 5 (unconditional radius arg) |
| `TestEvaluateOnSliceRadius` | Bug 6 (missing radius param) |
| `TestEvaluateKdnFsBranch` | Bug 7 (FS undefined vars + integrand) |
| `TestEvaluateKdnSfBranch` | Bug 8 (SF undefined vars + integrand) |

### test_objective_function.py

| Test function | Bug covered |
|---------------|-------------|
| `test_real_data_stored` | Bug 1: `self.real_data` not stored in `ObjectiveFunction.__init__` |
| `test_no_earth_radius` | Bugs 2+4: `Earth_Radius` → `Domain_Radius` |
| `test_stream_no_channels_U` | Bug 3: `channels=['U']` removed from `stream()` call |
| `test_no_data_time_attribute` | Bug 5: `forward_data.data_time` → metadata path |
| `test_no_coordinate_frame_attribute` | Bug 6: `forward_data.coordinate_frame` → metadata path |
| `test_evaluate_objective_passes_coord_in_deg` | Bug 7: `coord_in_deg=True` added to `stream()` |
| `test_rotation_method_name` | Bug 8: `totation` typo → `rotation` |

---

## 3D example: `adrian_kernel_3D`

Location: `axisem3d_root/AxiSEM3D/examples/adrian_kernel_3D/`

### Preprocessing scripts

| Script | Output | Description |
|--------|--------|-------------|
| `create_moho_topography.py` | `input_forward/moho_topography.nc` | Gaussian Moho undulation (5 km amplitude, centre 0°N/20°E) for `StructuredGridG3D`. |
| `convert_s362ani_to_radius.py` | `input_forward/S362ANI_radius.nc` | S362ANI converted from depth (km) to radius (m), axis flipped to monotonically increasing, for `StructuredGridV3D`. |

### NetCDF variable conventions

**`moho_topography.nc`:**
- `latitude` (degrees_north, shape 181), `longitude` (degrees_east, shape 361)
- `undulation_MOHO` (m, shape 181×361) — Gaussian, max ≈ 5000 m

**`S362ANI_radius.nc`:**
- `radius` (m, shape 25, monotonically increasing 3 481 000 → 6 346 000)
- `latitude` (degrees, shape 91), `longitude` (degrees, shape 181)
- `dvs`, `dvsv`, `dvsh` (percent, shape 25×91×181)

### `inparam.model.yaml` key parameters
- `MOHO_TOPOGRAPHY` (StructuredGridG3D) must be listed **before** `EMC_S362ANI`
- `undulation_range.interface`: 6346600 m (PREM Moho)
- `EMC_S362ANI` uses `undulated_geometry: true` so it follows the deformed Moho

### Tests (in `axikernels/tests/`)

| File | Tests |
|------|-------|
| `test_create_moho_topo.py` | 8 tests: file exists, variables, lat/lon range, max undulation ≈5000 m, near-zero at pole |
| `test_convert_s362ani.py` | 8 tests: file exists, `radius` not `depth`, monotonically increasing, value correctness, data shape |

---

## Common patterns

### Solid/fluid branch pattern for volumetric kernels
```python
material_mapping = self.forward_data._group_by_material(points)
solid_points = points[material_mapping == 0]
liquid_points = points[material_mapping == 1]
sensitivity = np.zeros(len(points))
dt = self.master_time[1] - self.master_time[0]   # MUST be before any branch
if len(solid_points) > 0:
    ...
    sensitivity[material_mapping == 0] = solid_sensitivities
if len(liquid_points) > 0:
    ...
    # Uses dt — ok because dt is defined at the top level
    sensitivity[material_mapping == 1] = liquid_sensitivities
return sensitivity
```

### P-wave interpolation pattern (fluid, shape (N,T))
```python
# Load and squeeze to (N, T) immediately:
P_fwd = np.nan_to_num(self.forward_data.load_data(..., channels=['P'], ...))[:, 0, :]
P_bwd = np.nan_to_num(self.backward_data.load_data(..., channels=['P'], ...))[:, 0, :]
P_bwd = np.flip(P_bwd, axis=1)   # axis=1 for (N, T) array
P_fwd_interp = np.empty((N, len(master_time)))
for i in range(N):
    P_fwd_interp[i] = np.interp(master_time, fw_time, P_fwd[i])
# Integrand needs factor[:, np.newaxis] for (N,) × (N,T):
integrand = factor[:, np.newaxis] * P_fwd_interp * P_bwd_interp
```
