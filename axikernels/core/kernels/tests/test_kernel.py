"""
Source-guard tests for Kernel (axikernels/core/kernels/kernel.py).

All tests are pure source-inspection tests using inspect.getsource — they do
*not* instantiate Kernel (which requires real AxiSEM3D simulation data).

Bugs tested:
  1. evaluate_mu: dt scoped inside solid branch → NameError when only fluid points.
  2. _find_material_property: mask drops boundary points → shape mismatch downstream.
  3. _find_material_property: filtered_index - 1 wraps to -1 at boundary.
  4. kernel_types dict: missing Kd, K_dn, SS, CMB_solid, CMB_fluid, geometric.
  5. evaluate_on_sphere_2: unconditionally passes radius to volumetric kernels.
  6. evaluate_on_slice: no radius parameter → discontinuity kernels always fail.
  7. evaluate_K_dn FS branch: P loaded from wrong side (solid instead of fluid).
  8. evaluate_K_dn SF branch: P loaded from wrong side (solid instead of fluid).
"""

import inspect
import pytest
from axikernels.core.kernels.kernel import Kernel


# ---------------------------------------------------------------------------
# Bug 1 – evaluate_mu dt scoping
# ---------------------------------------------------------------------------

class TestEvaluateMuDtScoping:
    """dt must be defined before the liquid branch to avoid NameError."""

    def test_dt_defined_before_liquid_branch(self):
        """dt must be at the same indentation level as the branch checks
        (i.e., NOT nested inside the solid block)."""
        source = inspect.getsource(Kernel.evaluate_mu)
        lines = source.splitlines()

        dt_line = None
        liquid_branch_line = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('dt = self.master_time') and dt_line is None:
                dt_line = line
            if 'if len(liquid_points)' in stripped and liquid_branch_line is None:
                liquid_branch_line = line

        assert dt_line is not None, \
            "dt = self.master_time[...] not found in evaluate_mu"
        assert liquid_branch_line is not None, \
            "'if len(liquid_points)' not found in evaluate_mu"

        dt_indent = len(dt_line) - len(dt_line.lstrip())
        liquid_indent = len(liquid_branch_line) - len(liquid_branch_line.lstrip())

        assert dt_indent <= liquid_indent, (
            f"Bug 1: dt is inside a nested block (indent {dt_indent} spaces) "
            f"but must be at the same top-level scope as  the liquid branch "
            f"(indent {liquid_indent} spaces). Move dt before the solid block."
        )

    def test_only_one_dt_assignment_in_evaluate_mu(self):
        """After the fix there should be exactly one dt assignment."""
        source = inspect.getsource(Kernel.evaluate_mu)
        count = sum(
            1 for line in source.splitlines()
            if line.strip().startswith('dt = self.master_time')
        )
        assert count == 1, (
            f"Bug 1: expected exactly one 'dt = self.master_time...' in "
            f"evaluate_mu, found {count}"
        )


# ---------------------------------------------------------------------------
# Bugs 2 & 3 – _find_material_property mask and clamping
# ---------------------------------------------------------------------------

class TestFindMaterialProperty:
    """Bug 2 (mask removed) + Bug 3 (index clamped): no mask, just clip."""

    def test_no_mask_applied(self):
        """Mask must be removed entirely; every input point must get a value."""
        source = inspect.getsource(Kernel._find_material_property)
        assert 'np.logical_and' not in source, (
            "_find_material_property should not apply a mask — "
            "use np.clip to handle boundary indices instead."
        )
        assert 'mask' not in source, (
            "_find_material_property must not filter points with a mask. "
            "np.clip ensures every input point gets a valid property value."
        )

    def test_index_clamped_before_subtraction(self):
        source = inspect.getsource(Kernel._find_material_property)
        assert 'np.clip' in source or '.clip(' in source, (
            "Bug 3: _find_material_property should call np.clip on filtered_index "
            "before '- 1' to prevent negative-index wraparound"
        )


# ---------------------------------------------------------------------------
# Bug 4 – kernel_types completeness
# ---------------------------------------------------------------------------

class TestKernelTypesComplete:
    """kernel_types dict must include all kernel methods."""

    required_keys = [
        'rho_0', 'lambda', 'mu', 'rho', 'vp', 'vs', 'dV',
        'Kd', 'K_dn', 'SS', 'CMB_solid', 'CMB_fluid', 'geometric',
    ]

    def test_all_required_types_in_init_source(self):
        source = inspect.getsource(Kernel.__init__)
        for key in self.required_keys:
            present = f"'{key}'" in source or f'"{key}"' in source
            assert present, (
                f"Bug 4: kernel_types dict is missing key '{key}'"
            )


# ---------------------------------------------------------------------------
# Bug 5 – evaluate_on_sphere_2 conditional dispatch
# ---------------------------------------------------------------------------

class TestEvaluateOnSphere2Dispatch:
    """evaluate_on_sphere_2 must not unconditionally pass radius to all kernels."""

    def test_uses_discontinuity_registry(self):
        source = inspect.getsource(Kernel.evaluate_on_sphere_2)
        assert '_discontinuity_kernels' in source or 'if parameter' in source, (
            "Bug 5: evaluate_on_sphere_2 must conditionally dispatch based "
            "on whether the kernel type is a discontinuity kernel"
        )

    def test_no_unconditional_radius_arg(self):
        """The old single unconditional call with radius must be gone."""
        source = inspect.getsource(Kernel.evaluate_on_sphere_2)
        lines = source.splitlines()
        # Count lines that call kernel_types with (mesh.points, radius)
        unconditional = [
            ln for ln in lines
            if 'self.kernel_types[parameter](mesh.points, radius)' in ln
            and not ln.strip().startswith('#')
        ]
        # There may still be a conditional branch with that call, but there
        # must also be a branch *without* radius (for volumetric kernels).
        has_no_radius_branch = any(
            'self.kernel_types[parameter](mesh.points)' in ln
            for ln in lines
        )
        assert has_no_radius_branch, (
            "Bug 5: evaluate_on_sphere_2 must have a branch that calls "
            "kernel_types[parameter](mesh.points) WITHOUT radius for "
            "volumetric kernels"
        )


# ---------------------------------------------------------------------------
# Bug 6 – evaluate_on_slice radius parameter
# ---------------------------------------------------------------------------

class TestEvaluateOnSliceRadius:
    """evaluate_on_slice must accept a radius parameter for discontinuity kernels."""

    def test_has_radius_param(self):
        sig = inspect.signature(Kernel.evaluate_on_slice)
        assert 'radius' in sig.parameters, (
            "Bug 6: evaluate_on_slice must have a 'radius' parameter "
            "so discontinuity kernels can be evaluated on a slice"
        )

    def test_radius_defaults_to_none(self):
        sig = inspect.signature(Kernel.evaluate_on_slice)
        assert 'radius' in sig.parameters, "radius parameter missing"
        assert sig.parameters['radius'].default is None, (
            "Bug 6: evaluate_on_slice 'radius' parameter should default to None"
        )

    def test_dispatch_uses_radius_for_discontinuity_kernels(self):
        source = inspect.getsource(Kernel.evaluate_on_slice)
        assert '_discontinuity_kernels' in source or 'if parameter' in source, (
            "Bug 6: evaluate_on_slice should conditionally pass radius "
            "only for discontinuity kernel types"
        )


# ---------------------------------------------------------------------------
# Bug 7 – evaluate_K_dn FS branch: P from fluid (lower), G/T from solid (upper)
# ---------------------------------------------------------------------------

class TestEvaluateKdnFsBranch:
    """FS: upper=SOLID, lower=FLUID. P must come from lower; G/T from upper."""

    def _get_fs_branch_source(self):
        source = inspect.getsource(Kernel.evaluate_K_dn)
        parts = source.split("elif disc_type == 'FS':")
        assert len(parts) >= 2, "FS branch not found in evaluate_K_dn"
        fs_and_beyond = parts[1]
        # Stop at the next elif (SF branch)
        fs_only = fs_and_beyond.split("elif disc_type == 'SF':")[0]
        return fs_only

    def test_p_squeezed_with_colons(self):
        """P array must be squeezed to (N, T) via [:, 0, :]."""
        fs = self._get_fs_branch_source()
        assert "[:, 0, :]" in fs or "[:,0,:]" in fs, (
            "FS branch: P should be squeezed with [:, 0, :] after loading "
            "(shape must be (N, T) not (N, 1, T))"
        )

    def test_gr_forward_upper_defined_in_fs(self):
        """G wavefields must be loaded from upper_points (SOLID side) in FS."""
        fs = self._get_fs_branch_source()
        assert 'Gr_forward_upper' in fs, (
            "FS branch must define Gr_forward_upper (solid side = upper in FS). "
            "G wavefields must NOT be loaded from lower_points (the fluid side)."
        )

    def test_gr_backward_upper_defined_in_fs(self):
        """G adjoint wavefields must come from upper_points (SOLID side) in FS."""
        fs = self._get_fs_branch_source()
        assert 'Gr_backward_upper' in fs, (
            "FS branch must define Gr_backward_upper (solid side = upper in FS). "
            "G wavefields must NOT be loaded from lower_points (the fluid side)."
        )

    def test_integrand_uses_newaxis_not_sum_axis1(self):
        """factor must be broadcast with [:, np.newaxis], not via np.sum(axis=1)."""
        import re
        fs = self._get_fs_branch_source()
        # Normalise whitespace to catch multi-line expressions
        fs_flat = re.sub(r'\s+', ' ', fs)
        assert 'factor[:, np.newaxis]' in fs_flat, (
            "FS integrand should multiply P arrays by "
            "factor[:, np.newaxis] (P has shape (N,T) after squeezing), "
            "not by factor * np.sum(..., axis=1)"
        )

    def test_evaluate_K_dn_fs_loads_P_from_fluid_side(self):
        """P must be loaded from lower_points (FLUID side) in FS branch."""
        fs = self._get_fs_branch_source()
        assert 'P_forward_lower' in fs, (
            "FS branch must load P from lower_points (the FLUID side: "
            "vs_lower == 0). Got P from upper (SOLID) which is wrong."
        )
        assert 'P_forward_upper' not in fs, (
            "FS branch must NOT load P from upper_points (the SOLID side). "
            "P is a fluid wavefield and belongs on the fluid (lower) side."
        )


# ---------------------------------------------------------------------------
# Bug 8 – evaluate_K_dn SF branch: P from fluid (upper), G/T from solid (lower)
# ---------------------------------------------------------------------------

class TestEvaluateKdnSfBranch:
    """SF: upper=FLUID, lower=SOLID. P must come from upper; G/T from lower."""

    def _get_sf_branch_source(self):
        source = inspect.getsource(Kernel.evaluate_K_dn)
        parts = source.split("elif disc_type == 'SF':")
        assert len(parts) >= 2, "SF branch not found in evaluate_K_dn"
        return parts[1]

    def test_p_squeezed_with_colons(self):
        """P array must be squeezed to (N, T) via [:, 0, :]."""
        sf = self._get_sf_branch_source()
        assert "[:, 0, :]" in sf or "[:,0,:]" in sf, (
            "SF branch: P should be squeezed with [:, 0, :] after loading "
            "(shape must be (N, T) not (N, 1, T))"
        )

    def test_gr_forward_lower_defined_in_sf(self):
        """G wavefields must be loaded from lower_points (SOLID side) in SF."""
        sf = self._get_sf_branch_source()
        assert 'Gr_forward_lower' in sf, (
            "SF branch must define Gr_forward_lower (solid side = lower in SF). "
            "G wavefields must NOT be loaded from upper_points (the fluid side)."
        )

    def test_gr_backward_lower_defined_in_sf(self):
        """G adjoint wavefields must come from lower_points (SOLID side) in SF."""
        sf = self._get_sf_branch_source()
        assert 'Gr_backward_lower' in sf, (
            "SF branch must define Gr_backward_lower (solid side = lower in SF). "
            "G wavefields must NOT be loaded from upper_points (the fluid side)."
        )

    def test_integrand_uses_newaxis_not_sum_axis1(self):
        """factor must be broadcast with [:, np.newaxis], not via np.sum(axis=1)."""
        import re
        sf = self._get_sf_branch_source()
        fs_flat = re.sub(r'\s+', ' ', sf)
        assert 'factor[:, np.newaxis]' in fs_flat, (
            "SF integrand should multiply P arrays by "
            "factor[:, np.newaxis] (P has shape (N,T) after squeezing), "
            "not by factor * np.sum(..., axis=1)"
        )

    def test_evaluate_K_dn_sf_loads_P_from_fluid_side(self):
        """P must be loaded from upper_points (FLUID side) in SF branch."""
        sf = self._get_sf_branch_source()
        assert 'P_forward_upper' in sf, (
            "SF branch must load P from upper_points (the FLUID side: "
            "vs_upper == 0). Got P from lower (SOLID) which is wrong."
        )
        assert 'P_forward_lower' not in sf, (
            "SF branch must NOT load P from lower_points (the SOLID side). "
            "P is a fluid wavefield and belongs on the fluid (upper) side."
        )
