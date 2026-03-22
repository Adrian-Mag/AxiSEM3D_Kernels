"""
Tests for axikernels/core/handlers/isoparametric.py

Phase 1: Reference-space isoparametric interpolation for AxiSEM3D element output.
All test data is synthetic — no simulation files required.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from axikernels.core.handlers.isoparametric import (
    GLL_SUBSET,
    GLJ_SUBSET,
    lagrange_1d,
    lagrange_weights_1d,
    detect_axial,
    reference_abscissae,
    compute_min_edge_length,
    forward_map_9node,
    jacobian_9node,
    newton_inverse,
    interpolation_weights_9node,
    build_element_kdtree,
    find_containing_element,
    find_containing_elements_batch,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic element constructors
# ---------------------------------------------------------------------------

def _make_linear_element():
    """
    Rectangular non-axial element.

    Corner mapping (ipol=ξ outer, jpol=η inner, ipnt = ipol*3 + jpol):
      (ipol=0, jpol=0) → corner (-1,-1) → (s=3_480_000, z=       0)
      (ipol=0, jpol=2) → corner (-1,+1) → (s=3_480_000, z= 500_000)
      (ipol=2, jpol=0) → corner (+1,-1) → (s=3_580_000, z=       0)
      (ipol=2, jpol=2) → corner (+1,+1) → (s=3_580_000, z= 500_000)

    The bilinear map is:
      s(ξ,η) = 3_530_000 + 50_000 * ξ
      z(ξ,η) =   250_000 + 250_000 * η
    """
    xi_nodes = GLL_SUBSET   # [-1, 0, 1]
    eta_nodes = GLL_SUBSET
    coords = np.zeros((9, 2), dtype=np.float64)
    for ipol, xi in enumerate(xi_nodes):
        for jpol, eta in enumerate(eta_nodes):
            ipnt = ipol * 3 + jpol
            coords[ipnt, 0] = 3_530_000.0 + 50_000.0 * xi   # s
            coords[ipnt, 1] =   250_000.0 + 250_000.0 * eta  # z
    return coords, xi_nodes, eta_nodes


def _make_spherical_element():
    """
    Curved non-axial element from a polar-grid segment.

    r  ∈ [6_000_000, 6_371_000] m
    θ  ∈ [0.5, 0.6] rad  (colatitude from symmetry axis)

    ref coords → (r, θ) bilinearly:
        r(ξ)  = r_mid  + Δr/2  * ξ    (r_mid  = 6_185_500, Δr  = 371_000)
        θ(η)  = θ_mid  + Δθ/2  * η    (θ_mid  = 0.55,      Δθ  = 0.1)
    Then  s = r sin θ,  z = r cos θ.
    """
    xi_nodes = GLL_SUBSET
    eta_nodes = GLL_SUBSET
    r_mid = (6_000_000.0 + 6_371_000.0) / 2.0
    dr = (6_371_000.0 - 6_000_000.0)
    th_mid = (0.5 + 0.6) / 2.0
    dth = 0.6 - 0.5

    coords = np.zeros((9, 2), dtype=np.float64)
    for ipol, xi in enumerate(xi_nodes):
        for jpol, eta in enumerate(eta_nodes):
            ipnt = ipol * 3 + jpol
            r = r_mid + dr / 2.0 * xi
            th = th_mid + dth / 2.0 * eta
            coords[ipnt, 0] = r * np.sin(th)  # s
            coords[ipnt, 1] = r * np.cos(th)  # z
    return coords, xi_nodes, eta_nodes


def _make_axial_element():
    """
    Axial element: LEFT edge (ipol=0, jpol=0,1,2) has s=0.

    Uses GLJ subset for ξ_nodes.
    Corner mapping:
      (ipol=0, jpol=0) → (-1,-1) → (s=0,  z=-500_000)
      (ipol=0, jpol=2) → (-1,+1) → (s=0,  z= 500_000)
      (ipol=2, jpol=0) → (+1,-1) → (s=100_000, z=-500_000)
      (ipol=2, jpol=2) → (+1,+1) → (s=100_000, z= 500_000)

    Bilinear map:
      s(ξ,η) = 50_000 + 50_000 * ξ   (so s=0 at ξ=-1)
      z(ξ,η) = 500_000 * η
    """
    xi_nodes = GLJ_SUBSET   # [-1, 0.132300820777, 1]
    eta_nodes = GLL_SUBSET
    coords = np.zeros((9, 2), dtype=np.float64)
    for ipol, xi in enumerate(xi_nodes):
        for jpol, eta in enumerate(eta_nodes):
            ipnt = ipol * 3 + jpol
            coords[ipnt, 0] = 50_000.0 + 50_000.0 * xi   # s
            coords[ipnt, 1] = 500_000.0 * eta             # z
    return coords, xi_nodes, eta_nodes


def _make_collapsed_element():
    """
    Degenerate element: all 9 nodes at the same point (1_000_000, 500_000).

    The Jacobian is identically zero everywhere, so np.linalg.solve will
    raise LinAlgError — triggering the singular-Jacobian branch in
    newton_inverse.
    """
    xi_nodes = GLL_SUBSET
    eta_nodes = GLL_SUBSET
    coords = np.full((9, 2), [1_000_000.0, 500_000.0], dtype=np.float64)
    return coords, xi_nodes, eta_nodes


# ---------------------------------------------------------------------------
# 1. Lagrange basis
# ---------------------------------------------------------------------------

class TestLagrange1D:
    """Tests for lagrange_1d and lagrange_weights_1d."""

    @pytest.mark.parametrize("nodes", [GLL_SUBSET, GLJ_SUBSET])
    def test_lagrange_kronecker_delta(self, nodes):
        """L_i(x_j) must equal δ_ij for all i,j."""
        n = len(nodes)
        for i in range(n):
            for j, xj in enumerate(nodes):
                val = lagrange_1d(xj, nodes, i)
                expected = 1.0 if i == j else 0.0
                npt.assert_allclose(
                    val, expected, atol=1e-14,
                    err_msg=f"L_{i}(x_{j}) = {val}, expected {expected}"
                )

    @pytest.mark.parametrize("nodes", [GLL_SUBSET, GLJ_SUBSET])
    def test_lagrange_weights_sum_to_one(self, nodes):
        """Weights must sum to 1 at arbitrary interior points (partition of unity)."""
        for t in [-0.7, -0.3, 0.0, 0.15, 0.45, 0.9]:
            w = lagrange_weights_1d(t, nodes)
            npt.assert_allclose(
                np.sum(w), 1.0, atol=1e-14,
                err_msg=f"weights don't sum to 1 at t={t}"
            )

    def test_lagrange_exact_quadratic(self):
        """3-node GLL Lagrange interpolation must reproduce a quadratic exactly."""
        nodes = GLL_SUBSET  # [-1, 0, 1]
        f_nodes = np.array([3.0, -1.0, 5.0])  # f(-1)=3, f(0)=-1, f(1)=5
        # True polynomial: determined by the 3 values
        # Using Lagrange: p(t) = sum_i f_i * L_i(t)
        for t in np.linspace(-1, 1, 17):
            w = lagrange_weights_1d(t, nodes)
            p_val = np.dot(w, f_nodes)
            # Fit reference polynomial
            coeffs = np.polyfit(nodes, f_nodes, 2)
            p_ref = np.polyval(coeffs, t)
            npt.assert_allclose(p_val, p_ref, atol=1e-12)

    def test_lagrange_weights_kronecker_delta(self):
        """lagrange_weights_1d(x_j, nodes)[i] == δ_ij."""
        nodes = GLL_SUBSET
        for j, xj in enumerate(nodes):
            w = lagrange_weights_1d(xj, nodes)
            expected = np.zeros(len(nodes))
            expected[j] = 1.0
            npt.assert_allclose(w, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Axial detection and reference abscissae
# ---------------------------------------------------------------------------

class TestAxialDetection:
    def test_detect_axial_true(self):
        """Element with s=0 on left edge is correctly detected as axial."""
        coords, _, _ = _make_axial_element()
        assert detect_axial(coords) is True

    def test_detect_axial_false(self):
        """Rectangular interior element is correctly identified as non-axial."""
        coords, _, _ = _make_linear_element()
        assert detect_axial(coords) is False

    def test_detect_axial_tolerance(self):
        """Left edge with |s| < tol × min_edge_length is detected as axial."""
        coords, _, _ = _make_axial_element()
        # Perturb left edge s-values by a tiny amount (< 1e-3 × edge_length)
        min_edge = compute_min_edge_length(coords)
        tiny = 1e-5 * min_edge  # << tol threshold → still axial
        coords[0, 0] = tiny
        coords[1, 0] = tiny / 2.0
        coords[2, 0] = tiny / 3.0
        assert detect_axial(coords) is True

    def test_detect_axial_tolerance_just_above(self):
        """Left edge with |s| >> tol × min_edge_length is NOT detected as axial."""
        coords, _, _ = _make_axial_element()
        min_edge = compute_min_edge_length(coords)
        big_s = 0.1 * min_edge  # >> tol threshold → not axial
        coords[0, 0] = big_s
        coords[1, 0] = big_s
        coords[2, 0] = big_s
        assert detect_axial(coords) is False

    def test_spherical_not_axial(self):
        """Spherical element far from axis is non-axial."""
        coords, _, _ = _make_spherical_element()
        assert detect_axial(coords) is False


class TestReferenceAbscissae:
    def test_non_axial_returns_gll(self):
        xi_nodes, eta_nodes = reference_abscissae(axial=False)
        npt.assert_allclose(xi_nodes, GLL_SUBSET, atol=1e-15)
        npt.assert_allclose(eta_nodes, GLL_SUBSET, atol=1e-15)

    def test_axial_returns_glj_gll(self):
        xi_nodes, eta_nodes = reference_abscissae(axial=True)
        npt.assert_allclose(xi_nodes, GLJ_SUBSET, atol=1e-15)
        npt.assert_allclose(eta_nodes, GLL_SUBSET, atol=1e-15)


# ---------------------------------------------------------------------------
# 3. Forward map
# ---------------------------------------------------------------------------

class TestForwardMap:
    def test_forward_map_recovers_nodes_linear(self):
        """F(ξ_i, η_j) == stored coords X_ij for all 9 nodes (linear element)."""
        coords, xi_nodes, eta_nodes = _make_linear_element()
        for ipol, xi in enumerate(xi_nodes):
            for jpol, eta in enumerate(eta_nodes):
                ipnt = ipol * 3 + jpol
                s_ref, z_ref = coords[ipnt]
                s_calc, z_calc = forward_map_9node(xi, eta, coords, xi_nodes, eta_nodes)
                npt.assert_allclose(s_calc, s_ref, rtol=1e-12,
                                    err_msg=f"s mismatch at ipnt={ipnt}")
                npt.assert_allclose(z_calc, z_ref, rtol=1e-12,
                                    err_msg=f"z mismatch at ipnt={ipnt}")

    def test_forward_map_recovers_nodes_spherical(self):
        """F(ξ_i, η_j) == stored coords for spherical element."""
        coords, xi_nodes, eta_nodes = _make_spherical_element()
        for ipol, xi in enumerate(xi_nodes):
            for jpol, eta in enumerate(eta_nodes):
                ipnt = ipol * 3 + jpol
                s_ref, z_ref = coords[ipnt]
                s_calc, z_calc = forward_map_9node(xi, eta, coords, xi_nodes, eta_nodes)
                npt.assert_allclose(s_calc, s_ref, rtol=1e-12)
                npt.assert_allclose(z_calc, z_ref, rtol=1e-12)

    def test_forward_map_recovers_nodes_axial(self):
        """F(ξ_i, η_j) == stored coords for axial element (GLJ × GLL nodes)."""
        coords, xi_nodes, eta_nodes = _make_axial_element()
        for ipol, xi in enumerate(xi_nodes):
            for jpol, eta in enumerate(eta_nodes):
                ipnt = ipol * 3 + jpol
                s_ref, z_ref = coords[ipnt]
                s_calc, z_calc = forward_map_9node(xi, eta, coords, xi_nodes, eta_nodes)
                npt.assert_allclose(s_calc, s_ref, rtol=1e-12)
                npt.assert_allclose(z_calc, z_ref, rtol=1e-12)

    def test_forward_map_interior_linear(self):
        """F at an interior point recovers the exact bilinear formula."""
        coords, xi_nodes, eta_nodes = _make_linear_element()
        xi, eta = 0.37, -0.21
        s_calc, z_calc = forward_map_9node(xi, eta, coords, xi_nodes, eta_nodes)
        s_exact = 3_530_000.0 + 50_000.0 * xi
        z_exact = 250_000.0 + 250_000.0 * eta
        npt.assert_allclose(s_calc, s_exact, rtol=1e-12)
        npt.assert_allclose(z_calc, z_exact, rtol=1e-12)


class TestJacobian:
    def test_jacobian_linear_element_exact(self):
        """
        For the linear element:
          s(ξ,η) = 3_530_000 + 50_000*ξ   → ∂s/∂ξ = 50_000, ∂s/∂η = 0
          z(ξ,η) =   250_000 + 250_000*η  → ∂z/∂ξ = 0, ∂z/∂η = 250_000

        J = [[∂s/∂ξ, ∂s/∂η],   =  [[50_000,      0],
             [∂z/∂ξ, ∂z/∂η]]       [      0, 250_000]]
        """
        coords, xi_nodes, eta_nodes = _make_linear_element()
        for xi, eta in [(0.0, 0.0), (0.5, -0.3), (-0.7, 0.9)]:
            J = jacobian_9node(xi, eta, coords, xi_nodes, eta_nodes)
            npt.assert_allclose(J[0, 0], 50_000.0, rtol=1e-10)
            npt.assert_allclose(J[0, 1], 0.0, atol=1e-7)
            npt.assert_allclose(J[1, 0], 0.0, atol=1e-7)
            npt.assert_allclose(J[1, 1], 250_000.0, rtol=1e-10)

    def test_jacobian_finite_difference(self):
        """Check Jacobian against finite-difference approximation for spherical element."""
        coords, xi_nodes, eta_nodes = _make_spherical_element()
        xi, eta = 0.1, -0.4
        h = 1e-5
        J = jacobian_9node(xi, eta, coords, xi_nodes, eta_nodes)

        s0, z0 = forward_map_9node(xi, eta, coords, xi_nodes, eta_nodes)
        s1x, z1x = forward_map_9node(xi + h, eta, coords, xi_nodes, eta_nodes)
        s1y, z1y = forward_map_9node(xi, eta + h, coords, xi_nodes, eta_nodes)

        npt.assert_allclose(J[0, 0], (s1x - s0) / h, rtol=1e-6)  # ∂s/∂ξ
        npt.assert_allclose(J[0, 1], (s1y - s0) / h, rtol=1e-6)  # ∂s/∂η
        npt.assert_allclose(J[1, 0], (z1x - z0) / h, rtol=1e-6)  # ∂z/∂ξ
        npt.assert_allclose(J[1, 1], (z1y - z0) / h, rtol=1e-6)  # ∂z/∂η


# ---------------------------------------------------------------------------
# 4. Newton inverse mapping
# ---------------------------------------------------------------------------

class TestNewtonInverse:
    def test_newton_converges_at_nodes_linear(self):
        """Newton returns the correct (ξ_i, η_j) when given the stored node coords."""
        coords, xi_nodes, eta_nodes = _make_linear_element()
        for ipol, xi_ref in enumerate(xi_nodes):
            for jpol, eta_ref in enumerate(eta_nodes):
                ipnt = ipol * 3 + jpol
                s_target, z_target = coords[ipnt]
                xi_calc, eta_calc, converged, inside = newton_inverse(
                    s_target, z_target, coords, xi_nodes, eta_nodes
                )
                assert converged, f"Did not converge at node ({ipol},{jpol})"
                assert inside, f"Node ({ipol},{jpol}) reported outside"
                npt.assert_allclose(xi_calc, xi_ref, atol=1e-8)
                npt.assert_allclose(eta_calc, eta_ref, atol=1e-8)

    def test_newton_converges_at_nodes_spherical(self):
        """Newton converges at all 9 stored nodes of the spherical element."""
        coords, xi_nodes, eta_nodes = _make_spherical_element()
        for ipol, xi_ref in enumerate(xi_nodes):
            for jpol, eta_ref in enumerate(eta_nodes):
                ipnt = ipol * 3 + jpol
                s_target, z_target = coords[ipnt]
                xi_calc, eta_calc, converged, inside = newton_inverse(
                    s_target, z_target, coords, xi_nodes, eta_nodes
                )
                assert converged
                assert inside
                npt.assert_allclose(xi_calc, xi_ref, atol=1e-7)
                npt.assert_allclose(eta_calc, eta_ref, atol=1e-7)

    def test_newton_converges_at_nodes_axial(self):
        """Newton converges at all 9 stored nodes of the axial element."""
        coords, xi_nodes, eta_nodes = _make_axial_element()
        for ipol, xi_ref in enumerate(xi_nodes):
            for jpol, eta_ref in enumerate(eta_nodes):
                ipnt = ipol * 3 + jpol
                s_target, z_target = coords[ipnt]
                xi_calc, eta_calc, converged, inside = newton_inverse(
                    s_target, z_target, coords, xi_nodes, eta_nodes
                )
                assert converged
                assert inside
                npt.assert_allclose(xi_calc, xi_ref, atol=1e-7)
                npt.assert_allclose(eta_calc, eta_ref, atol=1e-7)

    def test_newton_converges_interior(self):
        """Newton converges for an interior point of the linear element."""
        coords, xi_nodes, eta_nodes = _make_linear_element()
        xi_true, eta_true = 0.37, -0.21
        s_target, z_target = forward_map_9node(xi_true, eta_true, coords, xi_nodes, eta_nodes)
        xi_calc, eta_calc, converged, inside = newton_inverse(
            s_target, z_target, coords, xi_nodes, eta_nodes
        )
        assert converged
        assert inside
        npt.assert_allclose(xi_calc, xi_true, atol=1e-8)
        npt.assert_allclose(eta_calc, eta_true, atol=1e-8)

    def test_newton_reports_outside(self):
        """Newton returns inside=False for a point clearly outside the element."""
        coords, xi_nodes, eta_nodes = _make_linear_element()
        # Point far outside the element domain
        s_outside = 4_000_000.0  # well outside [3_480_000, 3_580_000]
        z_outside = 250_000.0
        _, _, _, inside = newton_inverse(
            s_outside, z_outside, coords, xi_nodes, eta_nodes
        )
        assert inside is False

    def test_newton_converges_interior_spherical(self):
        """Newton converges for an interior point of the spherical element."""
        coords, xi_nodes, eta_nodes = _make_spherical_element()
        xi_true, eta_true = -0.5, 0.4
        s_target, z_target = forward_map_9node(xi_true, eta_true, coords, xi_nodes, eta_nodes)
        xi_calc, eta_calc, converged, inside = newton_inverse(
            s_target, z_target, coords, xi_nodes, eta_nodes
        )
        assert converged
        assert inside
        npt.assert_allclose(xi_calc, xi_true, atol=1e-7)
        npt.assert_allclose(eta_calc, eta_true, atol=1e-7)


# ---------------------------------------------------------------------------
# 4b. Newton inverse — failure modes
# ---------------------------------------------------------------------------

class TestNewtonFailure:
    """Tests for newton_inverse failure modes: converged=False forces inside=False."""

    def test_singular_jacobian_converged_false_inside_false(self):
        """Collapsed element → singular Jacobian → converged=False and inside=False."""
        coords, xi_nodes, eta_nodes = _make_collapsed_element()
        # Target away from the degenerate map point to ensure a non-zero residual
        _, _, converged, inside = newton_inverse(
            2_000_000.0, 500_000.0, coords, xi_nodes, eta_nodes
        )
        assert converged is False
        assert inside is False

    def test_max_iter_exhaustion_converged_false_inside_false(self):
        """max_iter=1 on spherical element (non-linear) → loop exhausted → converged=False, inside=False."""
        coords, xi_nodes, eta_nodes = _make_spherical_element()
        # Target not at F(0,0) — any interior point far from origin works
        xi_true, eta_true = 0.8, 0.7
        s_target, z_target = forward_map_9node(
            xi_true, eta_true, coords, xi_nodes, eta_nodes
        )
        _, _, converged, inside = newton_inverse(
            s_target, z_target, coords, xi_nodes, eta_nodes, max_iter=1
        )
        assert converged is False
        assert inside is False

    def test_strict_boundary_rule(self):
        """
        Linear element: test the ≤ boundary at ξ = 1 + 20×tol.

        With bound = 1 + 20×tol (inclusive):
          ξ = 1 + 20×tol  → inside=True   (exactly on the boundary)
          ξ = 1 + 21×tol  → inside=False  (just outside)
        """
        coords, xi_nodes, eta_nodes = _make_linear_element()
        tol = 1e-6  # larger tolerance avoids float64 rounding at the boundary

        # --- point exactly at ξ = 1 + 20*tol, η = 0 ---
        xi_at_bound = 1.0 + 20.0 * tol
        s_at = 3_530_000.0 + 50_000.0 * xi_at_bound
        _, _, conv_in, inside_in = newton_inverse(
            s_at, 250_000.0, coords, xi_nodes, eta_nodes, tolerance=tol
        )
        assert conv_in, "Newton should converge for ξ = 1 + 20×tol"
        assert inside_in, "ξ = 1 + 20×tol should be inside (inclusive boundary)"

        # --- point at ξ = 1 + 21*tol, η = 0 ---
        xi_outside_bound = 1.0 + 21.0 * tol
        s_out = 3_530_000.0 + 50_000.0 * xi_outside_bound
        _, _, conv_out, inside_out = newton_inverse(
            s_out, 250_000.0, coords, xi_nodes, eta_nodes, tolerance=tol
        )
        assert conv_out, "Newton should converge for ξ = 1 + 21×tol"
        assert not inside_out, "ξ = 1 + 21×tol should be outside"


# ---------------------------------------------------------------------------
# 5. Interpolation weights
# ---------------------------------------------------------------------------

class TestInterpolationWeights:
    @pytest.mark.parametrize("xi,eta", [
        (0.0, 0.0), (-0.5, 0.3), (0.7, -0.8), (-1.0, 1.0)
    ])
    def test_weights_sum_to_one(self, xi, eta):
        """Interpolation weights must sum to 1 (partition of unity)."""
        w = interpolation_weights_9node(xi, eta, GLL_SUBSET, GLL_SUBSET)
        npt.assert_allclose(np.sum(w), 1.0, atol=1e-14)

    def test_weights_sum_to_one_glj(self):
        """Also holds for GLJ × GLL nodes."""
        for xi in [-0.3, 0.132300820777, 0.6]:
            for eta in [-0.7, 0.0, 0.4]:
                w = interpolation_weights_9node(xi, eta, GLJ_SUBSET, GLL_SUBSET)
                npt.assert_allclose(np.sum(w), 1.0, atol=1e-14)

    def test_weights_kronecker_delta_at_nodes(self):
        """At stored node (ξ_i, η_j): weight vector is e_{ipol*3+jpol}."""
        xi_nodes = GLL_SUBSET
        eta_nodes = GLL_SUBSET
        for ipol, xi in enumerate(xi_nodes):
            for jpol, eta in enumerate(eta_nodes):
                ipnt = ipol * 3 + jpol
                w = interpolation_weights_9node(xi, eta, xi_nodes, eta_nodes)
                expected = np.zeros(9)
                expected[ipnt] = 1.0
                npt.assert_allclose(w, expected, atol=1e-13,
                                    err_msg=f"weights wrong at node ({ipol},{jpol})")

    def test_weights_exact_bilinear_linear_element(self):
        """
        Interpolation must exactly reproduce the field on the linear element.

        Field: f(s, z) = 2*s - 3*z + 1
        Values at the 9 nodes are stored, then recovered at arbitrary interior points.
        """
        coords, xi_nodes, eta_nodes = _make_linear_element()
        f_nodes = 2.0 * coords[:, 0] - 3.0 * coords[:, 1] + 1.0  # shape (9,)

        rng = np.random.default_rng(42)
        test_points = rng.uniform(-0.9, 0.9, size=(20, 2))
        for xi, eta in test_points:
            w = interpolation_weights_9node(xi, eta, xi_nodes, eta_nodes)
            f_interp = np.dot(w, f_nodes)
            # Exact value from forward map
            s_pt, z_pt = forward_map_9node(xi, eta, coords, xi_nodes, eta_nodes)
            f_exact = 2.0 * s_pt - 3.0 * z_pt + 1.0
            npt.assert_allclose(f_interp, f_exact, rtol=1e-11,
                                err_msg=f"bilinear mismatch at ({xi:.3f},{eta:.3f})")

    def test_weights_shape(self):
        """Weight vector has shape (9,) for both GLL and GLJ nodes."""
        w_gll = interpolation_weights_9node(0.1, -0.2, GLL_SUBSET, GLL_SUBSET)
        w_glj = interpolation_weights_9node(0.1, -0.2, GLJ_SUBSET, GLL_SUBSET)
        assert w_gll.shape == (9,)
        assert w_glj.shape == (9,)

    def test_weights_nonnegative_at_corners(self):
        """At corner nodes the single nonzero weight must be 1."""
        corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        expected_idx = [0, 2, 6, 8]  # ipol*3+jpol
        for (xi, eta), idx in zip(corners, expected_idx):
            w = interpolation_weights_9node(xi, eta, GLL_SUBSET, GLL_SUBSET)
            npt.assert_allclose(w[idx], 1.0, atol=1e-14)
            other = np.delete(w, idx)
            npt.assert_allclose(other, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 6. Element search: KDTree + Newton containment
# ---------------------------------------------------------------------------

def _make_2x2_grid():
    """Build a 2×2 rectangular element grid covering (s,z) ∈ [3000,5000]×[0,2000]."""
    def _make_elem(s_min, s_max, z_min, z_max):
        coords = np.zeros((9, 2), dtype=np.float64)
        for ipol in range(3):
            for jpol in range(3):
                xi = GLL_SUBSET[ipol]
                eta = GLL_SUBSET[jpol]
                s = s_min + (s_max - s_min) * (xi + 1) / 2
                z = z_min + (z_max - z_min) * (eta + 1) / 2
                ipnt = ipol * 3 + jpol
                coords[ipnt] = [s, z]
        return coords

    elem0 = _make_elem(3000.0, 4000.0,    0.0, 1000.0)
    elem1 = _make_elem(4000.0, 5000.0,    0.0, 1000.0)
    elem2 = _make_elem(3000.0, 4000.0, 1000.0, 2000.0)
    elem3 = _make_elem(4000.0, 5000.0, 1000.0, 2000.0)
    return np.array([elem0, elem1, elem2, elem3])


class TestElementSearch:
    """Tests for build_element_kdtree, find_containing_element, find_containing_elements_batch."""

    @pytest.fixture
    def grid_2x2(self):
        """Build a 2×2 rectangular element grid."""
        all_coords = _make_2x2_grid()
        kdtree, centers = build_element_kdtree(all_coords)
        return all_coords, kdtree, centers

    def test_kdtree_construction(self, grid_2x2):
        """KDTree has correct number of entries."""
        all_coords, kdtree, centers = grid_2x2
        assert centers.shape == (4, 2)
        assert kdtree.n == 4

    def test_centers_are_element_means(self, grid_2x2):
        """Centers are the mean of the 9 nodes per element."""
        all_coords, kdtree, centers = grid_2x2
        expected_centers = np.mean(all_coords, axis=1)
        npt.assert_allclose(centers, expected_centers, atol=1e-12)

    def test_find_element_at_center(self, grid_2x2):
        """Point at element center is correctly assigned."""
        all_coords, kdtree, _ = grid_2x2
        # Center of element 0: s ∈ [3000,4000], z ∈ [0,1000] → center (3500, 500)
        idx, xi, eta = find_containing_element(3500.0, 500.0, all_coords, kdtree)
        assert idx == 0
        assert abs(xi) < 1e-6
        assert abs(eta) < 1e-6

    def test_find_element_at_boundary(self, grid_2x2):
        """Point on shared edge finds a valid element (either neighbor)."""
        all_coords, kdtree, _ = grid_2x2
        # (4000, 500) is on boundary between elem 0 and elem 1
        idx, xi, eta = find_containing_element(4000.0, 500.0, all_coords, kdtree)
        assert idx in (0, 1)  # Either neighbor is acceptable

    def test_find_element_outside_all(self, grid_2x2):
        """Point far outside returns -1."""
        all_coords, kdtree, _ = grid_2x2
        idx, xi, eta = find_containing_element(100000.0, 100000.0, all_coords, kdtree)
        assert idx == -1
        assert np.isnan(xi) and np.isnan(eta)

    def test_find_element_correct_among_neighbors(self, grid_2x2):
        """Selects correct element when multiple candidates are near."""
        all_coords, kdtree, _ = grid_2x2
        # (4800, 1800) should be in element 3
        idx, xi, eta = find_containing_element(4800.0, 1800.0, all_coords, kdtree)
        assert idx == 3

    def test_find_element_each_quadrant(self, grid_2x2):
        """Interior point of each element is found correctly."""
        all_coords, kdtree, _ = grid_2x2
        # Centres of the four elements
        centers_expected = [
            (3500.0,  500.0, 0),
            (4500.0,  500.0, 1),
            (3500.0, 1500.0, 2),
            (4500.0, 1500.0, 3),
        ]
        for s, z, expected_idx in centers_expected:
            idx, xi, eta = find_containing_element(s, z, all_coords, kdtree)
            assert idx == expected_idx, f"Expected element {expected_idx} for ({s},{z}), got {idx}"
            npt.assert_allclose(xi, 0.0, atol=1e-6)
            npt.assert_allclose(eta, 0.0, atol=1e-6)

    def test_find_element_batch_consistency(self, grid_2x2):
        """Batch and single-point results agree."""
        all_coords, kdtree, _ = grid_2x2
        test_points = np.array([
            [3500.0,  500.0],     # centre of elem 0
            [4500.0, 1500.0],     # centre of elem 3
            [100000.0, 100000.0], # outside
        ])
        indices, xis, etas = find_containing_elements_batch(test_points, all_coords, kdtree)
        for i in range(len(test_points)):
            idx_s, xi_s, eta_s = find_containing_element(
                test_points[i, 0], test_points[i, 1], all_coords, kdtree
            )
            assert indices[i] == idx_s
            if idx_s >= 0:
                npt.assert_allclose(xis[i], xi_s, atol=1e-10)
                npt.assert_allclose(etas[i], eta_s, atol=1e-10)

    def test_batch_return_shapes(self, grid_2x2):
        """Batch function returns arrays of correct shape and dtype."""
        all_coords, kdtree, _ = grid_2x2
        points = np.array([[3500.0, 500.0], [4500.0, 1500.0]])
        indices, xis, etas = find_containing_elements_batch(points, all_coords, kdtree)
        assert indices.shape == (2,)
        assert xis.shape == (2,)
        assert etas.shape == (2,)
        assert indices.dtype == np.intp or np.issubdtype(indices.dtype, np.integer)

    def test_batch_outside_point_nan_values(self, grid_2x2):
        """Unmatched points in batch have NaN xi/eta and index -1."""
        all_coords, kdtree, _ = grid_2x2
        points = np.array([[1e8, 1e8]])
        indices, xis, etas = find_containing_elements_batch(points, all_coords, kdtree)
        assert indices[0] == -1
        assert np.isnan(xis[0])
        assert np.isnan(etas[0])

    def test_k_clamped_to_n_elements(self, grid_2x2):
        """k larger than number of elements doesn't raise an error."""
        all_coords, kdtree, _ = grid_2x2
        # k=1000 >> 4 elements; should still return a valid result
        idx, xi, eta = find_containing_element(3500.0, 500.0, all_coords, kdtree, k=1000)
        assert idx == 0
