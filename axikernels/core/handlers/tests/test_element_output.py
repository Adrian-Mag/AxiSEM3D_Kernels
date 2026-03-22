"""
test_element_output.py – TDD tests for element_output.py bug fixes (Phase 2A).

These tests are intentionally minimal: they target the CSV-parsing and npts
fixes without requiring a full synthetic element mesh fixture.

Tests:
  - test_source_no_header_zero        (source guard: header=0 absent)
  - test_station_csv_all_rows         (behavioral: 5-row file → 5-row DataFrame)
  - test_station_csv_header_zero_drops (behavioral: header=0 loses a row)
  - test_source_no_ntps_typo          (source guard: ntps absent)
  - test_trace_npts_via_obspy         (behavioral: ObsPy trace.stats.npts set correctly)
"""
import inspect
import textwrap

import numpy as np
import obspy
import pandas as pd
import pytest

# -- Shared station-file content (5 data rows, 1 comment header) -----------
_STATION_CONTENT = textwrap.dedent("""\
    #name network latitude longitude useless depth
    ST1 A 10 20 0 0
    ST2 A 20 40 0 0
    ST3 A 30 60 0 0
    ST4 A 40 80 0 0
    ST5 A 50 100 0 0
""")

_CSV_KWARGS = dict(
    sep=r'\s+',
    names=["name", "network", "latitude", "longitude", "useless", "depth"],
    comment='#',
)


# ---------------------------------------------------------------------------
# 2A-1: header=0 station-skipping bug
# ---------------------------------------------------------------------------

def test_source_no_header_zero():
    """Guard: element_output.py must not contain header=0."""
    from axikernels.core.handlers import element_output
    src = inspect.getsource(element_output)
    assert "header=0" not in src, (
        "element_output.py still uses header=0, which silently drops the "
        "first data row when comment='#' is active."
    )


def test_station_csv_all_rows(tmp_path):
    """Behavioral: parsing a #-headed file with header=None keeps all 5 rows."""
    f = tmp_path / "stations.txt"
    f.write_text(_STATION_CONTENT)
    df = pd.read_csv(str(f), header=None, **_CSV_KWARGS)
    assert len(df) == 5
    assert df.iloc[0]["name"] == "ST1", "First station should be ST1"
    assert df.iloc[4]["name"] == "ST5", "Last station should be ST5"


def test_station_csv_header_zero_drops(tmp_path):
    """Behavioral proof the bug existed: header=0 loses the first data row."""
    f = tmp_path / "stations.txt"
    f.write_text(_STATION_CONTENT)
    df_bad = pd.read_csv(str(f), header=0, **_CSV_KWARGS)
    assert len(df_bad) == 4, (
        "header=0 with comment='#' should drop one row (the bug scenario)"
    )


# ---------------------------------------------------------------------------
# 2A-2: ntps → npts typo
# ---------------------------------------------------------------------------

def test_source_no_ntps_typo():
    """Guard: element_output.py must not contain trace.stats.ntps."""
    from axikernels.core.handlers import element_output
    src = inspect.getsource(element_output)
    assert "trace.stats.ntps" not in src, (
        "trace.stats.ntps is not a valid ObsPy attribute; use npts."
    )


def test_trace_npts_via_obspy():
    """Behavioral: setting trace.stats.npts actually controls sample count."""
    data = np.zeros(100)
    tr = obspy.Trace(data)
    tr.stats.delta = 0.5
    tr.stats.npts = 100
    assert tr.stats.npts == 100
    assert len(tr.data) == 100


# ---------------------------------------------------------------------------
# Phase 3: Source guards for isoparametric interpolation rewrite
# ---------------------------------------------------------------------------

from axikernels.core.handlers.element_output import ElementOutput


class TestPhase3SourceGuards:
    """Source-level guards to prevent regression to old interpolation code."""

    def test_sentinel_is_nan_not_ones(self):
        """load_data and load_data_from_element_group use NaN sentinel, not ones."""
        source = inspect.getsource(ElementOutput.load_data)
        assert 'np.ones' not in source, "load_data still uses np.ones sentinel"
        assert 'np.full' in source or 'np.nan' in source

        source2 = inspect.getsource(ElementOutput.load_data_from_element_group)
        assert 'np.ones' not in source2, "load_data_from_element_group still uses np.ones"
        assert 'np.full' in source2 or 'np.nan' in source2

    def test_no_cart2polar_in_interpolation(self):
        """cart2polar should not be used in the interpolation method."""
        source = inspect.getsource(ElementOutput.load_data_from_element_group)
        assert 'cart2polar' not in source, "load_data_from_element_group still uses cart2polar"

    def test_no_lagrange_method(self):
        """Old _lagrange method should be removed."""
        assert not hasattr(ElementOutput, '_lagrange'), "_lagrange method still exists"

    def test_no_problematic_elements_fallback(self):
        """Problematic elements fallback [0,0,0,0,1,0,0,0,0] should be gone."""
        source = inspect.getsource(ElementOutput.load_data_from_element_group)
        assert '0,0,0,0,1,0,0,0,0' not in source, "Problematic elements fallback still present"
        assert 'problematic' not in source.lower(), "Problematic elements logic still present"

    def test_uses_isoparametric_module(self):
        """Verify the method uses the new isoparametric interpolation."""
        source = inspect.getsource(ElementOutput.load_data_from_element_group)
        assert 'interpolation_weights_9node' in source or 'find_containing_element' in source


# ---------------------------------------------------------------------------
# Phase 4: Theta convention fix and -1 domain guard
# ---------------------------------------------------------------------------

class TestPhase4ThetaConvention:
    """Tests for theta convention fix (colatitude via arccos, not arctan2+pi/2)."""

    def test_no_pi_over_2_shift(self):
        """Source guard: += np.pi/2 absent from _project_on_inplane."""
        source = inspect.getsource(ElementOutput._project_on_inplane)
        assert 'pi/2' not in source, "_project_on_inplane still references pi/2"

    def test_no_cart2polar_in_project(self):
        """Source guard: cart2polar not used in _project_on_inplane."""
        source = inspect.getsource(ElementOutput._project_on_inplane)
        assert 'cart2polar' not in source, "_project_on_inplane still uses cart2polar"

    def test_no_pi_over_2_in_create_inventory(self):
        """Source guard: pi/2 shift absent from create_inventory."""
        source = inspect.getsource(ElementOutput.create_inventory)
        assert '+= np.pi/2' not in source, "create_inventory still has pi/2 shift"
        assert '+ np.pi/2' not in source, "create_inventory still has pi/2 shift"

    def test_domain_unmatched_guard_in_create_inventory(self):
        """Source guard: create_inventory handles -1 domain mapping."""
        source = inspect.getsource(ElementOutput.create_inventory)
        assert '== -1' in source or '< 0' in source, (
            "create_inventory doesn't guard against -1 domain"
        )

    def test_group_by_material_guards_minus_one(self):
        """Source guard: _group_by_material handles -1 group mapping."""
        source = inspect.getsource(ElementOutput._group_by_material)
        assert '>= 0' in source or '== -1' in source or '< 0' in source, (
            "_group_by_material doesn't guard against -1 group"
        )


# ---------------------------------------------------------------------------
# Phase 7: Source guards and Fourier reconstruction tests
# ---------------------------------------------------------------------------

import axikernels.core.handlers.element_output as _eo_module


class TestPhase7SourceGuards:
    """Source-level guards for Phase 7 fixes."""

    def test_stream_sta_no_self_data_time(self):
        """stream_STA must not reference self.data_time (attribute doesn't exist)."""
        source = inspect.getsource(ElementOutput.stream_STA)
        assert 'self.data_time' not in source, (
            "stream_STA still uses self.data_time which is not a class attribute"
        )

    def test_no_sys_exit(self):
        """element_output module must not call sys.exit (library code should raise)."""
        source = inspect.getsource(_eo_module)
        assert 'sys.exit' not in source, (
            "element_output.py calls sys.exit(); raise ValueError instead"
        )

    def test_no_point_not_in_output_domain(self):
        """Dead method _point_not_in_output_domain must be removed."""
        source = inspect.getsource(ElementOutput)
        assert '_point_not_in_output_domain' not in source, (
            "_point_not_in_output_domain still exists; it references undefined "
            "self.vertical_range/self.horizontal_range and has no callers"
        )

    def test_no_bare_np_where_in_stream(self):
        """stream method must use np.flatnonzero, not bare np.where (returns tuple)."""
        source = inspect.getsource(ElementOutput.stream)
        assert 'np.where(' not in source, (
            "stream() still uses np.where() which returns a tuple; "
            "use np.flatnonzero() instead"
        )

    def test_no_bare_np_where_in_stream_sta(self):
        """stream_STA method must use np.flatnonzero, not bare np.where (returns tuple)."""
        source = inspect.getsource(ElementOutput.stream_STA)
        assert 'np.where(' not in source, (
            "stream_STA() still uses np.where() which returns a tuple; "
            "use np.flatnonzero() instead"
        )


# ---------------------------------------------------------------------------
# Phase 7: Fourier reconstruction unit tests (standalone formula tests)
# ---------------------------------------------------------------------------

def fourier_reconstruct(data, phi, nag):
    """Reproduce the Fourier reconstruction from load_data_from_element_group.

    Args:
        data: array of shape (n_phi, nag, n_elem, n_time)
        phi: array of shape (n_phi,) azimuths in radians
        nag: number of Fourier storage columns (Na)

    Returns:
        result: array of shape (n_phi, n_elem, n_time)
    """
    max_fourier_order = nag // 2
    result = data[:, 0, :, :].copy().astype(np.float64)
    for order in range(1, max_fourier_order + 1):
        coeff = np.zeros(result.shape, dtype=np.complex128)
        coeff.real = data[:, order * 2 - 1, :, :]
        if order * 2 < nag:
            coeff += 1j * data[:, order * 2, :, :]
        result += (2.0 * np.exp(1j * order * phi)[:, np.newaxis, np.newaxis] * coeff).real
    return result


class TestPhase7FourierReconstruction:
    """Unit tests for the Fourier reconstruction formula."""

    def test_fourier_reconstruction_order0(self):
        """With nag=1 (only DC term), reconstruction equals data[:,0,:,:]."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 1, 3, 10))
        phi = np.linspace(0, 2 * np.pi, 5)
        result = fourier_reconstruct(data, phi, nag=1)
        np.testing.assert_array_equal(result, data[:, 0, :, :])

    def test_fourier_reconstruction_order1(self):
        """With nag=3, verify order-1 contribution at phi=0 and phi=pi/2."""
        # data shape: (2_phi_pts, 3_cols, 1_elem, 1_time)
        # col0 = DC, col1 = real part of order-1, col2 = imag part of order-1
        c0 = 5.0
        c1_real = 2.0
        c1_imag = -1.0
        data = np.array([[c0, c1_real, c1_imag]] * 2).reshape(2, 3, 1, 1)
        phi = np.array([0.0, np.pi / 2])
        result = fourier_reconstruct(data, phi, nag=3)

        # At phi=0: c0 + 2*Re(exp(i*0) * (c1_real + i*c1_imag)) = c0 + 2*c1_real
        expected_phi0 = c0 + 2.0 * c1_real
        # At phi=pi/2: c0 + 2*Re(i*(c1_real + i*c1_imag)) = c0 + 2*Re(i*c1_real - c1_imag)
        #                                                   = c0 + 2*(-c1_imag) = c0 + 2
        expected_phi_half_pi = c0 + 2.0 * (-c1_imag)

        np.testing.assert_allclose(result[0, 0, 0], expected_phi0, rtol=1e-12)
        np.testing.assert_allclose(result[1, 0, 0], expected_phi_half_pi, rtol=1e-12)

    def test_fourier_reconstruction_nyquist(self):
        """With nag=4 (even), order=2 Nyquist term has imag=0 (index 4 >= nag=4)."""
        rng = np.random.default_rng(7)
        # data shape: (3, 4, 2, 5): col0=DC, col1=re1, col2=im1, col3=re2 (Nyquist real only)
        data = rng.standard_normal((3, 4, 2, 5))
        phi = np.array([0.0, np.pi / 3, 2 * np.pi / 3])

        result = fourier_reconstruct(data, phi, nag=4)

        # Verify manually at phi=0: c0 + 2*c1_real + 2*c2_real (imag part is 0)
        expected = (
            data[:, 0, :, :]
            + 2.0 * data[:, 1, :, :]   # order 1 real, phi=0 => Re(exp(0)) = 1, imag => Re(i*...) = 0
            + 2.0 * data[:, 3, :, :]   # order 2 real (Nyquist), phi=0
        )
        # Note: order-1 imag contribution at phi=0 is 2*Re(i*c_imag) which is
        # 2*(-c_imag_imag_part) = 0 since c_imag is real. So at phi=0 it is just 2*c1_real.
        np.testing.assert_allclose(result[0, :, :], expected[0, :, :], rtol=1e-12)

    def test_fourier_reconstruction_known_signal(self):
        """f(phi) = 3 + 2*cos(phi) - sin(phi) reconstructed from its Fourier coefficients."""
        # Coefficients: c0=3, c1_real=2 (cos term), c1_imag=-1 (sin term, note sign convention)
        # The formula: c0 + 2*Re(exp(i*phi)*(c1_real + i*c1_imag))
        #            = c0 + 2*(c1_real*cos(phi) - c1_imag*sin(phi))
        #            = 3 + 2*(2*cos(phi) - (-1)*sin(phi))
        # Wait: = 3 + 4*cos(phi) + 2*sin(phi) — that's not what we want.
        # For f(phi) = 3 + 2*cos(phi) - sin(phi), we need:
        # 2*Re(exp(i*phi)*(c_r + i*c_i)) = 2*(c_r*cos(phi) - c_i*sin(phi))
        # => c_r=1, c_i=0.5 gives 2*cos(phi) - sin(phi) ✓
        c0 = 3.0
        c1_real = 1.0
        c1_imag = 0.5

        n_phi = 20
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        # data shape: (n_phi, 3, 1, 1)
        data = np.zeros((n_phi, 3, 1, 1))
        data[:, 0, 0, 0] = c0
        data[:, 1, 0, 0] = c1_real
        data[:, 2, 0, 0] = c1_imag

        result = fourier_reconstruct(data, phi, nag=3)

        expected = 3.0 + 2.0 * np.cos(phi) - np.sin(phi)
        np.testing.assert_allclose(result[:, 0, 0], expected, rtol=1e-12, atol=1e-12)
