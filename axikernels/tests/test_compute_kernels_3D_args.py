"""
test_compute_kernels_3D_args.py
================================
Tests for argument parsing in compute_kernels_3D.py.
"""

import sys
import os
import importlib
import pytest

# Path to the script under test
_SCRIPT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "axisem3d_root", "AxiSEM3D", "examples", "adrian_kernel_3D",
)
_SCRIPT_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "compute_kernels_3D.py"))


def _load_module():
    """Import compute_kernels_3D as a module by injecting its directory into sys.path."""
    script_dir = os.path.normpath(_SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("compute_kernels_3D", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── load once at collection time ──────────────────────────────────────────────
_mod = _load_module()
parse_args = _mod.parse_args


class TestParseArgsDefaults:
    """parse_args() returns correct defaults when no sys.argv arguments are given."""

    def setup_method(self):
        # sys.argv must only contain the script name for argparse defaults to apply
        self._orig_argv = sys.argv[:]
        sys.argv = ["compute_kernels_3D.py"]

    def teardown_method(self):
        sys.argv = self._orig_argv

    def test_forward_default(self):
        args = parse_args()
        assert args.forward == "simu_forward"

    def test_output_default(self):
        args = parse_args()
        assert args.output == "kernel_output"

    def test_tau_default(self):
        args = parse_args()
        assert args.tau == pytest.approx(2.0)

    def test_receiver_default(self):
        args = parse_args()
        assert args.receiver == pytest.approx([0.0, 40.0])

    def test_window_default(self):
        args = parse_args()
        assert args.window == pytest.approx([425.0, 475.0])

    def test_channel_default(self):
        args = parse_args()
        assert args.channel == "UZ"

    def test_cores_default(self):
        args = parse_args()
        assert args.cores == 8

    def test_resolution_default(self):
        args = parse_args()
        assert args.resolution == 200

    def test_topography_default(self):
        args = parse_args()
        assert args.topography == "input_forward/moho_topography.nc"


class TestParseArgsCustomTopography:
    """parse_args() correctly parses a custom --topography path."""

    def setup_method(self):
        self._orig_argv = sys.argv[:]

    def teardown_method(self):
        sys.argv = self._orig_argv

    def test_custom_topography_path(self):
        sys.argv = ["compute_kernels_3D.py", "--topography", "/some/path.nc"]
        args = parse_args()
        assert args.topography == "/some/path.nc"

    def test_custom_topography_relative_path(self):
        sys.argv = ["compute_kernels_3D.py", "--topography", "data/topo.nc"]
        args = parse_args()
        assert args.topography == "data/topo.nc"

    def test_other_args_unchanged_with_topography(self):
        """Other defaults are preserved when only --topography is overridden."""
        sys.argv = ["compute_kernels_3D.py", "--topography", "/data/topo.nc"]
        args = parse_args()
        assert args.topography == "/data/topo.nc"
        assert args.tau == pytest.approx(2.0)
        assert args.cores == 8
        assert args.channel == "UZ"

    def test_topography_combined_with_other_args(self):
        """--topography can be combined with other arguments."""
        sys.argv = [
            "compute_kernels_3D.py",
            "--topography", "/topo/moho.nc",
            "--tau", "3.5",
            "--cores", "16",
        ]
        args = parse_args()
        assert args.topography == "/topo/moho.nc"
        assert args.tau == pytest.approx(3.5)
        assert args.cores == 16
