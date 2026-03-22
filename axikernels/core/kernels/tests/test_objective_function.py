"""Source-guard tests for objective_function.py bugs.

These tests read source code to verify structural fixes without needing
to instantiate the classes (which require heavy data dependencies).
"""
import inspect
import ast
import importlib


def _get_module():
    import axikernels.core.kernels.objective_function as mod
    return mod


def _get_source():
    mod = _get_module()
    return inspect.getsource(mod)


# ---------------------------------------------------------------------------
# Bug 1: self.real_data not stored in __init__
# ---------------------------------------------------------------------------

def test_real_data_stored():
    """ObjectiveFunction.__init__ must store self.real_data = real_data."""
    mod = _get_module()
    src = inspect.getsource(mod.ObjectiveFunction.__init__)
    assert 'self.real_data' in src, (
        "ObjectiveFunction.__init__ does not store self.real_data"
    )


# ---------------------------------------------------------------------------
# Bug 2 & 4: Earth_Radius used instead of Domain_Radius
# ---------------------------------------------------------------------------

def test_no_earth_radius():
    """No occurrence of 'Earth_Radius' should remain anywhere in the module."""
    src = _get_source()
    assert 'Earth_Radius' not in src, (
        "'Earth_Radius' still appears in objective_function.py; "
        "should be replaced with 'Domain_Radius'"
    )


# ---------------------------------------------------------------------------
# Bug 3: channels=['U'] passed to stream()
# ---------------------------------------------------------------------------

def test_stream_no_channels_U():
    """_compute_adjoint_STF must not pass channels=['U'] to stream()."""
    mod = _get_module()
    src = inspect.getsource(mod.L2ObjectiveFunction._compute_adjoint_STF)
    assert "channels=['U']" not in src, (
        "_compute_adjoint_STF still passes channels=['U'] to stream(); "
        "stream() does not accept that parameter"
    )


# ---------------------------------------------------------------------------
# Bug 5: self.forward_data.data_time used in evaluate_objective_function
# ---------------------------------------------------------------------------

def test_no_data_time_attribute():
    """evaluate_objective_function must not access self.forward_data.data_time."""
    mod = _get_module()
    src = inspect.getsource(mod.L2ObjectiveFunction.evaluate_objective_function)
    assert 'forward_data.data_time' not in src, (
        "evaluate_objective_function still uses self.forward_data.data_time "
        "which does not exist on ElementOutput"
    )


# ---------------------------------------------------------------------------
# Bug 6: self.forward_data.coordinate_frame used in evaluate_objective_function
# ---------------------------------------------------------------------------

def test_no_coordinate_frame_attribute():
    """evaluate_objective_function must not access self.forward_data.coordinate_frame."""
    mod = _get_module()
    src = inspect.getsource(mod.L2ObjectiveFunction.evaluate_objective_function)
    assert 'forward_data.coordinate_frame' not in src, (
        "evaluate_objective_function still uses self.forward_data.coordinate_frame "
        "which does not exist on ElementOutput"
    )


# ---------------------------------------------------------------------------
# Bug 7: coord_in_deg=True missing in evaluate_objective_function stream call
# ---------------------------------------------------------------------------

def test_evaluate_objective_passes_coord_in_deg():
    """evaluate_objective_function must pass coord_in_deg=True to stream()."""
    mod = _get_module()
    src = inspect.getsource(mod.L2ObjectiveFunction.evaluate_objective_function)
    assert 'coord_in_deg=True' in src, (
        "evaluate_objective_function does not pass coord_in_deg=True to stream()"
    )


# ---------------------------------------------------------------------------
# Bug 8: typo '_compute_RT_totation_matrix' (missing 'r')
# ---------------------------------------------------------------------------

def test_rotation_method_name():
    """The typo '_compute_RT_totation_matrix' must not exist; correct name must."""
    src = _get_source()
    assert 'totation' not in src, (
        "'totation' typo still present in objective_function.py"
    )
    # The corrected method must exist on any concrete class in the module
    mod = _get_module()
    classes = [
        obj for name, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__module__ == mod.__name__
    ]
    has_correct = any(
        '_compute_RT_rotation_matrix' in cls.__dict__ or
        hasattr(cls, '_compute_RT_rotation_matrix')
        for cls in classes
    )
    assert has_correct, (
        "'_compute_RT_rotation_matrix' method not found in any class "
        "in objective_function.py"
    )


# ---------------------------------------------------------------------------
# Phase 6 post-review: channels= passed to stream() calls
# ---------------------------------------------------------------------------

def test_stream_passes_channels():
    """_compute_adjoint_STF must pass channels= keyword to stream()."""
    mod = _get_module()
    src = inspect.getsource(mod.L2ObjectiveFunction._compute_adjoint_STF)
    assert 'channels=' in src, (
        "_compute_adjoint_STF does not pass channels= to stream()"
    )


def test_evaluate_objective_passes_channels():
    """evaluate_objective_function must pass channels= keyword to stream()."""
    mod = _get_module()
    src = inspect.getsource(mod.L2ObjectiveFunction.evaluate_objective_function)
    assert 'channels=' in src, (
        "evaluate_objective_function does not pass channels= to stream()"
    )


# ---------------------------------------------------------------------------
# Phase 6 post-review: window attributes initialized in ObjectiveFunction
# ---------------------------------------------------------------------------

def test_window_attributes_initialized():
    """ObjectiveFunction.__init__ must initialize window_left and window_right."""
    mod = _get_module()
    src = inspect.getsource(mod.ObjectiveFunction.__init__)
    assert 'self.window_left' in src, (
        "ObjectiveFunction.__init__ does not initialize self.window_left"
    )
    assert 'self.window_right' in src, (
        "ObjectiveFunction.__init__ does not initialize self.window_right"
    )
