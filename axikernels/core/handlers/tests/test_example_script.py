"""
test_example_script.py
======================
End-to-end smoke test for ``examples/example_obspy_workflows.py``.

Runs the script once as a subprocess and asserts that all key output
markers are present.  No fixtures, no real simulation data, no files
written under the source tree.  The script itself creates a temporary
directory and cleans it up before exiting.
"""
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_SCRIPT = os.path.join(_REPO_ROOT, "examples", "example_obspy_workflows.py")


def test_example_obspy_workflows_end_to_end():
    """examples/example_obspy_workflows.py exits 0 and reports key workflow results."""
    result = subprocess.run(
        [sys.executable, _SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = result.stdout
    assert result.returncode == 0, (
        f"Script exited {result.returncode}.\n"
        f"--- stdout ---\n{out}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "SECTION 1" in out, "Missing station-output section header"
    assert "SECTION 2" in out, "Missing element-output section header"
    assert "Stations : 5" in out, "Script did not report the expected station count"
    assert "Events: 1" in out, "Script did not report the expected event count"
    assert "Traces : 6" in out, "Two-station stream summary is missing or incorrect"
    assert "Traces: 15" in out, "All-station MiniSEED trace count is missing or incorrect"
    assert "Inventory: 5 stations" in out, "Reloaded inventory station count is missing or incorrect"
    assert "Files written to" in out, "obspyfy() did not report written files"
    assert "Station_grid.mseed" in out, "MiniSEED output file was not reported"
    assert "cat.xml" in out, "QuakeML output file was not reported"
    assert "ElementOutput class confirmed importable" in out, \
        "ElementOutput import marker missing"
    assert "Done." in out, "Script did not reach clean exit"
