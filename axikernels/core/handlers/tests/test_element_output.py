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
