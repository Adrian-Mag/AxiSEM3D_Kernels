#!/usr/bin/env bash
# run_demo.sh — Run the handlers_demo tiny AxiSEM3D simulation.
#
# This script lives alongside the binary and input bundle it uses, so all
# paths are relative to this directory.  It produces both station output
# (output/stations/GSN_Station_Grid/) and element output
# (output/elements/mantle/) inside a fresh local run directory, ready for
# post-processing with the axikernels handlers.
#
# Usage:
#   ./run_demo.sh [--help] [--dry-run]
#
# Environment variables:
#   NRANKS   Number of MPI ranks  (default: 2)
#
# Examples:
#   ./run_demo.sh --dry-run
#   NRANKS=4 ./run_demo.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve the directory that contains this script (self-contained)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUNDLE_BINARY="${SCRIPT_DIR}/axisem3d"
BUNDLE_INPUT="${SCRIPT_DIR}/input"
RUN_DIR="${SCRIPT_DIR}/sim_run"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            cat <<'EOF'
Usage:
  ./run_demo.sh [--help] [--dry-run]

Options:
  --help      Show this message and exit.
  --dry-run   Validate bundle files and report what would be done without
              running the simulation.

Environment variables:
  NRANKS      Number of MPI ranks (default: 2)

Expected outputs (after a full run):
  sim_run/output/stations/GSN_Station_Grid/
  sim_run/output/elements/mantle/

Example:
  NRANKS=4 ./run_demo.sh
EOF
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            echo "Unknown argument: $arg  (use --help for usage)" >&2
            exit 1
            ;;
    esac
done

NRANKS="${NRANKS:-2}"
if ! [[ "${NRANKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NRANKS must be a positive integer (got: ${NRANKS})" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate bundle files
# ---------------------------------------------------------------------------
for required in \
    "${BUNDLE_BINARY}" \
    "${BUNDLE_INPUT}/inparam.model.yaml" \
    "${BUNDLE_INPUT}/inparam.output.yaml" \
    "${BUNDLE_INPUT}/inparam.source.yaml" \
    "${BUNDLE_INPUT}/inparam.advanced.yaml" \
    "${BUNDLE_INPUT}/inparam.nr.yaml" \
    "${BUNDLE_INPUT}/global_mesh__prem_ani__50s.e" \
    "${BUNDLE_INPUT}/1dmodel_axisem.bm" \
    "${BUNDLE_INPUT}/GSN_small.txt" \
    "${BUNDLE_INPUT}/HANDLERS_EXAMPLE_cat.xml"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: required file not found: ${required}" >&2
        exit 1
    fi
done
echo "Bundle OK: ${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Dry-run: report and exit without touching sim_run/
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo ""
    echo "--- DRY RUN: all bundle files present, no simulation will be run ---"
    echo ""
    echo "Run directory  : ${RUN_DIR}"
    echo "MPI ranks      : ${NRANKS}"
    echo ""
    echo "Expected outputs after a full run:"
    echo "  ${RUN_DIR}/output/stations/GSN_Station_Grid/"
    echo "  ${RUN_DIR}/output/elements/mantle/"
    exit 0
fi

# ---------------------------------------------------------------------------
# Resolve an MPI launcher that matches the active environment
# ---------------------------------------------------------------------------
MPI_RUNNER=""
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/mpirun" ]]; then
    MPI_RUNNER="${CONDA_PREFIX}/bin/mpirun"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/mpiexec" ]]; then
    MPI_RUNNER="${CONDA_PREFIX}/bin/mpiexec"
elif command -v mpirun &>/dev/null; then
    MPI_RUNNER="$(command -v mpirun)"
elif command -v mpiexec &>/dev/null; then
    MPI_RUNNER="$(command -v mpiexec)"
else
    echo "ERROR: no MPI launcher found. Activate the matching conda environment or install MPI." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Set up a fresh run directory
# ---------------------------------------------------------------------------
echo "Setting up run directory: ${RUN_DIR}"
mkdir -p "${RUN_DIR}"

# Remove only known AxiSEM3D output/log directories to avoid wiping
# anything the user may have placed alongside them.
for stale in output/ logs/; do
    if [[ -d "${RUN_DIR}/${stale}" ]]; then
        echo "  Removing stale: ${RUN_DIR}/${stale}"
        rm -rf "${RUN_DIR:?}/${stale}"
    fi
done

# Deploy binary and input from the bundle (inside this demo folder)
cp "${BUNDLE_BINARY}" "${RUN_DIR}/axisem3d"
chmod +x "${RUN_DIR}/axisem3d"

rm -rf "${RUN_DIR}/input"
cp -r "${BUNDLE_INPUT}" "${RUN_DIR}/input"

echo "Bundle deployed to run directory."

# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------
echo ""
echo "Running AxiSEM3D with ${NRANKS} MPI rank(s) ..."
echo "  Working directory: ${RUN_DIR}"
echo "  MPI launcher     : ${MPI_RUNNER}"
echo ""

(
    cd "${RUN_DIR}"
    "${MPI_RUNNER}" -n "${NRANKS}" ./axisem3d
)

# ---------------------------------------------------------------------------
# Verify outputs and report
# ---------------------------------------------------------------------------
STATION_OUT="${RUN_DIR}/output/stations/GSN_Station_Grid"
ELEMENT_OUT="${RUN_DIR}/output/elements/mantle"

if [[ ! -d "${STATION_OUT}" ]]; then
    echo "ERROR: expected station output was not created: ${STATION_OUT}" >&2
    exit 1
fi
if [[ ! -d "${ELEMENT_OUT}" ]]; then
    echo "ERROR: expected element output was not created: ${ELEMENT_OUT}" >&2
    exit 1
fi

echo ""
echo "=========================================="
echo " Simulation complete"
echo "=========================================="
echo ""
echo "Station output:"
echo "  ${STATION_OUT}"
echo ""
echo "Element output:"
echo "  ${ELEMENT_OUT}"
echo ""
echo "------------------------------------------"
echo " Next step: open the notebook in examples/ and point it at:"
echo "   ${RUN_DIR}"
echo "------------------------------------------"
