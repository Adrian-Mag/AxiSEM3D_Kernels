#!/usr/bin/env bash
# run_handlers_demo.sh — Thin wrapper that delegates to handlers_demo/run_demo.sh.
#
# The demo is self-contained inside examples/handlers_demo/.  This wrapper
# exists purely for discoverability from the examples/ root.
#
# Usage:
#   ./examples/run_handlers_demo.sh [--help] [--dry-run]
#
# Environment variables:
#   NRANKS   Number of MPI ranks  (default: 2)
#
# Example:
#   NRANKS=4 ./examples/run_handlers_demo.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/handlers_demo/run_demo.sh" "$@"
