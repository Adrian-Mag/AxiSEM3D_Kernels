#!/usr/bin/env bash
# clean_demo.sh — remove simulation output produced by run_demo.sh
#
# Usage:
#   ./clean_demo.sh            # show what would be removed, ask for confirmation
#   ./clean_demo.sh --dry-run  # show what would be removed, then exit
#   ./clean_demo.sh --yes      # skip confirmation prompt and delete immediately

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
YES=false
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --yes)     YES=true ;;
        *)
            echo "Usage: $0 [--dry-run] [--yes]" >&2
            exit 1
            ;;
    esac
done

# Directories produced by run_demo.sh (and any stale top-level output/)
TARGETS=(
    "${SCRIPT_DIR}/sim_run"
    "${SCRIPT_DIR}/output"
)

# Collect targets that actually exist
FOUND=()
TOTAL_SIZE=0
for target in "${TARGETS[@]}"; do
    if [[ -e "${target}" ]]; then
        size="$(du -sh "${target}" 2>/dev/null | cut -f1)"
        echo "  ${size}  ${target}"
        FOUND+=("${target}")
    fi
done

if [[ ${#FOUND[@]} -eq 0 ]]; then
    echo "Nothing to clean — no simulation output directories found."
    exit 0
fi

if ${DRY_RUN}; then
    echo ""
    echo "(dry-run) No files were removed."
    exit 0
fi

if ! ${YES}; then
    echo ""
    read -r -p "Remove the above directories? [y/N] " reply
    case "${reply}" in
        [yY][eE][sS]|[yY]) ;;
        *)
            echo "Aborted."
            exit 0
            ;;
    esac
fi

for target in "${FOUND[@]}"; do
    rm -rf "${target:?}"
    echo "Removed: ${target}"
done

echo ""
echo "Done. Run ./run_demo.sh to regenerate the simulation output."
