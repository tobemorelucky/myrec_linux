#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-check}" in
 check) exec bash "$ROOT/scripts/gpu_stages.sh" check ;;
 *) echo "Base-SCFT draft superseded. Use separately approved gpu_stages C/D/E after REAL Rich freeze.";exit 2 ;;
esac
