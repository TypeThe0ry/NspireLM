#!/usr/bin/env bash
# Performance gate: predicted tok/s on a 396 MHz CX II must not regress below the floors.
#   perf/check_speed.sh BUILD_DIR MIN_TOKS_POS8
set -euo pipefail
out=$("$(dirname "$0")/simulate.sh" "$1" 255)
echo "$(basename "$1"): $out"
python3 - "$out" "$2" <<'PY'
import re, sys
toks = float(re.search(r"pos8: [\d.]+ ms \(([\d.]+) tok/s", sys.argv[1]).group(1))
sys.exit(0 if toks >= float(sys.argv[2]) else f"predicted {toks} tok/s < floor {sys.argv[2]}")
PY
