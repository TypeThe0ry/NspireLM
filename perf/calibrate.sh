#!/usr/bin/env bash
# Re-derive and check the cycle model's calibration (docs/perf-model.md).
#   1. ChatTLM's DRAM stream benchmark: device ticks fix the linefill cost (lc + 7*beat).
#   2. ChatTLM's whole engine at three shapes: an out-of-sample check of the model at pos 8.
# Device numbers are ChatTLM's own logs (results/device_g88p/bench_results.txt, 396 MHz, battery).
set -euo pipefail
cd "$(dirname "$0")/.."
QEMU=${QEMU_PLUGIN_QEMU:?}; SO=${ARM926_SO:?}
ARGS=${ARM926_ARGS:-lc=55,beat=8,hum=0,wb=0,prow=30,rowbits=11}
W=build/calib; mkdir -p $W
TLM=$W/nspire-tlm
if [ ! -d $TLM ]; then
  git clone -q https://github.com/alex-messerlian/nspire-tlm $TLM
  git -C $TLM checkout -q 16d129324921d87278e87c894f705c1a3a4dee65
fi
CC="arm-linux-gnueabi-gcc -O2 -marm -mcpu=arm926ej-s -static"
$CC perf/calib/stream.c -o $W/stream
$CC -std=gnu99 -w -DFIXED_GS=88 -DTLM_HOST -DCHATTLM_SRC="\"$PWD/$TLM/src/runq_nspire.c\"" \
    -I$TLM/src/store -I$TLM/tools/eval perf/calib/chattlm_driver.c -o $W/drv -lm
for d in 176 352; do [ -f $W/m$d.bin ] || python3 $TLM/tools/make_shape.py --dim $d --out $W/m$d.bin >/dev/null; done
[ -f $W/m264.bin ] || python3 $TLM/tools/make_shape.py --dim 264 --heads 6 --out $W/m264.bin >/dev/null

marks() { # elf args... -> "c0 c1 c2" cycles at the marks
  local elf=$1; shift
  local m; m=$(arm-linux-gnueabi-nm "$elf" | awk '$3=="perf_mark"{print "0x"$1}')
  $QEMU -cpu arm926 -plugin "$SO,mark=$m,$ARGS" -d plugin "$elf" "$@" 2>&1 |
    awk '/^MARK/{split($3,a,"="); printf "%s ", a[2]}'
}
fail=0
check() { # name predicted_ms device_ms tolerance
  python3 - "$@" <<'PY' || fail=1
import sys
n, p, d, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
err = (p - d) / d
print(f"{n:22s} model {p:8.1f}   device {d:8.1f}   error {100*err:+5.1f}%   (tolerance {100*tol:.0f}%)")
sys.exit(0 if abs(err) <= tol else 1)
PY
}
read c0 c1 c2 <<<"$(marks $W/stream)"
check "DRAM stream 16 MB (ms)" "$(echo "($c1-$c0)/396000" | bc -l)" "$(echo "17039601/99000" | bc -l)" 0.03
for sd in "176 146.728" "264 274.505" "352 445.220"; do
  set -- $sd
  read c0 c1 c2 <<<"$(marks $W/drv $W/m$1.bin)"
  check "ChatTLM m$1 pos8 (ms)" "$(echo "($c1-$c0)/396000" | bc -l)" "$2" 0.12
done
[ $fail = 0 ] && echo "CALIBRATION OK" || { echo "CALIBRATION FAILED"; exit 1; }
