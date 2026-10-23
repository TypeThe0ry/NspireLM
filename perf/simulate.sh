#!/usr/bin/env bash
# Predict per-token time on a 396 MHz CX II with the calibrated ARM926 model (docs/perf-model.md).
#   perf/simulate.sh BUILD_DIR [FAR_POS]
# Needs QEMU_PLUGIN_QEMU (qemu-arm built with --enable-plugins) and ARM926_SO (perf/arm926.c built).
set -euo pipefail
b=$1; far=${2:-255}
QEMU=${QEMU_PLUGIN_QEMU:-qemu-arm}
SO=${ARM926_SO:-perf/arm926.so}
ARGS=${ARM926_ARGS:-lc=55,beat=8,hum=0,wb=0,prow=30,rowbits=11}
M=$(arm-linux-gnueabi-nm "$b/${BIN:-nslm_arm}" | awk '$3=="nslm_perf_mark"{print "0x"$1}')
$QEMU -cpu arm926 -plugin "$SO,mark=$M,$ARGS" -d plugin "$b/${BIN:-nslm_arm}" "$b/model.nslm" "$b/tok.nslt" --bench "$far" 2>&1 |
awk -v far="$far" '/MARK/{split($3,a,"="); c[$2]=a[2]; split($4,n,"="); ins[$2]=n[2]; split($6,m,"="); dm[$2]=m[2]}
 END{p8=(c[1]-c[0])/396e3; pf=(c[2]-c[1])/396e3;
     printf "pos8: %.1f ms (%.2f tok/s, %.1fM insns, %.0fK dmiss)  pos%d: %.1f ms (%.2f tok/s)\n",
            p8, 1000/p8, (ins[1]-ins[0])/1e6, (dm[1]-dm[0])/1e3, far, pf, 1000/pf}'
