#!/usr/bin/env bash
# Full gate: compile models, build host + ARMv5 (qemu) runtimes, check quality and bit-exactness.
# Env: MODELS_DIR (checkpoints + tokenizers), VALID (TinyStories-valid.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS_DIR=${MODELS_DIR:-$HOME/models}
VALID=${VALID:-$HOME/data/TinyStories-valid.txt}
QEMU=${QEMU:-qemu-arm -cpu arm926}

run_model() {
  local name=$1 tok=$2 stories=$3 variant=${4:-} ; shift 3; shift || true
  local b=build/$name$variant
  python3 compiler/nslmc.py "$MODELS_DIR/$name.bin" "$MODELS_DIR/$tok.bin" --out "$b" "$@"
  make -s MODEL="$b" host arm
  python3 tests/quality.py "$MODELS_DIR/$name.bin" "$MODELS_DIR/$tok.bin" "$b" "$VALID" \
    --stories "$stories" --min-top1 0.96 --max-kl 0.005 --max-ppl-ratio 1.01 "$@" 2>/dev/null
  # bit-exactness: host and ARMv5 builds must produce identical logits (model-vocab ids from the C encoder)
  story=$(python3 -c 'import sys; print(open(sys.argv[1], encoding="utf-8").read().split("<|endoftext|>")[1].strip())' "$VALID")
  "$b/nslm_host" "$b/model.nslm" "$b/tok.nslt" --encode "$story" |
    python3 -c 'import sys, numpy as np; np.asarray(sys.stdin.read().split()[:48], np.int32).tofile(sys.argv[1])' "$b/bitexact.i32"
  "$b/nslm_host" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/host.i32" 2>/dev/null
  $QEMU "$b/nslm_arm" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/arm.i32" 2>/dev/null
  cmp "$b/host.i32" "$b/arm.i32"
  $QEMU "$b/nslm_armc" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/armc.i32" 2>/dev/null
  cmp "$b/host.i32" "$b/armc.i32"
  echo "$name$variant: host == armv5 asm == armv5 C (bit-exact)"
}

run_model stories260K tok512 8
run_model stories15M tokenizer 4
run_model stories15M tokenizer 4 -v12k --vocab-usage data/tinystories_usage.npy
echo "ALL CHECKS PASSED"
