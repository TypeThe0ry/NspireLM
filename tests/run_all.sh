#!/usr/bin/env bash
# Full gate: compile models, build host + ARMv5 (qemu) runtimes, check quality and bit-exactness.
# Env: MODELS_DIR (checkpoints + tokenizers), VALID (TinyStories-valid.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS_DIR=${MODELS_DIR:-$HOME/models}
VALID=${VALID:-$HOME/data/TinyStories-valid.txt}
QEMU=${QEMU:-qemu-arm -cpu arm926}

run_model() {
  local name=$1 tok=$2 stories=$3
  local b=build/$name
  python3 compiler/nslmc.py "$MODELS_DIR/$name.bin" "$MODELS_DIR/$tok.bin" --out "$b"
  make -s MODEL="$b" host arm
  python3 tests/quality.py "$MODELS_DIR/$name.bin" "$MODELS_DIR/$tok.bin" "$b" "$VALID" \
    --stories "$stories" --min-top1 0.96 --max-kl 0.005 --max-ppl-ratio 1.01 2>/dev/null
  # bit-exactness: host and ARMv5 builds must produce identical logits
  python3 - "$MODELS_DIR/$tok.bin" "$b" "$VALID" <<'PY'
import sys, numpy as np
sys.path.insert(0, "compiler"); import nslmc
tokpath, b, valid = sys.argv[1:]
import re
vocab = int(re.search(r"#define NSLM_VOCAB (\d+)", open(f"{b}/nslm_model.h").read()).group(1))
t = nslmc.Tokenizer(tokpath, vocab)
story = open(valid, encoding="utf-8").read().split("<|endoftext|>")[1].strip()
np.asarray(t.encode(story)[:48], np.int32).tofile(f"{b}/bitexact.i32")
PY
  "$b/nslm_host" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/host.i32" 2>/dev/null
  $QEMU "$b/nslm_arm" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/arm.i32" 2>/dev/null
  cmp "$b/host.i32" "$b/arm.i32"
  $QEMU "$b/nslm_armc" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/armc.i32" 2>/dev/null
  cmp "$b/host.i32" "$b/armc.i32"
  echo "$name: host == armv5 asm == armv5 C (bit-exact)"
}

run_model stories260K tok512 8
run_model stories15M tokenizer 4
echo "ALL CHECKS PASSED"
