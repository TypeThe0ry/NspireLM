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
  local b=build/$name$variant ckpt=${CKPT:-$MODELS_DIR/$name.bin} tokf=${TOKF:-$MODELS_DIR/$tok.bin}
  python3 compiler/nslmc.py "$ckpt" "$tokf" --out "$b" "$@"
  make -s MODEL="$b" host arm
  python3 tests/quality.py "$ckpt" "$tokf" "$b" "${QVALID:-$VALID}" \
    --stories "$stories" --min-top1 ${MIN_TOP1:-0.96} --max-kl ${MAX_KL:-0.005} --max-ppl-ratio ${MAX_PPL:-1.01} \
    $(printf '%s\n' "$@" | grep -A1 -- --vocab-usage | tr '\n' ' ') 2>/dev/null
  # bit-exactness: host and ARMv5 builds must produce identical logits (model-vocab ids from the C encoder)
  story=${STORY:-$(python3 -c 'import sys; print(open(sys.argv[1], encoding="utf-8").read().split("<|endoftext|>")[1].strip())' "$VALID")}
  "$b/nslm_host" "$b/model.nslm" "$b/tok.nslt" --encode "$story" |
    python3 -c 'import sys, numpy as np; np.asarray(sys.stdin.read().split()[:48], np.int32).tofile(sys.argv[1])' "$b/bitexact.i32"
  "$b/nslm_host" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/host.i32" 2>/dev/null
  $QEMU "$b/nslm_arm" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/arm.i32" 2>/dev/null
  cmp "$b/host.i32" "$b/arm.i32"
  $QEMU "$b/nslm_armc" "$b/model.nslm" "$b/tok.nslt" --eval "$b/bitexact.i32" "$b/armc.i32" 2>/dev/null
  cmp "$b/host.i32" "$b/armc.i32"
  echo "$name$variant: host == armv5 asm == armv5 C (bit-exact)"
  # the app's on-device self-test, run headless on the host and on ARMv5 under qemu
  make -s MODEL="$b" app-host
  for run in "" "$QEMU"; do
    exe="$b/nslm_app"; [ -n "$run" ] && exe="$b/nslm_app_arm"
    NSLM_ABORT_AFTER=${ABORT_AFTER:-100000} NSLM_DIR="$b" NSLM_OUT="$b" NSLM_KEYS="Once upon{ENTER}{MENU}" $run "$exe" >/dev/null
    grep -q "BITEXACT" "$b/nslm_selftest.txt" || { cat "$b/nslm_selftest.txt"; exit 1; }
  done
  echo "$name$variant: app self-test BITEXACT on host and armv5"
  # undefined behaviour would let a different compiler (the Ndless gcc) produce different integers
  ${CC:-cc} -O1 -g -fsanitize=undefined,address -fno-sanitize-recover=all -std=c11 -D_POSIX_C_SOURCE=200112L \
    -Iruntime -I"$b" -Iapp runtime/nslm.c runtime/nslm_kern.c runtime/nslm_tok.c app/app.c app/font8x8.c \
    app/plat_host.c -o "$b/nslm_app_san"
  NSLM_ABORT_AFTER=${ABORT_AFTER:-100000} NSLM_DIR="$b" NSLM_OUT="$b" NSLM_KEYS="Once upon a time{ENTER}{TAB}Hello wörld 你好{ENTER}{MENU}" "$b/nslm_app_san" >/dev/null
  grep -q "BITEXACT" "$b/nslm_selftest.txt"
  echo "$name$variant: UBSan + ASan clean"
}

if [ "${ONLY_CHAT:-0}" != 1 ]; then
run_model stories260K tok512 8
run_model stories15M tokenizer 4
run_model stories15M tokenizer 4 -v12k --vocab-usage data/tinystories_usage.npy
# all-int4 (round-to-nearest, no calibration): covers the int4 kernels, int4 embedding and classifier parts
MIN_TOP1=0.85 MAX_KL=0.15 MAX_PPL=1.15 run_model stories15M tokenizer 4 -q4 --vocab-usage data/tinystories_usage.npy \
  --bits-att 4 --bits-ffn 4 --bits-emb 4
fi
# chat model (byte-level BPE, GQA, RoPE 1e5, learned-scale int4): an untrained 9-layer student from the
# SmolLM2 teacher (train/make_test_student.py) exercises the whole path; needs torch + transformers
if [ "${CHAT:-1}" = 1 ] && python3 -c "import torch, transformers" 2>/dev/null; then
  TJ=$(python3 -c "from huggingface_hub import hf_hub_download as h; print(h('HuggingFaceTB/SmolLM2-135M-Instruct', 'tokenizer.json'))")
  NSLM_BLTOK_JSON=$TJ python3 tests/test_bltok.py 2>/dev/null
  c=build/chat-test
  python3 train/make_test_student.py --layers 0,4,8,11,15,18,22,26,29 --out $c/student.pt 2>/dev/null
  python3 train/export.py $c/student.pt $c/student.bin 2>/dev/null
  CKPT=$c/student.bin TOKF=$TJ QVALID=smoltalk STORY="What is the capital of France? Paris is the capital." \
    ABORT_AFTER=60 MIN_TOP1=0.80 MAX_KL=0.06 MAX_PPL=1.05 run_model chat-test x 3 "" --bits-att 4 --bits-ffn 4 --bits-emb 4
  # the C byte-level encoder (sanitized) against the Python reference, pruned vocabulary
  ${CC:-cc} -O1 -g -fsanitize=undefined,address -fno-sanitize-recover=all -Iruntime -I$c \
    tests/tok_c.c runtime/nslm_tok.c -o $c/tok_c
  NSLM_TOKC=$c/tok_c NSLM_BLTOK_KEEP=train/artifacts/keep.npy NSLM_BLTOK_JSON=$TJ python3 tests/test_bltok.py 2>/dev/null | tail -1
fi
echo "ALL CHECKS PASSED"
