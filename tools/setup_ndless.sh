#!/usr/bin/env bash
# Fetch the Ndless SDK at the pinned commit, build its toolchain (once, ~1 h) and its libraries.
#   NDLESS_DIR (default ~/ndless) receives the checkout; prints the NDLESS= value for make.
set -euo pipefail
NDLESS_DIR=${NDLESS_DIR:-$HOME/ndless}
REV=9484d8da7c7a4dde9766138c2e42e1d1e3acfcd4
if [ ! -d "$NDLESS_DIR/.git" ]; then
  git clone -q https://github.com/ndless-nspire/Ndless.git "$NDLESS_DIR"
fi
git -C "$NDLESS_DIR" fetch -q origin "$REV" 2>/dev/null || true
git -C "$NDLESS_DIR" checkout -q "$REV"
git -C "$NDLESS_DIR" submodule update -q --init --recursive
SDK=$NDLESS_DIR/ndless-sdk
if [ ! -x "$SDK/toolchain/install/bin/arm-none-eabi-gcc" ]; then
  (cd "$SDK/toolchain" && PARALLEL=-j"$(nproc)" ./build_toolchain.sh)
fi
export PATH="$SDK/bin:$SDK/toolchain/install/bin:$PATH"
make -s -C "$SDK/thirdparty" all
make -s -C "$SDK/libsyscalls"
make -s -C "$SDK/libndls"
make -s -C "$SDK/tools/genzehn"
make -s -C "$SDK/tools/zehn_loader"
echo "NDLESS=$SDK"
