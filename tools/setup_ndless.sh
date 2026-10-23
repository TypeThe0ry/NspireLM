#!/usr/bin/env bash
# Fetch the Ndless SDK at the pinned commit, build its toolchain (once, ~1 h) and its libraries.
#   NDLESS_DIR (default ~/ndless) receives the checkout; prints the NDLESS= value for make.
set -euo pipefail
NDLESS_DIR=${NDLESS_DIR:-$HOME/ndless}
REV=9484d8da7c7a4dde9766138c2e42e1d1e3acfcd4
if [ ! -d "$NDLESS_DIR/.git" ]; then
  # the directory may already exist: CI restores the cached toolchain into it before this runs
  mkdir -p "$NDLESS_DIR"
  git -C "$NDLESS_DIR" init -q
  git -C "$NDLESS_DIR" remote add origin https://github.com/ndless-nspire/Ndless.git
fi
git -C "$NDLESS_DIR" fetch -q origin "$REV"
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
