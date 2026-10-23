#!/usr/bin/env bash
# Build qemu-arm with TCG plugin support (distribution packages ship without it) and the
# ARM926 cycle-model plugin. Output: $OUT/qemu-arm and $OUT/arm926.so
set -euo pipefail
OUT=${OUT:-$HOME/.cache/nslm-perf}
VER=9.2.4
mkdir -p "$OUT"
if [ ! -x "$OUT/qemu-arm" ]; then
  src=$(mktemp -d)
  curl -sSfL "https://download.qemu.org/qemu-$VER.tar.xz" | tar xJ -C "$src"
  (cd "$src/qemu-$VER" && mkdir b && cd b &&
   ../configure --target-list=arm-linux-user --enable-plugins --disable-docs --disable-tools --disable-werror >/dev/null &&
   ninja qemu-arm >/dev/null && cp qemu-arm "$OUT/")
  mkdir -p "$OUT/include" && cp "$src/qemu-$VER/include/qemu/qemu-plugin.h" "$OUT/include/"
  rm -rf "$src"
fi
gcc -O2 -shared -fPIC -Wall -Wextra -Werror -I"$OUT/include" $(pkg-config --cflags glib-2.0) \
    "$(dirname "$0")/arm926.c" -o "$OUT/arm926.so"
echo "QEMU_PLUGIN_QEMU=$OUT/qemu-arm ARM926_SO=$OUT/arm926.so"
