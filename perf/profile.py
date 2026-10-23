#!/usr/bin/env python3
"""Fold a perf/arm926.c block profile into per-function cycles using the ELF symbol table.
   perf/profile.py ELF PROFILE [--nm arm-linux-gnueabi-nm]"""
import bisect, subprocess, sys
elf, prof = sys.argv[1], sys.argv[2]
nm = sys.argv[4] if len(sys.argv) > 4 else "arm-linux-gnueabi-nm"
syms = []
for line in subprocess.run([nm, "-n", elf], capture_output=True, text=True).stdout.splitlines():
    p = line.split()
    if len(p) == 3 and p[1] in "tTwW":
        syms.append((int(p[0], 16), p[2]))
addrs = [a for a, _ in syms]
tot = {}
for line in open(prof):
    a, c = line.split()
    i = bisect.bisect_right(addrs, int(a, 16)) - 1
    name = syms[i][1] if i >= 0 else "?"
    tot[name] = tot.get(name, 0) + int(c)
all_c = sum(tot.values())
for name, c in sorted(tot.items(), key=lambda x: -x[1])[:15]:
    print(f"{c / 396e3:9.2f} ms {100 * c / all_c:5.1f}%  {name}")
print(f"{all_c / 396e3:9.2f} ms total")
