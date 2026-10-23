#!/usr/bin/env python3
"""Back-of-envelope ceilings for NspireLM (see docs/feasibility-report.md, section 3).

Every input is either MEASURED (cited in the report) or an ASSUMPTION (marked).
Run: python3 tools/estimate.py
"""
MiB = 1024 * 1024

# ---- measured inputs (ChatTLM, TI-Nspire CX II CAS, OS 6.4.0.74, battery, 396 MHz) ----
CLK = 396e6
HEAP_SINGLE = 22_576_128          # largest single malloc right after reset
HEAP_CHUNKED = 28_442_624         # total in 64 KB chunks
DRAM_SEQ = 70.5e6                 # B/s sequential read (clock of that run disputed, see report)
NAND_SEQ = 1.46e6                 # B/s sequential file read
NAND_RAND_4K = 0.058              # s per random 4 KB read
CT_FIXED = 0.032338               # s per token fixed cost (width fit, 6 layers)
CT_PER_M = 0.037474               # s per token per million params (int8 C engine, ChatTLM)

# ---- assumptions ----
RESERVE = 3 * MiB                 # activations, KV (int8), tokenizer, UI, stack, safety
OPT_CYC_PER_MAC = {"int8": 2.5, "int4": 3.0, "ternary": 2.0}   # hand-written ARM asm, ASSUMED
BYTES_PER_W = {"int8": 1.0 + 4/64, "int4": 0.5 + 2/64, "ternary": 0.25 + 2/256}  # incl. group scales

def capacity(budget):
    return {q: (budget - RESERVE) / b / 1e6 for q, b in BYTES_PER_W.items()}

print("== 1. RAM capacity (M params) ==")
for name, b in (("single block", HEAP_SINGLE), ("chunked heap", HEAP_CHUNKED)):
    print(f"  {name:13s}", {k: round(v, 1) for k, v in capacity(b).items()})

print("\n== 2. Per-token time (dense, params all touched once per token) ==")
def t_conservative(Mp):           # ChatTLM-calibrated C engine, int8
    return CT_FIXED + CT_PER_M * Mp
def t_realistic(Mp):              # ASSUMED 2x over ChatTLM's per-param cost (asm int8 matmul, int8 KV)
    return CT_FIXED + CT_PER_M / 2 * Mp
def t_optimistic(Mp, q):          # max(compute, DRAM streaming) + fixed
    comp = Mp * 1e6 * OPT_CYC_PER_MAC[q] / CLK
    mem = Mp * 1e6 * BYTES_PER_W[q] / DRAM_SEQ
    return CT_FIXED + max(comp, mem), comp, mem
for Mp in (0.26, 1, 3, 7, 11, 15, 20, 25, 45, 90):
    row = f"  {Mp:6.2f}M  C-int8 {1/t_conservative(Mp):6.2f} | 2x-int8 {1/t_realistic(Mp):6.2f} |"
    for q in ("int8", "int4", "ternary"):
        t, c, m = t_optimistic(Mp, q)
        row += f" asm-{q} {1/t:6.2f} ({'mem' if m > c else 'cpu'}) |"
    print(row)

print("\n  largest model at >=0.5 tok/s (budget 2.0 s):")
print(f"    C-int8 (ChatTLM-calibrated): {(2.0-CT_FIXED)/CT_PER_M:.0f}M (but RAM caps int8 at ~{capacity(HEAP_CHUNKED)['int8']:.0f}M)")
print(f"    2x-int8 (realistic target):  {(2.0-CT_FIXED)/(CT_PER_M/2):.0f}M")
for q in ("int8", "int4", "ternary"):
    per_m = max(1e6*OPT_CYC_PER_MAC[q]/CLK, 1e6*BYTES_PER_W[q]/DRAM_SEQ)
    print(f"    asm-{q}: {(2.0-CT_FIXED)/per_m:.0f}M")

print("\n== 3. Streaming weights from NAND ==")
for q in ("int8", "int4", "ternary"):
    per_tok_M = 2.0 * NAND_SEQ / BYTES_PER_W[q] / 1e6
    print(f"  {q:7s}: at 0.5 tok/s only {per_tok_M:.2f}M streamed params/token; "
          f"90 MB file = {90e6/BYTES_PER_W[q]/1e6:.0f}M params -> {90e6/NAND_SEQ:.0f} s/token")

print("\n== 4. PLE (per-layer embedding) table ==")
for L, d_ple in ((6, 128), (8, 128), (8, 256)):
    row_b = L * d_ple * 0.5 + L * 2       # int4 + one fp16 scale per layer
    print(f"  L={L} d_ple={d_ple}: row {row_b:.0f} B; one contiguous row per token")
    for V, where in ((8192, "RAM"), (32768, "RAM"), (32768, "NAND"), (65536, "NAND")):
        size = V * row_b
        params = V * L * d_ple / 1e6
        t = 0 if where == "RAM" else NAND_RAND_4K
        print(f"     V={V:6d} in {where:4s}: {size/MiB:6.1f} MiB, {params:5.1f}M params, +{t*1000:.0f} ms/token")

print("\n== 5. KV cache, vocab ==")
def kv(L, ctx, d, b): return 2 * L * ctx * d * b
for (L, d) in ((6, 352), (8, 384), (8, 512), (12, 512)):
    print(f"  L={L:2d} d={d}: " + ", ".join(
        f"ctx{c} int8 {kv(L,c,d,1)/MiB:.2f} / fp32 {kv(L,c,d,4)/MiB:.1f} MiB" for c in (256, 512, 1024)))
for V, d in ((128, 384), (2048, 384), (4096, 384), (8192, 512), (32000, 288)):
    print(f"  vocab {V:5d} x d{d}: {V*d/1e6:.2f}M params, classifier {V*d/1e6:.2f}M MAC/token")
