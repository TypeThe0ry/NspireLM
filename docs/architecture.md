# NspireLM architecture

NspireLM is an **ahead-of-time (AOT) compiled**, **integer-only** Llama inference engine for the
TI-Nspire CX II (ARM926EJ-S, ARMv5TE, no FPU). The compiler makes every decision it can before the
calculator runs. The runtime is a fixed sequence of integer kernels, specialised to one model.

```
checkpoint.bin ─┐                 ┌─ model.nslm    packed int8 weights in kernel layout
tokenizer.bin  ─┼─► nslmc (AOT) ──┼─ nslm_model.h  shapes, offsets, LUTs as C constants
                │                 └─ tok.nslt      integer-rank BPE tokenizer
                └────────────────────────────────► runtime/*.c (+ ARM asm) compiled per model
```

## What the compiler bakes in

| Item | Done at compile time | Runtime cost removed |
|---|---|---|
| Weights | Group-wise symmetric int8. The group size is the largest divisor of `cols` that is ≤ 64 (a multiple of 8 where possible). Scales are stored as a uint16 mantissa per group plus one exponent per row | float scale multiplies |
| Softmax base | `log2(e)/sqrt(head_dim)` is folded into `Wq`, so attention needs only `exp2` | one multiply per score, `expf` |
| RoPE | cos/sin table in Q30, computed from run.c's float32 angles | `powf`, `cosf`, `sinf` per token |
| Nonlinearities | `exp2` table (257 entries, Q30) and sigmoid table (1025 entries over ±16, Q30), with linear interpolation | `expf` |
| Memory | Every tensor offset and every run-state buffer size is a compile-time constant. State is one allocation | allocator use, fragmentation |
| Tokenizer | Float merge scores are replaced by integer ranks, and a pre-sorted index enables binary search | float compares, `qsort` at startup |
| Vocabulary | `--vocab-usage`: tokens that never occur in the training corpus are compiled out of the embedding and classifier. stories15M goes from 32000 to 12750 tokens. The encoder still uses the full BPE and re-segments any compiled-out piece into kept pieces | 60% of classifier MACs, 5.5 MB |

| Mixed precision | `--bits-att/--bits-ffn/--bits-emb 4\|8`. int4 uses the x4n layout (4 rows × 16 columns per cache line). Rounding is GPTQ with `--calib` text, otherwise round-to-nearest with an MSE clipping search | memory, DRAM traffic |
| Segments | The image is split at tensor boundaries into ≤ 1 MiB pieces (`nslm_seg_off[]`), and the embedding/classifier into row parts. Each piece is allocated separately: the Ndless heap gives ~21.5 MiB in one block but ~27 MiB in pieces | the single-block ceiling |

## Number formats (runtime)

| Name | Representation | Used for |
|---|---|---|
| fx32 | int32, Q16 | residual stream, matmul outputs, logits |
| act16 | int16 plus one power-of-two exponent per vector | matmul inputs (RMSNorm output, attention output, SwiGLU output) |
| q8 | int8 weights, uint16 group mantissa, int8 row exponent | all matrices including the (tied) classifier |
| q4 | int4 codes (offset 8), same scale format, x4n layout | selected matrices (`--bits-*`) |
| kv8 | int8 per (position, kv-head), with a Q24 scale (max\|·\|/127) | KV cache: 4× smaller than fp32 |

The int16 activations carry 15 bits of precision, which is more than the int8 activations of llama2.c's `runq` or
ChatTLM. Matmul partial sums per group fit in int32 (gs·127·2^15 < 2^31). Groups are combined in int64.

## Verification

`tests/run_all.sh` (also the CI gate) checks two things:

1. **Quality against float32.** `tests/quality.py` runs the integer engine and a float32 NumPy reference
   (the same math as `run.c`) over TinyStories validation stories. It reports top-1 agreement, KL divergence
   and perplexity, and fails below the thresholds. It also checks the C tokenizer against the Python port.
2. **Bit-exactness across targets.** The host build and the ARMv5TE build (`-mcpu=arm926ej-s`, run under
   `qemu-arm -cpu arm926`) must produce byte-identical logits.

Current results (2026-10-23):

| Model | positions | top-1 agree | KL (nats) | PPL float → int |
|---|---|---|---|---|
| stories260K | 1922 | 97.6% | 0.0019 | 3.218 → 3.216 |
| stories15M | 702 | 98.4% | 0.0006 | 2.364 → 2.362 |
| stories15M, vocab 12750 | 702 | 98.4% | 0.0006 | 2.364 → 2.362 |

Greedy generations of both models match llama2.c `run.c` (float32) token for token over the first 64 tokens of
"Once upon a time".

## The calculator app (`app/`)

`app.c` is platform independent: a prompt editor, streamed generation, and an on-device self-test.
`plat_nspire.c` is the Ndless layer: LCD blit, key scanning, the SP804 timer, PMU clock decode, and
`fopen` with 64 KB reads. `plat_host.c` runs the same app headless, with scripted keys, and saves frames
as images.

The self-test hashes the logits of a fixed prompt and compares them with a hash produced at build time by
the host build. The gate runs the app on the host and on ARMv5 under qemu, and both must report
`BITEXACT`. On the calculator, the same report also includes the CPU clock, ms/token, the free heap, and a
sequential-vs-stride memory benchmark. That benchmark measures the one open parameter of the cycle model
(`docs/perf-model.md`).
