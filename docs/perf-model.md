# ARM926 cycle model (no calculator required)

`perf/arm926.c` is a qemu TCG plugin. It estimates cycle counts on the TI-Nspire CX II's ARM926EJ-S at
396 MHz, so kernels can be designed and checked before anything runs on a real device.

## What it models

| Part | Model | Source |
|---|---|---|
| Instruction issue | Cycles per instruction class: data-processing 1 (2 with a register shift), LDR 1, LDM n, MUL/MLA 2, SMLAxy 1, SMLAL 3, B 3 if taken | ARM9EJ-S TRM DDI0222B table 8-2 |
| Interlocks | Word-load result used by the next instruction +1; byte/halfword load +2 (+1 if used two instructions later); multiply result used next +1 (the accumulate-chain case is exempt) | same |
| Caches | I-cache 16 KB and D-cache 8 KB, both 4-way with 32 B lines, round-robin replacement, write-back, read-allocate | CP15 values measured on a CX II (ChatTLM HARDWARE.md A10–A12) |
| Line fill | Critical word after `lc` cycles, then one word every `beat` cycles. The D-cache blocks: any other access waits for the fill | calibrated below |
| DRAM rows | Optional open-row penalty (`prow`). It has little effect on these workloads | — |

## Calibration (`perf/calibrate.sh`, run in CI)

1. **Line-fill cost.** ChatTLM's `bench_mem` reads 16 MB with an 8-word LDM loop. On a battery-powered
   396 MHz CX II it took 17,039,601 ticks of the 99 MHz timer, i.e. 172.1 ms (about 97 MB/s). The same
   tick count appears in all three logged runs. Replaying the loop under the plugin fixes
   `lc + 7·beat ≈ 111` cycles per 32-byte line.
2. **Out-of-sample check.** ChatTLM's whole inference engine (`runq_nspire.c`, pinned commit) is run
   under the plugin at three model shapes. No parameter is fitted to these runs. The results are compared
   with the device's per-token times:

| Workload | Model | Device | Error |
|---|---|---|---|
| DRAM stream, 16 MB | 172.7 ms | 172.1 ms | +0.4% |
| ChatTLM d176, pos 8 | 135.0 ms | 146.7 ms | −8.0% |
| ChatTLM d264, pos 8 | 265.5 ms | 274.5 ms | −3.3% |
| ChatTLM d352, pos 8 | 440.4 ms | 445.2 ms | −1.1% |

A blocking D-cache fits better than hit-under-miss (437 vs 408 ms predicted for d352, against 445 ms
measured). This agrees with ARM926EJ-S having no hit-under-miss.

## Known limits

- **Soft-float code is under-costed by about 1.6×.** ChatTLM's long-context slope (fp32 attention and
  softmax) is predicted 18–27% low at pos 256. NspireLM has no floating point, so this does not affect
  its predictions.
- **The split between `lc` and `beat` is not identified.** Any split with `lc + 7·beat = 111` fits all
  the data above, because ChatTLM's code does not overlap compute with line fills. NspireLM's kernel
  does, and `lc=55, beat=8` is the default. Measured sensitivity for stories15M (int8, pos 8): `lc=27,beat=12` gives
  5.04 tok/s, `lc=55,beat=8` 5.04, `lc=83,beat=4` 4.31, and `lc=104,beat=1` (no overlap possible) 3.88.
  Every one of these settings passes the calibration check. **Quote predictions as a range: 3.9–5.0
  tok/s for stories15M.** One device run of a stride benchmark would settle it.
- TLB misses, write-buffer stalls and OS interrupts are not modelled. Expect real numbers within
  roughly ±15% of the predictions.
