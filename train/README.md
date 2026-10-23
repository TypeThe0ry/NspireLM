# Training the calculator chat model

The calculator has about 27 MiB of heap. SmolLM2-135M-Instruct is the smallest model that answers questions
usefully. In int4 it needs about 60 MB, and lower precision breaks it: measured perplexity on chat answers
goes from 2.18 in float to 3.85 at 3 bits, and above 10⁴ at 2 bits. So we **distill** it into a student
that fits:

| | teacher | student |
|---|---|---|
| layers | 30 | 9, copied from the teacher (`select_layers.py`: greedy, least KL first) |
| width / heads | 576, 9 q / 3 kv, head 64 | same, so the copied layers fit |
| vocabulary | 49152 | 16384 most used in chat (97.2% of tokens directly). Other tokens are split along their BPE merges, which adds 4.3% tokens |
| weights | bf16 | int4 with learned group scales (QAT, exact runtime grid) |
| size | 270 MB | ≈ 22 MB on the calculator |

Steps:

1. `build_vocab.py` counts teacher-token usage on smol-smoltalk and keeps the top 16384 → `keep.npy`.
2. `select_layers.py` picks the layers → `layers.json`. CPU, about 30 min. Both outputs are committed in
   `train/artifacts/`, so this step can be skipped.
3. `distill.py` runs on a GPU:
   - **Phase 1:** KL to the teacher's kept-vocabulary distribution on every token, plus 0.1 × CE on
     assistant tokens.
   - **Phase 2 (last 25% of steps):** the same loss with 4-bit quantization-aware training. Each group of
     32 weights shares one learned scale, and the codes are −8..7, the same as `runtime` q4. The exported
     weights therefore lose nothing more when compiled.
4. `export.py` writes the llama2.c layout, the scales and the metadata. It checks itself against the HF model.
5. `compiler/nslmc.py student.bin tokenizer.json --bits-att 4 --bits-ffn 4 --bits-emb 4` produces
   `model.nslm` and `tok.nslt` (byte-level BPE, `compiler/bltok.py`, exact against HF).

## Running on Colab

Open `train/distill_colab.ipynb` in Colab with a GPU runtime and run all cells. It clones the repo,
installs the dependencies and trains. Checkpoints go to Google Drive, so a disconnect resumes where it
stopped. At the end it exports `student.bin*`. Send those files back, or put them in the repo's release,
and the rest (compile, verify, package) happens here.

| GPU | throughput (est.) | default run (12k steps × 16k tokens ≈ 200M tokens) |
|---|---|---|
| A100 | ~120k tok/s | ~30 min |
| L4 | ~45k tok/s | ~1.3 h |
| T4 (free tier) | ~15k tok/s | ~4 h (use `--steps 6000` for ~2 h) |
