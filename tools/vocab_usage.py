#!/usr/bin/env python3
"""Token usage over a corpus (sentencepiece, same model as llama2.c's tokenizer.bin).
   python3 tools/vocab_usage.py tokenizer.model CORPUS.txt OUT.npy [--max-mb N]"""
import sys, numpy as np, sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
counts = np.zeros(sp.get_piece_size(), np.int64)
limit = int(sys.argv[4]) << 20 if len(sys.argv) > 4 else 1 << 62
done = 0
with open(sys.argv[2], encoding="utf-8", errors="ignore") as f:
    buf = []
    for line in f:
        buf.append(line)
        done += len(line)
        if len(buf) >= 20000 or done >= limit:
            for ids in sp.encode(buf):
                np.add.at(counts, ids, 1)
            buf = []
            if done >= limit:
                break
    if buf:
        for ids in sp.encode(buf):
            np.add.at(counts, ids, 1)
np.save(sys.argv[3], counts)
used = (counts > 0).sum()
print(f"{done / 2**20:.0f} MB: {counts.sum()} tokens, {used} distinct of {len(counts)}")
for th in (1, 2, 5, 10, 100):
    print(f"  tokens seen >= {th:3d} times: {(counts >= th).sum()}")
