#!/usr/bin/env python3
"""Quality gate: integer runtime vs the float32 reference on held-out TinyStories text.

Reports, over every position of N validation stories:
  top1   argmax agreement between integer and float logits
  kl     mean KL(p_float || p_int) in nats
  ppl    perplexity of the true next token, float and integer
Also checks the C tokenizer against the Python port of llama2.c's tokenizer.

  python3 tests/quality.py CKPT TOKENIZER BUILD_DIR VALID.txt [--stories N] [--bin nslm_host]
           [--runner "qemu-arm -cpu arm926"] [--min-top1 0.97] [--max-kl 0.01]
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "compiler"))
import nslmc  # noqa: E402


def log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("tokenizer")
    ap.add_argument("build")
    ap.add_argument("valid")
    ap.add_argument("--stories", type=int, default=4)
    ap.add_argument("--max-pos", type=int, default=256)
    ap.add_argument("--bin", default=None)
    ap.add_argument("--runner", default="")
    ap.add_argument("--min-top1", type=float, default=0.0)
    ap.add_argument("--max-kl", type=float, default=1e9)
    ap.add_argument("--max-ppl-ratio", type=float, default=1e9)
    ap.add_argument("--vocab-usage", help="same file the model was compiled with (vocab pruning)")
    ap.add_argument("--min-count", type=int, default=1)
    a = ap.parse_args()

    ck = nslmc.Ckpt(a.ckpt)
    tok = nslmc.Tokenizer(a.tokenizer, ck.vocab)
    if a.vocab_usage:
        counts = np.load(a.vocab_usage)
        tok.set_keep(sorted(set(np.nonzero(counts >= a.min_count)[0].tolist()) | {0, 1, 2} | set(range(3, 259))))
    keep = np.asarray(tok.new2old)
    nv = len(keep)
    exe = shlex.split(a.runner) + [a.bin or os.path.join(a.build, "nslm_host"),
                                    os.path.join(a.build, "model.nslm"), os.path.join(a.build, "tok.nslt")]
    with open(a.valid, encoding="utf-8") as f:
        stories = [s.strip() for s in f.read().split("<|endoftext|>") if s.strip()][: a.stories]

    fm = nslmc.FloatModel(ck)
    tot = dict(n=0, top1=0, kl=0.0, nll_f=0.0, nll_i=0.0)
    tok_mismatch = 0
    for si, story in enumerate(stories):
        ids = tok.encode(story)[: min(a.max_pos, ck.seq)]
        c_ids = [int(x) for x in subprocess.run(exe + ["--encode", story], capture_output=True, text=True,
                                                check=True).stdout.split()][: len(ids)]
        if c_ids != ids:
            tok_mismatch += 1
        with tempfile.TemporaryDirectory() as td:
            tp, lp = os.path.join(td, "t.i32"), os.path.join(td, "l.i32")
            np.asarray(ids, np.int32).tofile(tp)
            subprocess.run(exe + ["--eval", tp, lp], check=True, capture_output=True)
            li = np.fromfile(lp, np.int32).reshape(len(ids), nv).astype(np.float64) / 65536.0
        fm.reset()
        lf = np.stack([fm.forward(int(keep[t]), p) for p, t in enumerate(ids)]).astype(np.float64)
        pf_full = log_softmax(lf)
        pf, pi = log_softmax(lf[:, keep]), log_softmax(li)  # float renormalised over the model vocab
        n = len(ids) - 1
        tgt = np.asarray(ids[1:])
        tot["n"] += n
        tot["top1"] += int((lf[:n].argmax(-1) == keep[li[:n].argmax(-1)]).sum())
        tot["kl"] += float((np.exp(pf[:n]) * (pf[:n] - pi[:n])).sum())
        tot["nll_f"] += float(-pf_full[np.arange(n), keep[tgt]].sum())
        tot["nll_i"] += float(-pi[np.arange(n), tgt].sum())
        tot["outside"] = tot.get("outside", 0.0) + float(1 - np.exp(pf_full[:n][:, keep]).sum(-1).sum() / n) * n
        print(f"story {si}: {len(ids)} tokens", file=sys.stderr)

    n = tot["n"]
    top1, kl = tot["top1"] / n, tot["kl"] / n
    ppl_f, ppl_i = np.exp(tot["nll_f"] / n), np.exp(tot["nll_i"] / n)
    print(f"positions={n} top1={top1:.4f} kl={kl:.5f} ppl_float={ppl_f:.4f} ppl_int={ppl_i:.4f} "
          f"ratio={ppl_i / ppl_f:.4f} tokenizer_mismatch={tok_mismatch} vocab={nv} "
          f"float_mass_outside_vocab={tot.get('outside', 0) / n:.2e}")
    ok = top1 >= a.min_top1 and kl <= a.max_kl and ppl_i / ppl_f <= a.max_ppl_ratio and tok_mismatch == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
