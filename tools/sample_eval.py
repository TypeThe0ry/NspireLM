#!/usr/bin/env python3
"""Choose a decoding policy by measurement, not taste.

For each policy the integer runtime continues N validation-story openings. Each continuation is scored:
  judge_ppl  perplexity of the continuation under a larger float model (stories42M): coherence
  rep4       fraction of repeated 4-grams within a story: loops
  distinct2  distinct bigrams / bigrams across all stories: diversity
  len        mean continuation length (an early end-of-story token stops it)

  python3 tools/sample_eval.py BUILD_DIR TOKENIZER JUDGE_CKPT VALID.txt [--n 40] [--steps 160]
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "compiler"))
import nslmc

LN2 = 0.6931471805599453
POLICIES = [  # name, temp_q8, minp (probability, 0 = off), rep (logit units), window
    ("greedy", 0, 0, 0.0, 64),
    ("greedy+rep1.0", 0, 0, 1.0, 64),
    ("T1.0", 256, 0, 0.0, 64),
    ("T0.8", 205, 0, 0.0, 64),
    ("T0.8 minp.05", 205, 0.05, 0.0, 64),
    ("T1.0 minp.10", 256, 0.10, 0.0, 64),
    ("T0.8 minp.05 rep0.5", 205, 0.05, 0.5, 64),
    ("T0.8 minp.05 rep1.0", 205, 0.05, 1.0, 64),
    ("T0.6 minp.05 rep0.5", 154, 0.05, 0.5, 64),
]


def log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build"); ap.add_argument("tokenizer"); ap.add_argument("judge"); ap.add_argument("valid")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--vocab-usage", default=os.path.join(os.path.dirname(__file__), "..", "data", "tinystories_usage.npy"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    hdr = open(os.path.join(a.build, "nslm_model.h")).read()
    full = int(re.search(r"NSLM_VOCAB_FULL (\d+)", hdr).group(1))
    nv = int(re.search(r"NSLM_VOCAB (\d+)", hdr).group(1))
    tok = nslmc.Tokenizer(a.tokenizer, full)
    if nv != full:
        c = np.load(a.vocab_usage)
        tok.set_keep(sorted(set(np.nonzero(c >= 1)[0].tolist()) | {0, 1, 2} | set(range(3, 259))))
    judge = nslmc.Ckpt(a.judge)
    jm = nslmc.FloatModel(judge)
    stories = [s.strip() for s in open(a.valid, encoding="utf-8").read().split("<|endoftext|>") if s.strip()]
    prompts = [" ".join(s.split()[:8]) for s in stories[2000:2000 + a.n]]
    exe = [os.path.join(a.build, "nslm_host"), os.path.join(a.build, "model.nslm"), os.path.join(a.build, "tok.nslt")]
    for name, t, minp, rep, win in POLICIES:
        if a.only and a.only not in name:
            continue
        with tempfile.TemporaryDirectory() as td:
            pf, of = os.path.join(td, "p.txt"), os.path.join(td, "o.txt")
            open(pf, "w").write("\n".join(prompts) + "\n")
            minp_q16 = int(round(np.log2(minp) * 65536)) if minp else 0
            subprocess.run(exe + ["-t", str(t), "--minp", str(minp_q16), "--rep", str(int(rep * 65536)),
                                  "--window", str(win), "-n", str(a.steps), "--gen", pf, of], check=True)
            outs = [list(map(int, l.split())) for l in open(of).read().splitlines()]
        nll = n = 0
        rep4s, bigrams, total_bi, lens = [], set(), 0, []
        for prompt, gen in zip(prompts, outs):
            ids = [int(tok.new2old[x]) for x in tok.encode(prompt)]
            g = [int(tok.new2old[x]) for x in gen]
            lens.append(len(g))
            jm.reset()
            seq = ids + g
            for p in range(len(seq) - 1):
                lp = jm.forward(seq[p], p)
                if p + 1 >= len(ids):
                    nll -= log_softmax(lp.astype(np.float64))[seq[p + 1]]
                    n += 1
            grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
            rep4s.append(1 - len(set(grams)) / len(grams) if grams else 0)
            bi = [tuple(g[i:i + 2]) for i in range(len(g) - 1)]
            bigrams |= set(bi)
            total_bi += len(bi)
        print(f"{name:24s} judge_ppl={np.exp(nll / max(n, 1)):6.3f} rep4={np.mean(rep4s):.3f} "
              f"distinct2={len(bigrams) / max(total_bi, 1):.3f} len={np.mean(lens):5.1f}", flush=True)


if __name__ == "__main__":
    main()
