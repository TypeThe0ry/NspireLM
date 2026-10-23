#!/usr/bin/env python3
"""Fake-quantisation study: perplexity of the float model with weights rounded to int8/int4 schemes.
   python3 tools/quant_study.py CKPT TOKENIZER VALID.txt [--stories N]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "compiler"))
import nslmc

def fq(w, bits, gs, clip_search=False):
    r, c = w.shape
    while c % gs:
        gs //= 2
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(r, c // gs, gs).astype(np.float64)
    amax = np.abs(g).max(axis=2, keepdims=True)
    best = None
    ratios = [1.0] if not clip_search else [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    for k in ratios:
        s = np.where(amax == 0, 1e-12, amax * k / qmax)
        q = np.clip(np.round(g / s), -qmax - (1 if bits == 4 else 0), qmax) * s
        err = ((q - g) ** 2).sum(axis=2, keepdims=True)
        if best is None:
            best, berr = q, err
        else:
            m = err < berr
            best = np.where(m, q, best)
            berr = np.where(m, err, berr)
    return best.reshape(r, c).astype(np.float32)

def ppl(ck, tok, stories):
    fm = nslmc.FloatModel(ck)
    nll, n = 0.0, 0
    for st in stories:
        ids = tok.encode(st)[: ck.seq]
        fm.reset()
        for p in range(len(ids) - 1):
            lg = fm.forward(ids[p], p).astype(np.float64)
            lg -= lg.max()
            nll += np.log(np.exp(lg).sum()) - lg[ids[p + 1]]
            n += 1
    return np.exp(nll / n)

def apply(ck0, scheme):
    import copy
    ck = copy.copy(ck0)
    for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        w = getattr(ck0, name)
        bits, gs, cs = scheme.get(name, scheme["default"])
        setattr(ck, name, np.stack([fq(w[l], bits, gs, cs) for l in range(w.shape[0])]))
    bits, gs, cs = scheme.get("emb", scheme["default"])
    ck.tok_emb = fq(ck0.tok_emb, bits, gs, cs)
    ck.wcls = ck.tok_emb if ck0.shared else fq(ck0.wcls, bits, gs, cs)
    return ck

# ---------------------------------------------------------------- GPTQ variants (run with --gptq)
def gptq_apply(ck0, X, scheme):
    import copy, gptq
    ck = copy.copy(ck0)
    src = {"wq": "att_in", "wk": "att_in", "wv": "att_in", "wo": "wo_in", "w1": "ffn_in", "w3": "ffn_in", "w2": "w2_in"}
    for name, xin in src.items():
        w = getattr(ck0, name)
        bits, gs = scheme.get(name, scheme["default"])
        out = []
        for l in range(w.shape[0]):
            g = gs
            while w.shape[2] % g:
                g //= 2
            Q, S = gptq.gptq_quant(w[l], X[f"l{l}_{xin}"], bits, g)
            out.append((Q.reshape(Q.shape[0], -1, g) * S[:, :, None]).reshape(Q.shape))
        setattr(ck, name, np.stack(out).astype(np.float32))
    bits, gs = scheme.get("emb", scheme["default"])
    ck.tok_emb = fq(ck0.tok_emb, bits, gs, True)
    ck.wcls = ck.tok_emb if ck0.shared else fq(ck0.wcls, bits, gs, True)
    return ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tokenizer"); ap.add_argument("valid")
    ap.add_argument("--stories", type=int, default=6)
    ap.add_argument("--gptq", action="store_true")
    ap.add_argument("--calib", type=int, default=24)
    a = ap.parse_args()
    ck = nslmc.Ckpt(a.ckpt)
    tok = nslmc.Tokenizer(a.tokenizer, ck.vocab)
    stories = [s.strip() for s in open(a.valid, encoding="utf-8").read().split("<|endoftext|>") if s.strip()][: a.stories]
    base = ppl(ck, tok, stories)
    print(f"float32                          ppl={base:.4f}")
    if a.gptq:
        import gptq
        allst = [s.strip() for s in open(a.valid, encoding="utf-8").read().split("<|endoftext|>") if s.strip()]
        calib = [tok.encode(s) for s in allst[1000:1000 + a.calib]]  # disjoint from the eval stories
        X = gptq.calibrate(ck, calib)
        print(f"calibration: {len(X['cls_in'])} tokens", flush=True)
        for name, sc in {"GPTQ int4 g32, emb int8": {"default": (4, 32), "emb": (8, 64)},
                         "GPTQ int4 g64, emb int8": {"default": (4, 64), "emb": (8, 64)},
                         "GPTQ int4 g32, emb+attn int8": {"default": (4, 32), "emb": (8, 64), "wq": (8, 64),
                                                         "wk": (8, 64), "wv": (8, 64), "wo": (8, 64)},
                         "GPTQ int8 g64": {"default": (8, 64)}}.items():
            p = ppl(gptq_apply(ck, X, sc), tok, stories)
            print(f"{name:32s} ppl={p:.4f}  (+{100 * (p / base - 1):.2f}%)", flush=True)
        sys.exit(0)
    schemes = {
        "int8 g64": {"default": (8, 64, False)},
        "int4 g64": {"default": (4, 64, False)},
        "int4 g32": {"default": (4, 32, False)},
        "int4 g32 clip": {"default": (4, 32, True)},
        "int4 g16 clip": {"default": (4, 16, True)},
        "int4 g32 clip, emb int8": {"default": (4, 32, True), "emb": (8, 64, False)},
        "int4 g32 clip, emb+attn int8": {"default": (4, 32, True), "emb": (8, 64, False),
                                         "wq": (8, 64, False), "wk": (8, 64, False), "wv": (8, 64, False), "wo": (8, 64, False)},
    }
    for name, sc in schemes.items():
        p = ppl(apply(ck, sc), tok, stories)
        print(f"{name:32s} ppl={p:.4f}  (+{100 * (p / base - 1):.2f}%)", flush=True)
