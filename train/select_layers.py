"""Greedy depth pruning of the teacher: repeatedly drop the layer whose removal hurts least (KL to the full
teacher on assistant tokens of held-out chats), down to --keep layers. The result initialises the student.

    python3 train/select_layers.py --keep 9 --out build/distill/layers.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import TEACHER, chat_text  # noqa: E402
from data import load_chats  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--keep", type=int, default=9)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cand", type=int, default=8, help="candidates re-scored per step (best of last step)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    tok = AutoTokenizer.from_pretrained(a.teacher)
    m = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.float32).eval()
    seqs = []
    for c in load_chats("test", a.n, seed=3):
        p = tok(chat_text(c[:1]) + "<|im_start|>assistant\n", add_special_tokens=False).input_ids
        f = tok(chat_text(c), add_special_tokens=False).input_ids[:192]
        seqs.append((torch.tensor(f)[None], len(p)))
    ref = [torch.log_softmax(m(x).logits[0, p - 1:-1].float(), -1) for x, p in seqs]
    all_layers = list(m.model.layers)

    def score(idx):
        m.model.layers = torch.nn.ModuleList([all_layers[i] for i in idx])
        m.config.num_hidden_layers = len(idx)
        kl, n = 0.0, 0
        for (x, p), r in zip(seqs, ref):
            lp = torch.log_softmax(m(x, use_cache=False).logits[0, p - 1:-1].float(), -1)
            kl += (r.exp() * (r - lp)).sum().item()
            n += r.shape[0]
        return kl / n

    cur = list(range(len(all_layers)))
    log = []
    rank = cur[1:-1]  # the first and last layers stay
    while len(cur) > a.keep:
        scored = sorted((score([j for j in cur if j != i]), i) for i in rank[:a.cand if log else None])
        best = scored[0]
        cur.remove(best[1])
        # next step re-scores the layers that were cheapest to drop now, plus those never tried
        rank = [i for _, i in scored[1:]] + [i for i in rank if i not in {j for _, j in scored}]
        log.append({"dropped": best[1], "kl": best[0], "layers": list(cur)})
        print(f"drop {best[1]:2d} -> {len(cur)} layers, KL {best[0]:.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"layers": cur, "log": log}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
