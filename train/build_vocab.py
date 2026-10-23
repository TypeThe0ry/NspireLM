"""Count teacher-token usage over the chat training data and keep the --n most used (plus the 256 byte
symbols and the special tokens).

    python3 train/build_vocab.py --n 16384 --chats 40000 --out build/distill/keep.npy
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import TEACHER, Vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--chats", type=int, default=40000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pandas as pd
    tok = AutoTokenizer.from_pretrained(a.teacher)
    tj = hf_hub_download(a.teacher, "tokenizer.json")
    d = pd.read_parquet(hf_hub_download("HuggingFaceTB/smol-smoltalk", "data/train-00000-of-00004.parquet",
                                        repo_type="dataset"))
    cnt = np.zeros(len(tok), np.int64)
    texts = [x["content"] for i in range(min(a.chats, len(d))) for x in d.iloc[i]["messages"]]
    for i in range(0, len(texts), 512):
        for ids in tok(texts[i:i + 512], add_special_tokens=False).input_ids:
            np.add.at(cnt, ids, 1)
    v = Vocab.build(tj, cnt, a.n)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    v.save(a.out)
    tot = cnt.sum()
    direct = cnt[v.keep].sum() / tot
    # token count growth from re-segmentation
    grown = sum(len(v.pieces(t)) * c for t, c in enumerate(cnt) if c) / tot
    print(f"kept {len(v.keep)}: {direct:.2%} of tokens directly, sequence length x{grown:.3f}")


if __name__ == "__main__":
    main()
