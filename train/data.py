"""Chat data: HuggingFaceTB/smol-smoltalk (the SFT mix SmolLM2-135M-Instruct itself was tuned on).

Conversations with a system turn are dropped (the device prompt has none). load_chats() returns lists of
{"role", "content"} messages; packed_batches() yields fixed-length token blocks for training.
"""
import random

import numpy as np

REPO = "HuggingFaceTB/smol-smoltalk"
FILES = {"test": ["data/test-00000-of-00001.parquet"],
         "train": [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]}


def _frames(split):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for f in FILES[split]:
        yield pd.read_parquet(hf_hub_download(REPO, f, repo_type="dataset"))


def load_chats(split, n=None, seed=0, max_chars=1200):
    out = []
    for d in _frames(split):
        idx = list(range(len(d)))
        random.Random(seed).shuffle(idx)
        for i in idx:
            msgs = [dict(role=x["role"], content=x["content"]) for x in d.iloc[i]["messages"]]
            if not msgs or msgs[0]["role"] != "user" or len(msgs) < 2:
                continue
            if sum(len(x["content"]) for x in msgs[:2]) > max_chars:
                continue
            out.append(msgs[:2] if split == "test" else msgs)
            if n is not None and len(out) >= n:
                return out
    return out


_BL = {}


def _bl(tok):
    """compiler/bltok.py on the teacher's tokenizer.json: the exact encoder the calculator runs (HF builds
    differ in one whitespace corner case, see bltok.pretokenize)."""
    key = tok.name_or_path
    if key not in _BL:
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compiler"))
        import bltok
        from huggingface_hub import hf_hub_download
        _BL[key] = bltok.ByteLevelBPE(hf_hub_download(key, "tokenizer.json"))
    return _BL[key]


def tokenize_chat(tok, vocab, msgs, chat_text):
    """Teacher BPE (bltok, as on the device), then the merge split onto the kept vocabulary. Returns
    (ids, loss_mask) where the mask marks assistant tokens (including their <|im_end|>)."""
    bl = _bl(tok)
    enc = lambda text: vocab.resegment(bl.encode_full(text))  # noqa: E731
    ids, mask = [], []
    for m in msgs:
        head = [1] + enc(f"{m['role']}\n")
        body = enc(m["content"]) + [2]
        nl = enc("\n")
        a = m["role"] == "assistant"
        ids += head + body + nl
        mask += [0] * len(head) + [int(a)] * len(body) + [0] * len(nl)
    return ids, mask


def packed_batches(tok, vocab, chat_text, seq, batch, seed=0, max_chars=6000):
    """Endless stream of (ids [batch, seq+1], mask [batch, seq+1]) teacher-space ids, chats packed back to
    back. Chats longer than one block are cut."""
    rng = random.Random(seed)
    buf_i, buf_m = [], []
    rows_i, rows_m = [], []
    epoch = 0
    while True:
        for d in _frames("train"):
            idx = list(range(len(d)))
            rng.shuffle(idx)
            for i in idx:
                msgs = [dict(role=x["role"], content=x["content"]) for x in d.iloc[i]["messages"]]
                if not msgs or msgs[0]["role"] == "system" or sum(len(x["content"]) for x in msgs) > max_chars:
                    continue
                a, b = tokenize_chat(tok, vocab, msgs, chat_text)
                buf_i += a
                buf_m += b
                while len(buf_i) >= seq + 1:
                    rows_i.append(buf_i[:seq + 1])
                    rows_m.append(buf_m[:seq + 1])
                    buf_i, buf_m = buf_i[seq + 1:], buf_m[seq + 1:]
                    if len(rows_i) == batch:
                        yield np.array(rows_i, dtype=np.int64), np.array(rows_m, dtype=np.float32), epoch
                        rows_i, rows_m = [], []
        epoch += 1
