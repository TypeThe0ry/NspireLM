"""An untrained student for the CI gate: --layers teacher layers, the committed 16k vocabulary, int4 QAT
scales at their initial values. It answers nothing useful, but it exercises the whole chat path:
byte-level tokenizer, GQA 9/3, RoPE theta 1e5, learned-scale int4 compile, chat app.

    python3 train/make_test_student.py --layers 0,29 --out build/chat-test/student.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import TEACHER, attach_qat, init_student, load_vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,29")
    ap.add_argument("--keep", default=os.path.join(os.path.dirname(__file__), "artifacts", "keep.npy"))
    ap.add_argument("--gs", type=int, default=64, help="int4 group size (64: 4.25 bits/weight, fits with margin)")
    ap.add_argument("--no-qat", action="store_true", help="float weights (compile with any precision)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.float32)
    vocab = load_vocab(hf_hub_download(TEACHER, "tokenizer.json"), a.keep)
    layers = [int(x) for x in a.layers.split(",")]
    s = init_student(teacher, layers, vocab)
    if not a.no_qat:
        attach_qat(s, a.gs, a.gs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"model": s.state_dict(), "config": s.config.to_dict(), "layers": layers,
                "keep": np.asarray(vocab.keep), "gs": a.gs, "qat": not a.no_qat}, a.out)
    print("test student:", layers, len(vocab.keep), "->", a.out)


if __name__ == "__main__":
    main()
