"""Student (train/distill.py student.pt) -> llama2.c v0 checkpoint + sidecars for compiler/nslmc.py.

    python3 train/export.py runs/s9/student.pt build/chat/student.bin
writes student.bin (float32 weights), student.bin.json (rope_theta, kept teacher ids) and, after QAT,
student.bin.qat.npz (the learned int4 group scales). Checks that the compiler's float reference
(nslmc.FloatModel, llama2.c layout) reproduces the HF student's logits.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "compiler"))


def hf_to_interleaved(w, n_heads):
    """HF rotate-half RoPE row order -> llama2.c's interleaved pairs (llama2.c export.py permute_reverse)."""
    d1 = w.shape[0]
    return w.reshape(n_heads, 2, d1 // n_heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(d1, *w.shape[1:])


def load_student(path):
    from transformers import LlamaConfig, LlamaForCausalLM
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LlamaConfig(**st["config"])
    m = LlamaForCausalLM(cfg)
    sd = {k: v for k, v in st["model"].items() if not k.endswith("qat_s")}
    m.load_state_dict(sd, strict=False)
    m.tie_weights()
    scales = {k: v for k, v in st["model"].items() if k.endswith("qat_s")}
    return m.eval(), st, scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("student")
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=320, help="context compiled for the device")
    a = ap.parse_args()
    m, st, scales = load_student(a.student)
    c = m.config
    L, H, KV = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
    g = lambda t: t.detach().float().numpy()  # noqa: E731
    lay = m.model.layers
    hd = c.hidden_size // H
    parts = [
        g(m.model.embed_tokens.weight),
        np.stack([g(l.input_layernorm.weight) for l in lay]),
        np.stack([hf_to_interleaved(g(l.self_attn.q_proj.weight), H) for l in lay]),
        np.stack([hf_to_interleaved(g(l.self_attn.k_proj.weight), KV) for l in lay]),
        np.stack([g(l.self_attn.v_proj.weight) for l in lay]),
        np.stack([g(l.self_attn.o_proj.weight) for l in lay]),
        np.stack([g(l.post_attention_layernorm.weight) for l in lay]),
        np.stack([g(l.mlp.gate_proj.weight) for l in lay]),
        np.stack([g(l.mlp.down_proj.weight) for l in lay]),
        np.stack([g(l.mlp.up_proj.weight) for l in lay]),
        g(m.model.norm.weight),
        np.zeros((a.seq, hd // 2), np.float32), np.zeros((a.seq, hd // 2), np.float32),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(struct.pack("7i", c.hidden_size, c.intermediate_size, L, H, KV, c.vocab_size, a.seq))
        for p in parts:
            f.write(np.ascontiguousarray(p, dtype=np.float32).tobytes())
    meta = {"rope_theta": float(getattr(c, "rope_theta", None) or c.rope_parameters["rope_theta"]),
            "rms_eps": c.rms_norm_eps, "keep": [int(x) for x in st["keep"]], "layers": st["layers"],
            "tokenizer": "byte-level", "chat": "chatml"}
    json.dump(meta, open(a.out + ".json", "w"))
    if scales:
        names = {"q_proj": ("wq", H), "k_proj": ("wk", KV), "v_proj": ("wv", 0), "o_proj": ("wo", 0),
                 "gate_proj": ("w1", 0), "down_proj": ("w2", 0), "up_proj": ("w3", 0)}
        out = {}
        for k, v in scales.items():
            v = v.float().numpy()
            if k.startswith("model.embed_tokens"):
                out["emb"] = v
                continue
            li = int(k.split(".")[2])
            nm, nh = names[k.split(".")[-2]]
            out[f"l{li}_{nm}"] = hf_to_interleaved(v, nh) if nh else v
        np.savez(a.out + ".qat.npz", **out)

    # self-check: llama2.c-layout float reference vs the HF model (with QAT: the fake-quantised weights)
    import nslmc
    ck = nslmc.Ckpt(a.out)
    if scales:
        ck.fake_quant()
        from common import attach_qat
        attach_qat(m, st["gs"], st["gs"])
        with torch.no_grad():
            for k, v in scales.items():
                mod = m.get_submodule(k.rsplit(".", 1)[0])
                mod.qat_s.copy_(v)
    fm = nslmc.FloatModel(ck)
    ids = [int(x) for x in np.random.RandomState(0).randint(0, c.vocab_size, 24)]
    with torch.no_grad():
        ref = m(torch.tensor(ids)[None]).logits[0].float().numpy()
    got = np.stack([fm.forward(t, p).copy() for p, t in enumerate(ids)])
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print(f"export: {L} layers, vocab {c.vocab_size}, seq {a.seq}, qat={'yes' if scales else 'no'}; "
          f"llama2.c-layout float vs HF: max rel err {err:.2e}")
    assert err < 1e-3, "export mismatch"


if __name__ == "__main__":
    main()
