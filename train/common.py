"""Shared pieces of the distillation pipeline (train/README.md).

The student is a Llama model at the teacher's width (SmolLM2-135M-Instruct: 576 wide, GQA 9/3 heads,
RoPE theta 1e5). It differs from the teacher in two ways:
  * depth: a subset of the teacher's 30 layers (train/select_layers.py), copied as the initialisation;
  * vocabulary: the NSLM_KEEP most used tokens of the teacher's 49152. Text is tokenized with the teacher's
    BPE, and every compiled-out token is split along its own BPE merge until all pieces are kept
    (`Vocab.resegment`). The calculator runtime does exactly the same split, so training and device
    agree token for token.
Teacher and student share token ids in teacher space, so distillation needs no alignment: the teacher's
distribution is restricted to the kept tokens and renormalised.
"""
import json
import os

import numpy as np

TEACHER = "HuggingFaceTB/SmolLM2-135M-Instruct"
SPECIAL = (0, 1, 2)  # <|endoftext|>, <|im_start|>, <|im_end|>
IM_START, IM_END, EOT = 1, 2, 0


def bytes_to_unicode():
    """GPT-2's byte -> printable character map used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def chat_text(messages):
    """The device prompt format: ChatML without a system prompt (it would cost ~20 of 256 positions)."""
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


class Vocab:
    """Kept token set and the merge-split re-segmentation."""

    def __init__(self, tokenizer_json, keep):
        d = json.load(open(tokenizer_json))
        model = d["model"]
        self.t2i = model["vocab"]
        self.n_full = len(self.t2i)
        self.i2t = [None] * self.n_full
        for t, i in self.t2i.items():
            self.i2t[i] = t
        # each merged token -> its two parts (the first merge producing a token is the one BPE uses)
        self.split = {}
        for mg in model["merges"]:
            a, b = mg.split(" ") if isinstance(mg, str) else mg
            t = self.t2i[a + b]
            self.split.setdefault(t, (self.t2i[a], self.t2i[b]))
        self.keep = np.array(sorted(set(int(k) for k in keep)), dtype=np.int64)
        self.old2new = np.full(self.n_full, -1, dtype=np.int64)
        self.old2new[self.keep] = np.arange(len(self.keep))
        self._memo = {}

    @classmethod
    def build(cls, tokenizer_json, usage, n_keep):
        """Most used tokens, plus everything that must exist: the special tokens and the 256 byte symbols
        (so every split terminates)."""
        d = json.load(open(tokenizer_json))
        t2i = d["model"]["vocab"]
        must = set(SPECIAL)
        for ch in bytes_to_unicode().values():
            if ch in t2i:  # bytes that never occur in UTF-8 (0xC0, 0xC1, 0xF5..) may be absent
                must.add(t2i[ch])
        order = [int(i) for i in np.argsort(-usage, kind="stable") if int(i) not in must]
        keep = sorted(must) + order[: n_keep - len(must)]
        return cls(tokenizer_json, keep)

    def pieces(self, t):
        """Kept teacher ids that token t splits into."""
        if self.old2new[t] >= 0:
            return (t,)
        r = self._memo.get(t)
        if r is None:
            if t in self.split:
                a, b = self.split[t]
                r = self.pieces(a) + self.pieces(b)
            else:  # added tokens (<repo_name> ...) have no merge: spell them out in byte symbols
                r = sum((self.pieces(self.t2i[ch]) for ch in self.i2t[t]), ())
            self._memo[t] = r
        return r

    def resegment(self, ids):
        out = []
        for t in ids:
            out.extend(self.pieces(int(t)))
        return out

    def save(self, path):
        np.save(path, self.keep)


def load_vocab(tokenizer_json, keep_path):
    return Vocab(tokenizer_json, np.load(keep_path))


def student_config(teacher_cfg, n_layers, n_vocab):
    from transformers import LlamaConfig
    c = teacher_cfg.to_dict()
    c.update(num_hidden_layers=n_layers, vocab_size=n_vocab, bos_token_id=None, eos_token_id=None,
             pad_token_id=None, tie_word_embeddings=True)
    for k in ("architectures", "transformers.js_config", "torch_dtype", "dtype", "_name_or_path"):
        c.pop(k, None)
    return LlamaConfig(**c)


def init_student(teacher, layers, vocab):
    """Student initialised from the chosen teacher layers and the kept embedding rows."""
    import torch
    from transformers import LlamaForCausalLM
    cfg = student_config(teacher.config, len(layers), len(vocab.keep))
    s = LlamaForCausalLM(cfg)
    with torch.no_grad():
        keep = torch.as_tensor(vocab.keep)
        s.model.embed_tokens.weight.copy_(teacher.model.embed_tokens.weight[keep].float())
        for j, i in enumerate(layers):
            s.model.layers[j].load_state_dict(teacher.model.layers[i].state_dict())
        s.model.norm.load_state_dict(teacher.model.norm.state_dict())
    s.tie_weights()
    return s


# ----------------------------------------------------------------------------- quantisation-aware training

class FakeQuant4:
    """Group-wise int4 with codes -8..7 and a learned per-group scale (LSQ), the exact grid of the runtime's
    q4 format (compiler/nslmc.py quant_q4). Applied to every 2D weight through a forward pre-hook."""

    @staticmethod
    def init_scale(w, gs):
        import torch
        wg = w.detach().float().reshape(w.shape[0], -1, gs)
        best, bs = None, None
        for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
            s = (wg.abs().amax(-1, keepdim=True) * c / 7.5).clamp_min(1e-8)
            err = ((torch.clamp(torch.round(wg / s), -8, 7) * s - wg) ** 2).sum(-1, keepdim=True)
            if best is None:
                best, bs = err, s
            else:
                m = err < best
                best, bs = torch.where(m, err, best), torch.where(m, s, bs)
        return bs.squeeze(-1)

    @staticmethod
    def apply(w, s, gs):
        import torch
        r, c = w.shape
        wg = w.reshape(r, c // gs, gs)
        sg = s[:, :, None]
        # LSQ gradient scaling
        g = 1.0 / (gs * 7) ** 0.5
        sg = sg + (sg * g - (sg * g).detach())  # forward value exactly s (export rounds with s)
        x = wg / sg
        q = torch.clamp(x + (torch.round(x) - x).detach(), -8, 7)
        return (q * sg).reshape(r, c)

    @staticmethod
    def codes(w, s, gs):
        import torch
        r, c = w.shape
        return torch.clamp(torch.round(w.reshape(r, c // gs, gs) / s[:, :, None]), -8, 7).reshape(r, c)


def attach_qat(model, gs=32, emb_gs=32):
    """Gives every linear weight and the (tied) embedding a learned scale per group and routes their use
    through FakeQuant4: the embedding lookup, every matmul and the tied classifier."""
    import torch.nn as nn
    import torch.nn.functional as F
    e = model.model.embed_tokens
    for m in model.modules():
        if isinstance(m, nn.Linear) and m is not model.lm_head:
            m.qat_s = nn.Parameter(FakeQuant4.init_scale(m.weight, gs))
            m.qat_gs = gs
            m.forward = (lambda mod: lambda x: F.linear(x, FakeQuant4.apply(mod.weight, mod.qat_s, mod.qat_gs)))(m)
    e.qat_s = nn.Parameter(FakeQuant4.init_scale(e.weight, emb_gs))
    e.qat_gs = emb_gs
    e.forward = lambda ids: F.embedding(ids, FakeQuant4.apply(e.weight, e.qat_s, e.qat_gs))
    model.lm_head.forward = lambda x: F.linear(x, FakeQuant4.apply(e.weight, e.qat_s, e.qat_gs))
    return model


def env_path(name, default):
    return os.environ.get(name, default)
