#!/usr/bin/env python3
"""nslmc: ahead-of-time compiler for NspireLM.

Input:  a llama2.c checkpoint (legacy v0 .bin) and a llama2.c tokenizer .bin.
Output (in --out DIR):
  model.nslm      packed integer weights, in the layout the runtime kernels read
  nslm_model.h    generated C header: every shape, offset and constant baked in
  tok.nslt        tokenizer with integer merge ranks and a pre-sorted lookup index

Everything that can be decided before the calculator runs is decided here:
  * group-wise int8 weights, scales as (uint16 mantissa, per-row exponent)
  * log2(e)/sqrt(head_dim) folded into Wq, so softmax needs only exp2
  * RoPE cos/sin, exp2 and sigmoid tables precomputed in fixed point
  * tensor offsets and the run-state memory plan fixed at compile time
The runtime therefore contains no floating point at all.
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

MAGIC = 0x4D4C534E  # "NSLM"
VERSION = 1
ALIGN = 16

# --------------------------------------------------------------------------- checkpoint

class Ckpt:
    def __init__(self, path):
        with open(path, "rb") as f:
            hdr = struct.unpack("7i", f.read(28))
            data = np.frombuffer(f.read(), dtype=np.float32)
        dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq = hdr
        self.shared = vocab > 0
        vocab = abs(vocab)
        self.dim, self.hidden, self.n_layers = dim, hidden, n_layers
        self.n_heads, self.n_kv_heads, self.vocab, self.seq = n_heads, n_kv_heads, vocab, seq
        self.head_dim = dim // n_heads
        self.kv_dim = self.head_dim * n_kv_heads
        off = 0

        def take(*shape):
            nonlocal off
            n = int(np.prod(shape))
            t = data[off:off + n].reshape(shape)
            off += n
            return t

        L, kvd, hd = n_layers, self.kv_dim, self.head_dim
        self.tok_emb = take(vocab, dim)
        self.rms_att = take(L, dim)
        self.wq = take(L, dim, dim)
        self.wk = take(L, kvd, dim)
        self.wv = take(L, kvd, dim)
        self.wo = take(L, dim, dim)
        self.rms_ffn = take(L, dim)
        self.w1 = take(L, hidden, dim)
        self.w2 = take(L, dim, hidden)
        self.w3 = take(L, hidden, dim)
        self.rms_final = take(dim)
        take(seq, hd // 2)  # freq_cis_real (unused: run.c recomputes RoPE)
        take(seq, hd // 2)  # freq_cis_imag
        self.wcls = self.tok_emb if self.shared else take(vocab, dim)
        assert off == data.size or not self.shared, (off, data.size)


# --------------------------------------------------------------------------- tokenizer

class Tokenizer:
    """Port of llama2.c's BPE tokenizer (encode/decode), used by the tests."""

    def __init__(self, path, vocab_size):
        self.vocab, self.scores = [], []
        with open(path, "rb") as f:
            self.max_len = struct.unpack("i", f.read(4))[0]
            for _ in range(vocab_size):
                score, n = struct.unpack("fi", f.read(8))
                self.scores.append(score)
                self.vocab.append(f.read(n))
        self.lookup = {}
        for i, s in enumerate(self.vocab):
            self.lookup.setdefault(s, i)

    def encode(self, text, bos=True, eos=False):
        toks = [1] if bos else []
        b = text.encode("utf-8")
        if b:
            toks.append(self.lookup[b" "])  # dummy prefix, as in run.c
        i = 0
        while i < len(b):  # one token per UTF-8 codepoint, byte fallback
            j = i + 1
            while j < len(b) and (b[j] & 0xC0) == 0x80 and j - i < 4:
                j += 1
            cp = b[i:j]
            if cp in self.lookup:
                toks.append(self.lookup[cp])
            else:
                toks.extend(x + 3 for x in cp)
            i = j
        while True:  # greedy merge of the best-scoring pair
            best, best_id, best_i = -1e10, -1, -1
            for i in range(len(toks) - 1):  # run.c also scans the BOS slot
                m = self.vocab[toks[i]] + self.vocab[toks[i + 1]]
                tid = self.lookup.get(m, -1)
                if tid != -1 and self.scores[tid] > best:
                    best, best_id, best_i = self.scores[tid], tid, i
            if best_i == -1:
                break
            toks[best_i:best_i + 2] = [best_id]
        if eos:
            toks.append(2)
        return toks

    def piece(self, prev, tok):
        s = self.vocab[tok]
        if prev == 1 and s.startswith(b" "):
            s = s[1:]
        if len(s) == 6 and s.startswith(b"<0x") and s.endswith(b">"):
            s = bytes([int(s[3:5], 16)])
        return s

    def write_nslt(self, path):
        """Integer-only tokenizer: merge score -> rank (higher rank wins)."""
        order = sorted(range(len(self.scores)), key=lambda i: (self.scores[i], -i))
        rank = [0] * len(order)
        for r, i in enumerate(order):
            rank[i] = r
        sorted_ids = sorted(range(len(self.vocab)), key=lambda i: (self.vocab[i], i))
        blob = bytearray()
        blob += struct.pack("<4I", 0x544C534E, len(self.vocab), self.max_len, 0)
        offs, strs = [], bytearray()
        for s in self.vocab:
            offs.append(len(strs))
            strs += s + b"\0"
        blob += struct.pack(f"<{len(rank)}i", *rank)
        blob += struct.pack(f"<{len(offs)}I", *offs)
        blob += struct.pack(f"<{len(sorted_ids)}I", *sorted_ids)
        blob += struct.pack("<I", len(strs))
        blob += strs
        with open(path, "wb") as f:
            f.write(blob)


# --------------------------------------------------------------------------- float reference

def rope_angles(seq, head_dim):
    """Exactly run.c's formula, in float32 like run.c."""
    i = np.arange(0, head_dim, 2, dtype=np.float32)
    freq = (1.0 / np.power(np.float32(10000.0), i / np.float32(head_dim))).astype(np.float32)
    pos = np.arange(seq, dtype=np.float32)[:, None]
    return (pos * freq[None, :]).astype(np.float32)


class FloatModel:
    """float32 forward pass that mirrors llama2.c run.c, used as the quality reference."""

    def __init__(self, ck):
        self.ck = ck
        ang = rope_angles(ck.seq, ck.head_dim)
        self.cos, self.sin = np.cos(ang), np.sin(ang)
        self.reset()

    def reset(self):
        ck = self.ck
        self.kc = np.zeros((ck.n_layers, ck.seq, ck.kv_dim), np.float32)
        self.vc = np.zeros((ck.n_layers, ck.seq, ck.kv_dim), np.float32)

    @staticmethod
    def rmsnorm(x, w):
        ss = np.float32(np.mean(x * x, dtype=np.float32) + np.float32(1e-5))
        return (w * (x / np.sqrt(ss))).astype(np.float32)

    def rope(self, v, pos, nh):
        hd = self.ck.head_dim
        v = v.reshape(nh, hd // 2, 2)
        c, s = self.cos[pos], self.sin[pos]
        a, b = v[..., 0].copy(), v[..., 1].copy()
        v[..., 0] = a * c - b * s
        v[..., 1] = a * s + b * c
        return v.reshape(-1)

    def forward(self, tok, pos):
        ck = self.ck
        hd, nh, nkv = ck.head_dim, ck.n_heads, ck.n_kv_heads
        x = ck.tok_emb[tok].astype(np.float32).copy()
        for l in range(ck.n_layers):
            xb = self.rmsnorm(x, ck.rms_att[l])
            q = ck.wq[l] @ xb
            k = ck.wk[l] @ xb
            v = ck.wv[l] @ xb
            q = self.rope(q, pos, nh)
            k = self.rope(k, pos, nkv)
            self.kc[l, pos], self.vc[l, pos] = k, v
            out = np.zeros(ck.dim, np.float32)
            kvmul = nh // nkv
            for h in range(nh):
                kh = h // kvmul
                qh = q[h * hd:(h + 1) * hd]
                K = self.kc[l, :pos + 1, kh * hd:(kh + 1) * hd]
                V = self.vc[l, :pos + 1, kh * hd:(kh + 1) * hd]
                att = (K @ qh) / np.float32(math.sqrt(hd))
                att = np.exp(att - att.max())
                att /= att.sum()
                out[h * hd:(h + 1) * hd] = att @ V
            x = x + ck.wo[l] @ out
            xb = self.rmsnorm(x, ck.rms_ffn[l])
            h1 = ck.w1[l] @ xb
            h3 = ck.w3[l] @ xb
            g = (h1 * (1.0 / (1.0 + np.exp(-h1)))) * h3
            x = x + ck.w2[l] @ g
        x = self.rmsnorm(x, ck.rms_final)
        return ck.wcls @ x


# --------------------------------------------------------------------------- quantization

def quant_q8(w, gs):
    """Group-wise symmetric int8. Returns (q int8 [r,c], mant uint16 [r,ng], exp int8 [r])."""
    r, c = w.shape
    assert c % gs == 0, (c, gs)
    ng = c // gs
    wg = w.astype(np.float64).reshape(r, ng, gs)
    s = np.abs(wg).max(axis=2) / 127.0
    s = np.where(s == 0, 1e-12, s)
    smax = s.max(axis=1)
    e = np.floor(np.log2(65535.0 / smax)).astype(np.int64)
    e = np.clip(e, -60, 60)
    m = np.clip(np.round(s * np.exp2(e)[:, None]), 1, 65535)
    s_eff = m / np.exp2(e)[:, None]
    q = np.clip(np.round(wg / s_eff[:, :, None]), -127, 127).astype(np.int8).reshape(r, c)
    return q, m.astype(np.uint16), e.astype(np.int8)


def dequant_q8(q, m, e, gs):
    r, c = q.shape
    s = m.astype(np.float64) / np.exp2(e.astype(np.float64))[:, None]
    return (q.reshape(r, c // gs, gs) * s[:, :, None]).reshape(r, c)


def pick_gs(cols, want):
    """Largest divisor of cols that is <= want, preferring multiples of 8 (kernel friendly)."""
    divs = [g for g in range(1, want + 1) if cols % g == 0]
    m8 = [g for g in divs if g % 8 == 0]
    return max(m8) if m8 else max(divs)


# --------------------------------------------------------------------------- emission

class Blob:
    def __init__(self):
        self.buf = bytearray()
        self.tensors = []  # (name, offset, nbytes)

    def add(self, name, arr):
        while len(self.buf) % ALIGN:
            self.buf.append(0)
        b = np.ascontiguousarray(arr).tobytes()
        self.tensors.append((name, len(self.buf), len(b)))
        self.buf += b


def compile_model(ck, tok, outdir, gs_want=64):
    os.makedirs(outdir, exist_ok=True)
    gs_d = pick_gs(ck.dim, gs_want)
    gs_h = pick_gs(ck.hidden, gs_want)
    hd = ck.head_dim
    qfold = math.log2(math.e) / math.sqrt(hd)  # softmax in base 2, scaled scores

    blob = Blob()
    q8 = []

    def add_mat(name, w):
        q, m, e = quant_q8(w, gs_d if w.shape[1] == ck.dim else gs_h)
        blob.add(name + "_q", q)
        blob.add(name + "_m", m)
        blob.add(name + "_e", e)
        q8.append(name)

    def q16(v):  # norm gains as Q16 int32
        return np.round(v.astype(np.float64) * 65536).astype(np.int32)

    add_mat("emb", ck.tok_emb)
    for l in range(ck.n_layers):
        blob.add(f"l{l}_rms_att", q16(ck.rms_att[l]))
        add_mat(f"l{l}_wq", ck.wq[l] * qfold)
        add_mat(f"l{l}_wk", ck.wk[l])
        add_mat(f"l{l}_wv", ck.wv[l])
        add_mat(f"l{l}_wo", ck.wo[l])
        blob.add(f"l{l}_rms_ffn", q16(ck.rms_ffn[l]))
        add_mat(f"l{l}_w1", ck.w1[l])
        add_mat(f"l{l}_w2", ck.w2[l])
        add_mat(f"l{l}_w3", ck.w3[l])
    blob.add("rms_final", q16(ck.rms_final))
    if not ck.shared:
        add_mat("cls", ck.wcls)

    # RoPE table in Q30 from the float32 angles run.c would use (rounded once, in float64).
    ang = rope_angles(ck.seq, hd).astype(np.float64)
    rope = np.stack([np.round(np.cos(ang) * (1 << 30)), np.round(np.sin(ang) * (1 << 30))], axis=-1)
    blob.add("rope", rope.astype(np.int32))

    header = struct.pack("<8I", MAGIC, VERSION, len(blob.buf), len(blob.tensors), 0, 0, 0, 0)
    with open(os.path.join(outdir, "model.nslm"), "wb") as f:
        f.write(header)
        f.write(blob.buf)

    # exp2(-f) for f in [0,1], Q30, 257 entries (interpolated by 8 fractional bits)
    exp2_lut = [int(round(2.0 ** (-i / 256.0) * (1 << 30))) for i in range(257)]
    # sigmoid over [-16, 16] in steps of 1/32, Q30, 1025 entries
    sig_lut = [int(round((1 << 30) / (1.0 + math.exp(-(-16.0 + i / 32.0))))) for i in range(1025)]

    H = []
    w = H.append
    w("/* Generated by compiler/nslmc.py. Do not edit. */")
    w("#ifndef NSLM_MODEL_H\n#define NSLM_MODEL_H\n#include <stdint.h>")
    for k, v in (("DIM", ck.dim), ("HIDDEN", ck.hidden), ("LAYERS", ck.n_layers),
                 ("HEADS", ck.n_heads), ("KV_HEADS", ck.n_kv_heads), ("HEAD_DIM", hd),
                 ("KV_DIM", ck.kv_dim), ("VOCAB", ck.vocab), ("SEQ", ck.seq), ("GS_D", gs_d), ("GS_H", gs_h),
                 ("SHARED_CLS", int(ck.shared)), ("BLOB_BYTES", len(blob.buf)),
                 ("MAGIC", MAGIC), ("VERSION", VERSION)):
        w(f"#define NSLM_{k} {v}")
    w("#define NSLM_HDR_BYTES 32")
    for name, off, n in blob.tensors:
        w(f"#define NSLM_OFF_{name} {off}u")
    w("#define NSLM_OFF_L_STRIDE (NSLM_OFF_l1_rms_att - NSLM_OFF_l0_rms_att)" if ck.n_layers > 1
      else "#define NSLM_OFF_L_STRIDE 0u")
    w("static const int32_t nslm_exp2_lut[257] = {" + ",".join(map(str, exp2_lut)) + "};")
    w("static const int32_t nslm_sig_lut[1025] = {" + ",".join(map(str, sig_lut)) + "};")
    w("#endif")
    with open(os.path.join(outdir, "nslm_model.h"), "w") as f:
        f.write("\n".join(H) + "\n")

    tok.write_nslt(os.path.join(outdir, "tok.nslt"))
    # layer offsets must be uniform so the runtime can index by stride
    if ck.n_layers > 1:
        base = dict((n, o) for n, o, _ in blob.tensors)
        stride = base["l1_rms_att"] - base["l0_rms_att"]
        for l in range(ck.n_layers):
            for t in ("rms_att", "wq_q", "w2_e", "rms_ffn"):
                assert base[f"l{l}_{t}"] == base[f"l0_{t}"] + l * stride, (l, t)
    return {"gs": (gs_d, gs_h), "blob_bytes": len(blob.buf)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gs", type=int, default=64)
    a = ap.parse_args(argv)
    ck = Ckpt(a.ckpt)
    tok = Tokenizer(a.tokenizer, ck.vocab)
    info = compile_model(ck, tok, a.out, a.gs)
    print(f"nslmc: dim={ck.dim} hidden={ck.hidden} layers={ck.n_layers} heads={ck.n_heads}/"
          f"{ck.n_kv_heads} vocab={ck.vocab} seq={ck.seq} gs={info['gs']} "
          f"blob={info['blob_bytes'] / 2**20:.2f} MiB -> {a.out}")


if __name__ == "__main__":
    main()
