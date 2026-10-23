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
import json
import math
import os
import struct
import sys

import numpy as np

MAGIC = 0x4D4C534E  # "NSLM"
VERSION = 1
ALIGN = 32  # cache line: one x4 weight chunk must be one line
SEG_MAX = 1 << 20  # largest separately allocated piece of the model (the Ndless heap fragments)

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
        # optional sidecars written by train/export.py: <ckpt>.json (rope_theta, tokenizer) and
        # <ckpt>.qat.npz (learned int4 group scales, so the weights compile without re-rounding)
        meta = path + ".json"
        self.meta = json.load(open(meta)) if os.path.exists(meta) else {}
        self.rope_theta = float(self.meta.get("rope_theta", 10000.0))
        qs = path + ".qat.npz"
        self.qat = dict(np.load(qs)) if os.path.exists(qs) else None

    def fake_quant(self):
        """Replace the weights by the int4 values QAT trained with (round(w / s) on the learned scales, in
        float32 like torch), so FloatModel is the reference for what the device should compute."""
        def fq(w, S):
            gs = w.shape[1] // S.shape[1]
            q = np.clip(np.round(w.reshape(w.shape[0], -1, gs) / S[:, :, None]), -8, 7)
            return (q * S[:, :, None]).reshape(w.shape).astype(np.float32)
        for k in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            setattr(self, k, np.stack([fq(getattr(self, k)[l], self.qat[f"l{l}_{k}"]) for l in range(self.n_layers)]))
        self.tok_emb = self.wcls = fq(self.tok_emb, self.qat["emb"])


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
        assert self.max_len <= 64, "runtime re-segmentation buffer"
        self.lookup = {}
        for i, s in enumerate(self.vocab):
            self.lookup.setdefault(s, i)
        self.set_keep(None)

    def set_keep(self, keep):
        """Restrict the model vocabulary to `keep` (sorted old ids); None = all."""
        n = len(self.vocab)
        self.new2old = list(range(n)) if keep is None else list(keep)
        self.old2new = [-1] * n
        for new, old in enumerate(self.new2old):
            self.old2new[old] = new

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
        return self.fixup(toks)

    def fixup(self, toks):
        """Map to model ids; a pruned token is re-segmented by longest match over kept tokens."""
        out = []
        for t in toks:
            if self.old2new[t] >= 0:
                out.append(self.old2new[t])
                continue
            b, i = self.vocab[t], 0
            while i < len(b):
                for L in range(min(self.max_len, len(b) - i), 0, -1):
                    tid = self.lookup.get(b[i:i + L], -1)
                    if tid >= 0 and self.old2new[tid] >= 0:
                        out.append(self.old2new[tid])
                        i += L
                        break
                else:
                    out.append(self.old2new[b[i] + 3])
                    i += 1
        return out

    def piece(self, prev, tok):  # model ids
        s = self.vocab[self.new2old[tok]]
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
        blob += struct.pack("<4I", 0x544C534E, len(self.vocab), self.max_len, len(self.new2old))
        offs, strs = [], bytearray()
        for s in self.vocab:
            offs.append(len(strs))
            strs += s + b"\0"
        blob += struct.pack(f"<{len(rank)}i", *rank)
        blob += struct.pack(f"<{len(offs)}I", *offs)
        blob += struct.pack(f"<{len(sorted_ids)}I", *sorted_ids)
        blob += struct.pack(f"<{len(self.old2new)}i", *self.old2new)
        blob += struct.pack(f"<{len(self.new2old)}i", *self.new2old)
        blob += struct.pack("<I", len(strs))
        blob += strs
        with open(path, "wb") as f:
            f.write(blob)


class BLModelTok:
    """compile_model's tokenizer interface for a byte-level BPE whose kept tokens (teacher ids, in model-id
    order) are already the checkpoint's rows."""

    def __init__(self, tokenizer_json, keep):
        import bltok
        self.bl = bltok.ByteLevelBPE(tokenizer_json, keep)
        self.new2old = list(range(len(self.bl.keep)))

    def set_keep(self, keep):
        assert keep is None, "vocabulary pruning happens in training for byte-level models"

    def encode(self, text):
        return self.bl.encode(text)

    def write_nslt(self, path):
        self.bl.write_nslt(path)


# --------------------------------------------------------------------------- float reference

def rope_angles(seq, head_dim, theta=10000.0):
    """Exactly run.c's formula, in float32 like run.c."""
    i = np.arange(0, head_dim, 2, dtype=np.float32)
    freq = (1.0 / np.power(np.float32(theta), i / np.float32(head_dim))).astype(np.float32)
    pos = np.arange(seq, dtype=np.float32)[:, None]
    return (pos * freq[None, :]).astype(np.float32)


class FloatModel:
    """float32 forward pass that mirrors llama2.c run.c, used as the quality reference."""

    def __init__(self, ck):
        self.ck = ck
        ang = rope_angles(ck.seq, ck.head_dim, ck.rope_theta)
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
    """Group-wise symmetric int8 with kernel-safe scales.

    Returns q (int8 [r,c]), m (uint16 [r,ng], <= 32767 so SMLAWB can read it as signed), e (int8 [r]).
    Real weight = q * m * 2^-e. The kernel accumulates sum_g (d_g * m_g) >> 16 in int32, where
    |d_g| <= 32768 * sum|q|. For each row we check that bound against 2^30 and lower the mantissa
    precision if needed, so the int32 accumulator cannot overflow for any activation vector.
    """
    r, c = w.shape
    assert c % gs == 0, (c, gs)
    ng = c // gs
    wg = w.astype(np.float64).reshape(r, ng, gs)
    s = np.abs(wg).max(axis=2) / 127.0
    s = np.where(s == 0, 1e-12, s)
    smax = s.max(axis=1)
    e = np.clip(np.floor(np.log2(32767.0 / smax)).astype(np.int64), -60, 90)
    while True:
        m = np.clip(np.round(s * np.exp2(e)[:, None]), 1, 32767)
        s_eff = m / np.exp2(e)[:, None]
        q = np.clip(np.round(wg / s_eff[:, :, None]), -127, 127)
        bound = (np.abs(q).sum(axis=2) * 32768.0 * m / 65536.0).sum(axis=1)
        bad = bound >= 2.0 ** 30
        if not bad.any():
            break
        e = np.where(bad, e - 1, e)
    return q.astype(np.int8).reshape(r, c), m.astype(np.uint16), e.astype(np.int8)


def scales_to_me(S, Q):
    """Real group scales S [r, ng] -> uint16 mantissas (<= 32767) and per-row exponents, lowering
    precision where needed so the kernel's int32 accumulator provably cannot overflow (see quant_q8)."""
    r, ng = S.shape
    gs = Q.shape[1] // ng
    S = np.where(S <= 0, 1e-12, S)
    e = np.clip(np.floor(np.log2(32767.0 / S.max(axis=1))).astype(np.int64), -60, 90)
    absq = np.abs(Q.reshape(r, ng, gs)).sum(axis=2).astype(np.float64)
    while True:
        m = np.clip(np.round(S * np.exp2(e)[:, None]), 1, 32767)
        bad = (absq * 32768.0 * m / 65536.0).sum(axis=1) >= 2.0 ** 30
        if not bad.any():
            return m.astype(np.uint16), e.astype(np.int8)
        e = np.where(bad, e - 1, e)


def quant_q4(w, gs, X=None, S=None):
    """Group-wise int4 (codes -8..7). With learned scales S [rows, groups] (quantisation-aware training)
    the codes are round(w / S); with calibration inputs X, GPTQ rounding; otherwise round-to-nearest with
    a per-group MSE clipping search."""
    import gptq
    if S is not None:
        r, c = w.shape
        S = S.astype(np.float64)
        assert S.shape == (r, c // gs), (S.shape, w.shape, gs)
        Q = np.clip(np.round(w.astype(np.float64).reshape(r, c // gs, gs) / S[:, :, None]), -8, 7)
        Q = Q.reshape(r, c).astype(np.int64)
    elif X is not None:
        Q, S = gptq.gptq_quant(w, X, 4, gs)
    else:
        r, c = w.shape
        wg = w.astype(np.float64).reshape(r, c // gs, gs)
        S = np.stack([gptq._group_scale(wg[:, g, :], 7, -8, (1.0, 0.95, 0.9, 0.85, 0.8, 0.75))
                      for g in range(c // gs)], axis=1)
        Q = np.clip(np.round(wg / S[:, :, None]), -8, 7).reshape(r, c).astype(np.int64)
    m, e = scales_to_me(S, Q)
    return Q.astype(np.int8), m, e


def pack_x4n(q, m, e):
    """Kernel layout "x4n" (int4): blocks of 4 rows; each 32-byte line holds 16 columns of the 4 rows.
    Words 0-3 are columns 0-7 of rows 0-3, words 4-7 columns 8-15; column j of a word is bits 4j..4j+3,
    stored offset-binary (q + 8)."""
    r, c = q.shape
    assert r % 4 == 0 and c % 16 == 0, (r, c)
    u = (q.astype(np.int16) + 8).astype(np.uint32)
    u = u.reshape(r // 4, 4, c // 16, 2, 8)              # rb, j, chunk, half, k
    words = (u << (4 * np.arange(8, dtype=np.uint32))).sum(axis=-1).astype(np.uint32)
    words = words.transpose(0, 2, 3, 1).copy()            # rb, chunk, half, j
    ng = m.shape[1]
    mp = m.reshape(r // 4, 4, ng).transpose(0, 2, 1).copy()
    return words, mp, e


def dequant_q8(q, m, e, gs):
    r, c = q.shape
    s = m.astype(np.float64) / np.exp2(e.astype(np.float64))[:, None]
    return (q.reshape(r, c // gs, gs) * s[:, :, None]).reshape(r, c)


def pack_x4(q, m, e):
    """Kernel layout "x4": blocks of 4 rows; each 32-byte cache line holds 8 columns of the 4 rows,
    stored offset-binary (q + 128) so the kernel can unpack bytes into 16-bit lanes with one AND.
    Scales are blocked the same way: [row_block][group][4]."""
    r, c = q.shape
    assert r % 4 == 0 and c % 8 == 0, (r, c)
    u = (q.astype(np.int16) + 128).astype(np.uint8)
    qp = u.reshape(r // 4, 4, c // 8, 8).transpose(0, 2, 1, 3).copy()
    ng = m.shape[1]
    mp = m.reshape(r // 4, 4, ng).transpose(0, 2, 1).copy()
    return qp, mp, e


def pick_gs16(cols, want):
    """int4 groups: a multiple of 16 (one x4n line) dividing cols, at most `want`."""
    g = [x for x in range(16, want + 1, 16) if cols % x == 0]
    assert g, f"no int4 group size for {cols} columns"
    return max(g)


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


def compile_model(ck, tok, outdir, gs_want=64, keep=None, bits_att=8, bits_ffn=8, gs4=32, calib=None, bits_emb=8):
    os.makedirs(outdir, exist_ok=True)
    assert ck.dim % 8 == 0 and ck.kv_dim % 4 == 0, "x4 layout needs these"
    # pad the FFN width to a multiple of 16 with zero weights: silu(0) * 0 contributes nothing
    hid = -(-ck.hidden // 16) * 16
    w1 = np.zeros((ck.n_layers, hid, ck.dim), np.float32); w1[:, :ck.hidden] = ck.w1
    w3 = np.zeros((ck.n_layers, hid, ck.dim), np.float32); w3[:, :ck.hidden] = ck.w3
    w2 = np.zeros((ck.n_layers, ck.dim, hid), np.float32); w2[:, :, :ck.hidden] = ck.w2
    gs_d = pick_gs(ck.dim, gs_want)
    gs_h = pick_gs(hid, gs_want)
    hd = ck.head_dim
    qfold = math.log2(math.e) / math.sqrt(hd)  # softmax in base 2, scaled scores

    blob = Blob()
    q8 = []
    g4_d, g4_h = pick_gs16(ck.dim, gs4), pick_gs16(hid, gs4)

    def qat_s(name, w, scale=1.0):
        """learned scales for this matrix, padded like the weights (padding gets a dummy scale)"""
        if ck.qat is None:
            return None
        S = ck.qat[name].astype(np.float64) * scale
        gs = g4_d if w.shape[1] == ck.dim else g4_h
        P = np.ones((w.shape[0], w.shape[1] // gs))
        P[:S.shape[0], :S.shape[1]] = S
        return P

    def add_mat(name, w, bits=8, X=None, S=None):
        assert (w.shape[1] == ck.dim) or (w.shape[1] == hid)
        if bits == 4:
            if X is not None and X.shape[1] < w.shape[1]:  # padded FFN columns see zero input
                X = np.concatenate([X, np.zeros((len(X), w.shape[1] - X.shape[1]))], axis=1)
            q, m, e = quant_q4(w, g4_d if w.shape[1] == ck.dim else g4_h, X, S)
            qp, mp, ep = pack_x4n(q, m, e)
        else:
            q, m, e = quant_q8(w, gs_d if w.shape[1] == ck.dim else gs_h)
            qp, mp, ep = pack_x4(q, m, e)
        blob.add(name + "_q", qp)
        blob.add(name + "_m", mp)
        blob.add(name + "_e", ep)
        q8.append(name)

    def add_mat_parts(name, w, bits, X, part_rows, S=None):
        """Quantise the whole matrix (rows are independent), store it as row parts of <= SEG_MAX bytes so
        each part can live in its own allocation."""
        q, m, e = (quant_q4(w, g4_d, X, S) if bits == 4 else quant_q8(w, gs_d))
        for p, r0 in enumerate(range(0, w.shape[0], part_rows)):
            r1 = min(r0 + part_rows, w.shape[0])
            qp, mp, ep = (pack_x4n if bits == 4 else pack_x4)(q[r0:r1], m[r0:r1], e[r0:r1])
            blob.add(f"{name}_p{p}_q", qp)
            blob.add(f"{name}_p{p}_m", mp)
            blob.add(f"{name}_p{p}_e", ep)
        return -(-w.shape[0] // part_rows)

    def cx(l, key):  # calibration inputs for GPTQ, if any
        return None if calib is None else calib[f"l{l}_{key}"]

    def q16(v):  # norm gains as Q16 int32
        return np.round(v.astype(np.float64) * 65536).astype(np.int32)

    tok.set_keep(keep)
    vocab = len(tok.new2old)
    vrows = -(-vocab // 4) * 4  # x4 layout: pad with zero rows, never sampled
    emb = np.zeros((vrows, ck.dim), np.float32)
    emb[:vocab] = ck.tok_emb[tok.new2old]
    if bits_emb == 4 and calib is not None and ck.shared:  # the tied matrix is calibrated as the classifier
        Xc = calib["cls_in"]
    else:
        Xc = None
    part_rows = max(4, (SEG_MAX // (ck.dim if bits_emb == 8 else ck.dim // 2)) // 4 * 4)
    if ck.qat is not None:
        assert keep is None and bits_att == bits_ffn == bits_emb == 4, "QAT scales are for the all-int4 student"
    emb_parts = add_mat_parts("emb", emb, bits_emb, Xc, part_rows, qat_s("emb", emb))
    for l in range(ck.n_layers):
        blob.add(f"l{l}_rms_att", q16(ck.rms_att[l]))
        add_mat(f"l{l}_wq", ck.wq[l] * qfold, bits_att, cx(l, "att_in"), qat_s(f"l{l}_wq", ck.wq[l], qfold))
        add_mat(f"l{l}_wk", ck.wk[l], bits_att, cx(l, "att_in"), qat_s(f"l{l}_wk", ck.wk[l]))
        add_mat(f"l{l}_wv", ck.wv[l], bits_att, cx(l, "att_in"), qat_s(f"l{l}_wv", ck.wv[l]))
        add_mat(f"l{l}_wo", ck.wo[l], bits_att, cx(l, "wo_in"), qat_s(f"l{l}_wo", ck.wo[l]))
        blob.add(f"l{l}_rms_ffn", q16(ck.rms_ffn[l]))
        add_mat(f"l{l}_w1", w1[l], bits_ffn, cx(l, "ffn_in"), qat_s(f"l{l}_w1", w1[l]))
        add_mat(f"l{l}_w2", w2[l], bits_ffn, cx(l, "w2_in"), qat_s(f"l{l}_w2", w2[l]))
        add_mat(f"l{l}_w3", w3[l], bits_ffn, cx(l, "ffn_in"), qat_s(f"l{l}_w3", w3[l]))
    blob.add("rms_final", q16(ck.rms_final))
    if not ck.shared:
        cls = np.zeros((vrows, ck.dim), np.float32)
        cls[:vocab] = ck.wcls[tok.new2old]
        add_mat_parts("cls", cls, bits_emb, None if calib is None else calib["cls_in"], part_rows)

    # RoPE table in Q30 from the float32 angles run.c would use (rounded once, in float64).
    ang = rope_angles(ck.seq, hd, ck.rope_theta).astype(np.float64)
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
    for k, v in (("DIM", ck.dim), ("HIDDEN", hid), ("LAYERS", ck.n_layers),
                 ("HEADS", ck.n_heads), ("KV_HEADS", ck.n_kv_heads), ("HEAD_DIM", hd),
                 ("KV_DIM", ck.kv_dim), ("VOCAB", vocab), ("VOCAB_ROWS", vrows), ("VOCAB_FULL", ck.vocab), ("SEQ", ck.seq), ("GS_D", gs_d), ("GS_H", gs_h),
                 ("BITS_ATT", bits_att), ("BITS_FFN", bits_ffn), ("BITS_EMB", bits_emb),
                 ("GS_EMB", g4_d if bits_emb == 4 else gs_d),
                 ("GS_ATT", g4_d if bits_att == 4 else gs_d), ("GS_FFN_D", g4_d if bits_ffn == 4 else gs_d),
                 ("GS_FFN_H", g4_h if bits_ffn == 4 else gs_h),
                 ("SHARED_CLS", int(ck.shared)), ("BLOB_BYTES", len(blob.buf)),
                 ("MAGIC", MAGIC), ("VERSION", VERSION)):
        w(f"#define NSLM_{k} {v}")
    w("#define NSLM_HDR_BYTES 32")
    w(f"#define NSLM_EMB_PARTS {emb_parts}")
    w(f"#define NSLM_EMB_PART_ROWS {part_rows}")
    w("#define NSLM_FOR_EMB_PARTS(X) " + " ".join(f"X({p})" for p in range(emb_parts)))
    # segment table: file offsets where each separately allocated piece starts (tensor boundaries)
    segs, cur, size = [0], 0, ALIGN
    for name, off, n in blob.tensors:
        fo = off + 32
        if fo - cur + n > SEG_MAX and fo > cur:
            segs.append(fo)
            cur = fo
    total = 32 + len(blob.buf)
    w(f"#define NSLM_NSEG {len(segs)}")
    w("static const uint32_t nslm_seg_off[NSLM_NSEG + 1] = {" + ",".join(f"{x}u" for x in segs + [total]) + "};")
    biggest = max(b - a for a, b in zip(segs, segs[1:] + [total]))
    w(f"#define NSLM_SEG_MAX_BYTES {biggest}u")
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
    return {"gs": (gs_d, gs_h), "blob_bytes": len(blob.buf), "vocab": vocab}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--vocab-usage", help="token counts over the training corpus (tools/vocab_usage.py); "
                    "tokens never seen are compiled out of the embedding and classifier")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--bits-att", type=int, choices=(4, 8), default=8)
    ap.add_argument("--bits-ffn", type=int, choices=(4, 8), default=8)
    ap.add_argument("--bits-emb", type=int, choices=(4, 8), default=8, help="embedding / classifier")
    ap.add_argument("--gs4", type=int, default=32, help="int4 group size (multiple of 16)")
    ap.add_argument("--calib", help="text file for GPTQ calibration of int4 matrices")
    ap.add_argument("--calib-stories", type=int, default=16)
    ap.add_argument("--seq", type=int, help="compile a shorter context than the checkpoint's (saves KV memory)")
    a = ap.parse_args(argv)
    ck = Ckpt(a.ckpt)
    if a.tokenizer.endswith(".json"):  # HF byte-level BPE; the checkpoint's rows are its kept tokens
        tok = BLModelTok(a.tokenizer, ck.meta["keep"])
        assert len(tok.new2old) == ck.vocab, (len(tok.new2old), ck.vocab)
    else:
        tok = Tokenizer(a.tokenizer, ck.vocab)
    keep = None
    if a.vocab_usage:
        counts = np.load(a.vocab_usage)
        assert len(counts) == ck.vocab
        keep = sorted(set(np.nonzero(counts >= a.min_count)[0].tolist()) | {0, 1, 2} | set(range(3, 259)))
    if a.seq:
        ck.seq = min(ck.seq, a.seq)
    if ck.qat is not None:  # the group size the scales were learned with
        a.gs4 = ck.dim // ck.qat["l0_wq"].shape[1]
    calib = None
    if a.calib and 4 in (a.bits_att, a.bits_ffn, a.bits_emb):
        import gptq
        sts = [x.strip() for x in open(a.calib, encoding="utf-8").read().split("<|endoftext|>") if x.strip()]
        full = Tokenizer(a.tokenizer, ck.vocab)  # calibration runs the float model on full-vocab ids
        calib = gptq.calibrate(ck, [full.encode(x) for x in sts[: a.calib_stories]])
    info = compile_model(ck, tok, a.out, a.gs, keep, a.bits_att, a.bits_ffn, a.gs4, calib, a.bits_emb)
    print(f"nslmc: dim={ck.dim} hidden={ck.hidden} layers={ck.n_layers} heads={ck.n_heads}/"
          f"{ck.n_kv_heads} vocab={info['vocab']}/{ck.vocab} seq={ck.seq} gs={info['gs']} "
          f"blob={info['blob_bytes'] / 2**20:.2f} MiB -> {a.out}")


if __name__ == "__main__":
    main()
