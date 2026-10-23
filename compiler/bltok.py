"""Byte-level BPE (GPT-2 family, as used by SmolLM2) for the AOT compiler and as the reference that the C
encoder in runtime/nslm_tok.c must match.

Pretokenizer: HF `Digits(individual_digits)` then the GPT-2 regex
    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
written as a hand scanner over bytes (no regex engine on the calculator). Bytes >= 0x80 count as letters:
exact for ASCII, which is all the calculator keyboard produces (tests/test_bltok.py checks against HF).
Each pre-token is mapped to byte symbols and merged by merge rank (lowest first). Tokens outside the kept
model vocabulary are then split along their own merges (train/common.py Vocab.pieces), like in training.

Image (tok.nslt, magic "NSLB"), all little endian:
    u32 magic, n_full, n_merges, n_model
    i32 byte_tok[256]                  byte -> token id of its byte symbol (-1 if absent)
    u32 merge_key[n_merges]            (left << 16) | right, sorted ascending
    u16 merge_rank[n_merges], u16 merge_out[n_merges]   (padded to 4 bytes)
    u32 piece_off[n_full + 1]; u16 piece[...]            full id -> model ids (padded to 4 bytes)
    u32 str_off[n_model + 1]; u8 str[...]                model id -> decoded bytes
"""
import json
import struct

import numpy as np


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _cls(c):
    if c in b" \t\n\r\x0b\x0c":
        return "s"
    if 48 <= c <= 57:
        return "n"
    if 65 <= c <= 90 or 97 <= c <= 122 or c >= 0x80:
        return "l"
    return "o"


CONTRACTIONS = (b"'s", b"'t", b"'re", b"'ve", b"'m", b"'ll", b"'d")


def pretokenize(text, lookahead_across_digits=False):
    """bytes -> list of byte strings.

    HF tokenizers builds disagree on one corner: whether `\\s+(?!\\S)` at the end of a Digits segment
    sees the digit that follows ("x  1": "  " or " ", " "). We fix the segment-local reading (the regex
    applied to each segment on its own); training tokenizes with this module too (train/data.py), so the
    device matches training whatever HF build is installed."""
    out = []
    segs, st = [], 0
    for i, c in enumerate(text):  # Digits(individual_digits=True): every digit is its own segment
        if 48 <= c <= 57:
            if st < i:
                segs.append((st, i))
            segs.append((i, i + 1))
            st = i + 1
    if st < len(text):
        segs.append((st, len(text)))
    s = text
    for i, n in segs:
        while i < n:
            c = _cls(s[i])
            m = next((k for k in CONTRACTIONS if s.startswith(k, i) and i + len(k) <= n), None) \
                if s[i] == 39 else None
            if m:
                j = i + len(m)
            elif c in "lno" or (s[i] == 32 and i + 1 < n and _cls(s[i + 1]) in "lno"):
                k = i + 1 if c == "s" else i
                t = _cls(s[k])
                j = k + 1
                while j < n and _cls(s[j]) == t:
                    j += 1
            else:  # whitespace run
                j = i
                while j < n and _cls(s[j]) == "s":
                    j += 1
                # \s+(?!\S): leave the last space to start the next token
                lim = len(s) if lookahead_across_digits else n
                if j < lim and _cls(s[j]) != "s" and j - i >= 2:
                    j -= 1
            out.append(s[i:j])
            i = j
    return out


class ByteLevelBPE:
    def __init__(self, tokenizer_json, keep=None):
        d = json.load(open(tokenizer_json))
        m = d["model"]
        self.t2i = m["vocab"]
        self.n_full = len(self.t2i)
        self.i2t = [None] * self.n_full
        for t, i in self.t2i.items():
            self.i2t[i] = t
        b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in b2u.items()}
        self.byte_tok = [self.t2i.get(b2u[b], -1) for b in range(256)]
        self.merges = {}
        self.split = {}
        for r, mg in enumerate(m["merges"]):
            a, b = mg.split(" ") if isinstance(mg, str) else mg
            ia, ib, io = self.t2i[a], self.t2i[b], self.t2i[a + b]
            self.merges.setdefault((ia, ib), (r, io))
            self.split.setdefault(io, (ia, ib))
        self._bpe_memo = {}
        self.set_keep(keep)

    def set_keep(self, keep):
        keep = np.arange(self.n_full) if keep is None else np.asarray(sorted(set(int(k) for k in keep)))
        self.keep = keep
        self.old2new = np.full(self.n_full, -1, dtype=np.int64)
        self.old2new[keep] = np.arange(len(keep))
        self._memo = {}

    def pieces(self, t):
        if self.old2new[t] >= 0:
            return (t,)
        r = self._memo.get(t)
        if r is None:
            if t in self.split:
                a, b = self.split[t]
                r = self.pieces(a) + self.pieces(b)
            else:
                r = sum((self.pieces(self.t2i[ch]) for ch in self.i2t[t]), ())
            self._memo[t] = r
        return r

    def bpe(self, word):
        r = self._bpe_memo.get(word)
        if r is None:
            r = self._bpe_memo[word] = self._bpe(word)
        return r

    def _bpe(self, word):
        toks = [self.byte_tok[c] for c in word]
        while len(toks) > 1:
            best, bi = None, -1
            for i in range(len(toks) - 1):
                v = self.merges.get((toks[i], toks[i + 1]))
                if v is not None and (best is None or v[0] < best[0]):
                    best, bi = v, i
            if best is None:
                break
            toks[bi:bi + 2] = [best[1]]
        return toks

    def encode_full(self, text, lookahead_across_digits=False):
        """teacher-space ids of plain text (no special tokens)"""
        if isinstance(text, str):
            text = text.encode("utf-8")
        return [t for w in pretokenize(text, lookahead_across_digits) for t in self.bpe(w)]

    def encode(self, text):
        """model ids"""
        return [int(self.old2new[p]) for t in self.encode_full(text) for p in self.pieces(t)]

    def token_bytes(self, t):
        return bytes(self.u2b[ch] for ch in self.i2t[t]) if all(ch in self.u2b for ch in self.i2t[t]) \
            else self.i2t[t].encode()

    def write_nslt(self, path):
        assert self.n_full < 65536 and len(self.keep) < 65536
        keys = sorted(self.merges)
        blob = bytearray(struct.pack("<4I", 0x424C534E, self.n_full, len(keys), len(self.keep)))
        blob += struct.pack("<256i", *self.byte_tok)
        blob += struct.pack(f"<{len(keys)}I", *[(a << 16) | b for a, b in keys])
        pad = lambda: blob.extend(b"\0" * (-len(blob) % 4))  # noqa: E731
        blob += struct.pack(f"<{len(keys)}H", *[self.merges[k][0] for k in keys])
        blob += struct.pack(f"<{len(keys)}H", *[self.merges[k][1] for k in keys])
        pad()
        offs, flat = [0], []
        for t in range(self.n_full):
            flat += [int(self.old2new[p]) for p in self.pieces(t)]
            offs.append(len(flat))
        blob += struct.pack(f"<{len(offs)}I", *offs)
        blob += struct.pack(f"<{len(flat)}H", *flat)
        pad()
        offs, strs = [0], bytearray()
        for t in self.keep:
            strs += self.token_bytes(int(t))
            offs.append(len(strs))
        blob += struct.pack(f"<{len(offs)}I", *offs)
        blob += strs
        with open(path, "wb") as f:
            f.write(blob)
