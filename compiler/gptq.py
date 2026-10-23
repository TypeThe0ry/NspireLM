"""GPTQ (Frantar et al., 2022) for nslmc: Hessian-aware rounding of group-wise int4/int8 weights.

calibrate(ck, token_lists) records the input of every matmul over calibration text (float model).
gptq_quant(W, X, bits, gs) returns integer codes q and per-group scales s with W ~= q * s,
choosing roundings that minimise the layer's output error on X rather than the weight error."""
import numpy as np

import nslmc


def calibrate(ck, token_lists):
    rec = {}

    class Rec(nslmc.FloatModel):
        def forward(self, tok, pos):
            ck = self.ck
            hd, nh, nkv = ck.head_dim, ck.n_heads, ck.n_kv_heads
            x = ck.tok_emb[tok].astype(np.float32).copy()
            for l in range(ck.n_layers):
                xb = self.rmsnorm(x, ck.rms_att[l])
                rec.setdefault(f"l{l}_att_in", []).append(xb)
                q, k, v = ck.wq[l] @ xb, ck.wk[l] @ xb, ck.wv[l] @ xb
                q, k = self.rope(q, pos, nh), self.rope(k, pos, nkv)
                self.kc[l, pos], self.vc[l, pos] = k, v
                out = np.zeros(ck.dim, np.float32)
                kvmul = nh // nkv
                for h in range(nh):
                    kh = h // kvmul
                    K = self.kc[l, :pos + 1, kh * hd:(kh + 1) * hd]
                    V = self.vc[l, :pos + 1, kh * hd:(kh + 1) * hd]
                    att = (K @ q[h * hd:(h + 1) * hd]) / np.float32(np.sqrt(hd))
                    att = np.exp(att - att.max())
                    out[h * hd:(h + 1) * hd] = (att / att.sum()) @ V
                rec.setdefault(f"l{l}_wo_in", []).append(out)
                x = x + ck.wo[l] @ out
                xb = self.rmsnorm(x, ck.rms_ffn[l])
                rec.setdefault(f"l{l}_ffn_in", []).append(xb)
                h1, h3 = ck.w1[l] @ xb, ck.w3[l] @ xb
                g = (h1 * (1.0 / (1.0 + np.exp(-h1)))) * h3
                rec.setdefault(f"l{l}_w2_in", []).append(g)
                x = x + ck.w2[l] @ g
            x = self.rmsnorm(x, ck.rms_final)
            rec.setdefault("cls_in", []).append(x)
            return ck.wcls @ x

    m = Rec(ck)
    for ids in token_lists:
        m.reset()
        for p, t in enumerate(ids[: ck.seq]):
            m.forward(t, p)
    return {k: np.stack(v).astype(np.float64) for k, v in rec.items()}


def _group_scale(wg, qmax, qmin, clip):
    """MSE-optimal clipped scale per row for one column group (rows x gs)."""
    amax = np.abs(wg).max(axis=1)
    amax = np.where(amax == 0, 1e-12, amax)
    best_s, best_e = None, None
    for k in clip:
        s = amax * k / qmax
        q = np.clip(np.round(wg / s[:, None]), qmin, qmax)
        e = ((q * s[:, None] - wg) ** 2).sum(axis=1)
        if best_s is None:
            best_s, best_e = s, e
        else:
            better = e < best_e
            best_s, best_e = np.where(better, s, best_s), np.where(better, e, best_e)
    return best_s


def gptq_quant(W, X, bits, gs, damp=0.01, clip=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75)):
    W = W.astype(np.float64).copy()
    r, c = W.shape
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1 if bits == 4 else -qmax
    H = X.T @ X / max(len(X), 1)
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += np.eye(c) * damp * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T  # upper triangular
    Q = np.zeros((r, c), np.int64)
    S = np.zeros((r, c // gs))
    for g in range(c // gs):
        i0, i1 = g * gs, (g + 1) * gs
        s = _group_scale(W[:, i0:i1], qmax, qmin, clip)
        S[:, g] = s
        for i in range(i0, i1):
            w = W[:, i]
            q = np.clip(np.round(w / s), qmin, qmax)
            Q[:, i] = q
            err = (w - q * s) / Hinv[i, i]
            W[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])
    return Q, S
