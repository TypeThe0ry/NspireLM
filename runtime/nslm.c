/* NspireLM integer-only forward pass. See nslm.h for the number formats. */
#include "nslm.h"

#include <string.h>

/* ------------------------------------------------------------------ helpers */

static inline int bitlen32(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline int bitlen64(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }

/* Round-half-up arithmetic shift right by s (s may be negative: shift left). */
static inline int64_t rshr(int64_t v, int s) {
    if (s > 0) return (v + ((int64_t)1 << (s - 1))) >> s;
    return v * ((int64_t)1 << -s);
}

static inline int32_t sat32(int64_t v) {
    return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v;
}

static inline int16_t sat16(int64_t v) {
    return v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)v;
}

static inline uint32_t uabs32(int32_t v) { return v < 0 ? 0u - (uint32_t)v : (uint32_t)v; }

static uint64_t isqrt64(uint64_t v) {
    uint64_t r = 0, bit = (uint64_t)1 << 62;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= r + bit) {
            v -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return r;
}

/* fx32 vector -> act16; returns the exponent E. */
static int quant_act(int16_t *a, const int32_t *x, int n) {
    uint32_t mx = 0;
    for (int i = 0; i < n; i++) mx |= uabs32(x[i]);
    if (!mx) {
        memset(a, 0, (size_t)n * sizeof *a);
        return 16;
    }
    int s = bitlen32(mx) - 15;
    for (int i = 0; i < n; i++) a[i] = sat16(rshr(x[i], s));
    return 16 - s;
}

/* RMSNorm(x) * g -> act16; returns E. */
static int rmsnorm_act(int16_t *a, const int32_t *x, const int32_t *g, int n, int64_t *tmp) {
    uint32_t mx = 0;
    for (int i = 0; i < n; i++) mx |= uabs32(x[i]);
    if (!mx) {
        memset(a, 0, (size_t)n * sizeof *a);
        return 0;
    }
    int k = bitlen32(mx) - 15;
    int64_t ss = 0;
    for (int i = 0; i < n; i++) {
        int64_t v = rshr(x[i], k);
        ss += v * v;
    }
    /* eps = 1e-5 in the scaled domain: 1e-5 * 2^32 = 42950, then divided by 2^(2k) */
    uint64_t mean = (uint64_t)ss / (uint64_t)n + (uint64_t)rshr(42950, 2 * k);
    uint64_t r = isqrt64(mean << 20);              /* sqrt(mean) * 2^10 */
    int64_t inv = (int64_t)(((uint64_t)1 << 40) / (r ? r : 1)); /* 2^30 / sqrt(mean) */
    uint64_t umx = 0;
    for (int i = 0; i < n; i++) {
        int64_t t = rshr(rshr(x[i], k) * inv, 15); /* real * 2^15 */
        int64_t u = t * g[i];                      /* real * 2^31 */
        tmp[i] = u;
        umx |= (uint64_t)(u < 0 ? -u : u);
    }
    if (!umx) {
        memset(a, 0, (size_t)n * sizeof *a);
        return 0;
    }
    int s = bitlen64(umx) - 15;
    for (int i = 0; i < n; i++) a[i] = sat16(rshr(tmp[i], s));
    return 31 - s;
}

/* exp2(d / 65536) for d <= 0, in Q30. */
static inline int32_t exp2_q30(int32_t d) {
    uint32_t nd = (uint32_t)(-(int64_t)d);
    uint32_t n = nd >> 16;
    if (n >= 31) return 0;
    uint32_t f = nd & 0xFFFF, idx = f >> 8, fr = f & 0xFF;
    int32_t a = nslm_exp2_lut[idx], b = nslm_exp2_lut[idx + 1];
    int32_t e = a - (int32_t)(((int64_t)(a - b) * fr + 128) >> 8);
    return e >> n;
}

static inline int32_t silu_mul(int32_t x, int32_t h3) {
    int64_t sig;
    if (x <= -(16 << 16)) {
        sig = nslm_sig_lut[0];
    } else if (x >= (16 << 16)) {
        sig = nslm_sig_lut[1024];
    } else {
        uint32_t t = (uint32_t)(x + (16 << 16));
        uint32_t idx = t >> 11, fr = t & 0x7FF;
        int32_t a = nslm_sig_lut[idx], b = nslm_sig_lut[idx + 1];
        sig = a + (((int64_t)(b - a) * fr + 1024) >> 11);
    }
    int64_t silu = rshr((int64_t)x * sig, 30);
    return sat32(rshr(silu * h3, 16));
}

/* Embedding row lookup -> fx32, same scale math as the matmul kernels (x4 int8 or x4n int4). */
static void embed(int32_t *x, const nslm_q8 *w, int tok) {
    const int ng = NSLM_DIM / NSLM_GS_EMB;
    const uint16_t *mb = w->m + (size_t)(tok >> 2) * 4 * ng;
    int j = tok & 3, sh = w->e[tok] - 16;
#if NSLM_BITS_EMB == 4
    const uint32_t *blk = (const uint32_t *)(const void *)(w->q + (size_t)(tok >> 2) * 32 * (NSLM_DIM / 16));
    for (int i = 0; i < NSLM_DIM; i++) {
        uint32_t word = blk[(i >> 4) * 8 + ((i >> 3) & 1) * 4 + j];
        int32_t q = (int32_t)((word >> (4 * (i & 7))) & 15) - 8;
        x[i] = sat32(rshr(q * (int32_t)mb[(i / NSLM_GS_EMB) * 4 + j], sh));
    }
#else
    const uint8_t *blk = w->q + (size_t)(tok >> 2) * 32 * (NSLM_DIM / 8);
    for (int i = 0; i < NSLM_DIM; i++) {
        int32_t q = (int32_t)blk[(i >> 3) * 32 + j * 8 + (i & 7)] - 128;
        x[i] = sat32(rshr(q * (int32_t)mb[(i / NSLM_GS_EMB) * 4 + j], sh));
    }
#endif
}

static void rope(int32_t *v, int nh, const int32_t *tab) {
    for (int h = 0; h < nh; h++) {
        int32_t *p = v + h * NSLM_HEAD_DIM;
        for (int i = 0; i < NSLM_HEAD_DIM / 2; i++) {
            int64_t c = tab[2 * i], s = tab[2 * i + 1];
            int64_t a = p[2 * i], b = p[2 * i + 1];
            p[2 * i] = sat32(rshr(a * c - b * s, 30));
            p[2 * i + 1] = sat32(rshr(a * s + b * c, 30));
        }
    }
}

/* fx32 head vector -> int8 + Q24 scale (scale = maxabs / 127). */
static int32_t quant_kv(int8_t *dst, const int32_t *src) {
    uint32_t mx = 0;
    for (int i = 0; i < NSLM_HEAD_DIM; i++) {
        uint32_t a = uabs32(src[i]);
        if (a > mx) mx = a;
    }
    if (!mx) {
        memset(dst, 0, NSLM_HEAD_DIM);
        return 0;
    }
    int64_t r = ((int64_t)127 << 32) / mx;
    for (int i = 0; i < NSLM_HEAD_DIM; i++) {
        int64_t q = rshr((int64_t)src[i] * r, 32);
        dst[i] = (int8_t)(q > 127 ? 127 : q < -127 ? -127 : q);
    }
    return sat32(((int64_t)mx * 256 + 63) / 127);
}

/* ------------------------------------------------------------------ binding */

/* The image may live in NSLM_NSEG separately allocated segments (nslm_seg_off, file offsets). */
static const uint8_t *const *g_seg;

static const void *at(uint32_t off) {
    uint32_t fo = NSLM_HDR_BYTES + off;
    int lo = 0, hi = NSLM_NSEG - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (nslm_seg_off[mid] <= fo) lo = mid;
        else hi = mid - 1;
    }
    return g_seg[lo] + (fo - nslm_seg_off[lo]);
}

static nslm_q8 q8_at(uint32_t oq, uint32_t om, uint32_t oe) {
    nslm_q8 w = {(const uint8_t *)at(oq), (const uint16_t *)at(om), (const int8_t *)at(oe)};
    return w;
}

#define PART(name, p) q8_at(NSLM_OFF_##name##_p##p##_q, NSLM_OFF_##name##_p##p##_m, NSLM_OFF_##name##_p##p##_e)
#define LQ8(l, name)                                                                           \
    q8_at(NSLM_OFF_l0_##name##_q + (l) * NSLM_OFF_L_STRIDE,                                    \
          NSLM_OFF_l0_##name##_m + (l) * NSLM_OFF_L_STRIDE,                                    \
          NSLM_OFF_l0_##name##_e + (l) * NSLM_OFF_L_STRIDE)

#if NSLM_SHARED_CLS
#define CLSPART(p) PART(emb, p)
#else
#define CLSPART(p) PART(cls, p)
#endif
static void bind_parts(nslm_q8 *dst, int cls) {
#define BP(p) dst[p] = cls ? CLSPART(p) : PART(emb, p);
    NSLM_FOR_EMB_PARTS(BP) /* generated list p0..p(N-1) */
#undef BP
}

int nslm_bind_segments(nslm_model *m, const void *const *segs, nslm_state *state) {
    const uint32_t *h = (const uint32_t *)segs[0];
    for (int k = 0; k < NSLM_NSEG; k++)
        if ((uintptr_t)segs[k] & 31) return -1; /* every segment line aligned */
    if (h[0] != NSLM_MAGIC || h[1] != NSLM_VERSION || h[2] != NSLM_BLOB_BYTES) return -2;
    g_seg = (const uint8_t *const *)segs;
    bind_parts(m->emb, 0);
    bind_parts(m->cls, 1);
    for (int l = 0; l < NSLM_LAYERS; l++) {
        nslm_layer *L = &m->layer[l];
        L->rms_att = (const int32_t *)at(NSLM_OFF_l0_rms_att + l * NSLM_OFF_L_STRIDE);
        L->rms_ffn = (const int32_t *)at(NSLM_OFF_l0_rms_ffn + l * NSLM_OFF_L_STRIDE);
        L->wq = LQ8(l, wq);
        L->wk = LQ8(l, wk);
        L->wv = LQ8(l, wv);
        L->wo = LQ8(l, wo);
        L->w1 = LQ8(l, w1);
        L->w2 = LQ8(l, w2);
        L->w3 = LQ8(l, w3);
    }
    m->rms_final = (const int32_t *)at(NSLM_OFF_rms_final);
    m->rope = (const int32_t *)at(NSLM_OFF_rope);
    m->s = state;
    memset(state, 0, sizeof *state);
    return 0;
}

int nslm_bind(nslm_model *m, const void *image, size_t len, nslm_state *state) {
    static const void *segs[NSLM_NSEG];
    if (len < NSLM_HDR_BYTES + NSLM_BLOB_BYTES) return -1;
    for (int k = 0; k < NSLM_NSEG; k++) segs[k] = (const uint8_t *)image + nslm_seg_off[k];
    return nslm_bind_segments(m, segs, state);
}

/* ------------------------------------------------------------------ forward */

static void matmul(nslm_state *s, int32_t *out, const nslm_q8 *w, int ea, int rows, int cols, int gs, int bits) {
    for (int g = 0; g < cols / gs; g++) {
        int32_t t = 0;
        for (int i = g * gs; i < (g + 1) * gs; i++) t += s->a[i];
        s->asum[g] = t;
    }
    if (bits == 4)
        nslm_matmul_q4(out, w, s->a, s->asum, ea, rows, cols, gs);
    else
        nslm_matmul_q8(out, w, s->a, s->asum, ea, rows, cols, gs);
}

static void attention(nslm_state *s, int l, int pos) {
    const int kvmul = NSLM_HEADS / NSLM_KV_HEADS;
    for (int h = 0; h < NSLM_HEADS; h++) {
        int kh = h / kvmul;
        int eq = quant_act(s->qh, s->q + h * NSLM_HEAD_DIM, NSLM_HEAD_DIM);
        int32_t zmax = INT32_MIN;
        for (int t = 0; t <= pos; t++) {
            const int8_t *kq = s->kc[l][t] + kh * NSLM_HEAD_DIM;
            int32_t raw = 0;
            for (int i = 0; i < NSLM_HEAD_DIM; i++) raw += (int32_t)s->qh[i] * kq[i];
            int32_t z = sat32(rshr((int64_t)raw * s->ks[l][t][kh], eq + 8));
            s->z[t] = z;
            if (z > zmax) zmax = z;
        }
        int64_t sum = 0;
        uint64_t wmax = 0;
        for (int t = 0; t <= pos; t++) {
            int32_t p = exp2_q30(sat32((int64_t)s->z[t] - zmax));
            sum += p;
            int64_t w = (int64_t)p * s->vs[l][t][kh];
            s->w[t] = w;
            if ((uint64_t)w > wmax) wmax = (uint64_t)w;
        }
        int32_t *out = s->xb + h * NSLM_HEAD_DIM;
        if (!wmax) {
            memset(out, 0, NSLM_HEAD_DIM * sizeof *out);
            continue;
        }
        int sh = bitlen64(wmax) - 13;
        for (int t = 0; t <= pos; t++) s->w16[t] = (int16_t)rshr(s->w[t], sh);
        int32_t acc[NSLM_HEAD_DIM];
        memset(acc, 0, sizeof acc);
        for (int t = 0; t <= pos; t++) {
            const int8_t *vq = s->vc[l][t] + kh * NSLM_HEAD_DIM;
            int32_t w = s->w16[t];
            if (!w) continue;
            for (int i = 0; i < NSLM_HEAD_DIM; i++) acc[i] += w * vq[i];
        }
        int64_t recip = (int64_t)(((uint64_t)1 << 62) / (uint64_t)sum);
        for (int i = 0; i < NSLM_HEAD_DIM; i++) out[i] = sat32(rshr((int64_t)acc[i] * recip, 70 - sh));
    }
}

void nslm_forward(nslm_model *m, int token, int pos) {
    nslm_state *s = m->s;
    static int64_t tmp[NSLM_MAXV];
    if (token < 0 || token >= NSLM_VOCAB) token = 0; /* never index outside the compiled vocabulary */
    if (pos < 0) pos = 0;
    if (pos >= NSLM_SEQ) pos = NSLM_SEQ - 1;
    embed(s->x, &m->emb[token / NSLM_EMB_PART_ROWS], token % NSLM_EMB_PART_ROWS);
    const int32_t *rope_tab = m->rope + (size_t)pos * NSLM_HEAD_DIM;
    for (int l = 0; l < NSLM_LAYERS; l++) {
        const nslm_layer *L = &m->layer[l];
        int ea = rmsnorm_act(s->a, s->x, L->rms_att, NSLM_DIM, tmp);
        matmul(s, s->q, &L->wq, ea, NSLM_DIM, NSLM_DIM, NSLM_GS_ATT, NSLM_BITS_ATT);
        matmul(s, s->k, &L->wk, ea, NSLM_KV_DIM, NSLM_DIM, NSLM_GS_ATT, NSLM_BITS_ATT);
        matmul(s, s->v, &L->wv, ea, NSLM_KV_DIM, NSLM_DIM, NSLM_GS_ATT, NSLM_BITS_ATT);
        rope(s->q, NSLM_HEADS, rope_tab);
        rope(s->k, NSLM_KV_HEADS, rope_tab);
        for (int h = 0; h < NSLM_KV_HEADS; h++) {
            s->ks[l][pos][h] = quant_kv(s->kc[l][pos] + h * NSLM_HEAD_DIM, s->k + h * NSLM_HEAD_DIM);
            s->vs[l][pos][h] = quant_kv(s->vc[l][pos] + h * NSLM_HEAD_DIM, s->v + h * NSLM_HEAD_DIM);
        }
        attention(s, l, pos);
        ea = quant_act(s->a, s->xb, NSLM_DIM);
        matmul(s, s->xb, &L->wo, ea, NSLM_DIM, NSLM_DIM, NSLM_GS_ATT, NSLM_BITS_ATT);
        for (int i = 0; i < NSLM_DIM; i++) s->x[i] = sat32((int64_t)s->x[i] + s->xb[i]);

        ea = rmsnorm_act(s->a, s->x, L->rms_ffn, NSLM_DIM, tmp);
        matmul(s, s->h1, &L->w1, ea, NSLM_HIDDEN, NSLM_DIM, NSLM_GS_FFN_D, NSLM_BITS_FFN);
        matmul(s, s->h3, &L->w3, ea, NSLM_HIDDEN, NSLM_DIM, NSLM_GS_FFN_D, NSLM_BITS_FFN);
        for (int i = 0; i < NSLM_HIDDEN; i++) s->h1[i] = silu_mul(s->h1[i], s->h3[i]);
        ea = quant_act(s->a, s->h1, NSLM_HIDDEN);
        matmul(s, s->xb, &L->w2, ea, NSLM_DIM, NSLM_HIDDEN, NSLM_GS_FFN_H, NSLM_BITS_FFN);
        for (int i = 0; i < NSLM_DIM; i++) s->x[i] = sat32((int64_t)s->x[i] + s->xb[i]);
    }
    int ea = rmsnorm_act(s->a, s->x, m->rms_final, NSLM_DIM, tmp);
    for (int p = 0; p < NSLM_EMB_PARTS; p++) { /* the classifier, one separately allocated part at a time */
        int r0 = p * NSLM_EMB_PART_ROWS;
        int rows = NSLM_VOCAB_ROWS - r0 < NSLM_EMB_PART_ROWS ? NSLM_VOCAB_ROWS - r0 : NSLM_EMB_PART_ROWS;
        matmul(s, s->logits + r0, &m->cls[p], ea, rows, NSLM_DIM, NSLM_GS_EMB, NSLM_BITS_EMB);
    }
}

/* ------------------------------------------------------------------ sampling */

int nslm_argmax(const int32_t *logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

int nslm_sample(const int32_t *logits, int n, int temp_q8, uint32_t *rng) {
    if (temp_q8 <= 0) return nslm_argmax(logits, n);
    int32_t mx = logits[nslm_argmax(logits, n)];
    /* p_i = 2^((l_i - mx) * log2(e) / T); log2(e) in Q16 = 94548 */
    int64_t k = ((int64_t)94548 << 8) / temp_q8; /* Q16 multiplier */
    uint64_t sum = 0;
    for (int i = 0; i < n; i++) sum += (uint32_t)exp2_q30(sat32(rshr((int64_t)(logits[i] - mx) * k, 16)));
    uint32_t x = *rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *rng = x;
    uint64_t target = ((sum >> 16) * (uint64_t)x) >> 16; /* sum < 2^45 */
    uint64_t acc = 0;
    for (int i = 0; i < n; i++) {
        acc += (uint32_t)exp2_q30(sat32(rshr((int64_t)(logits[i] - mx) * k, 16)));
        if (acc > target) return i;
    }
    return n - 1;
}

int nslm_decode(int32_t *logits, int n, const nslm_policy *p, const int *hist, int nhist, uint32_t *rng) {
    if (p->rep_q16 && nhist > 0) {
        int w = p->window < nhist ? p->window : nhist;
        if (w > NSLM_REP_MAX) w = NSLM_REP_MAX;
        const int *h = hist + nhist - w;
        for (int i = 0; i < w; i++) {
            int t = h[i], seen = 0;
            for (int j = 0; j < i && !seen; j++) seen = h[j] == t;
            if (!seen && t >= 0 && t < n) logits[t] = sat32((int64_t)logits[t] - p->rep_q16);
        }
    }
    if (p->temp_q8 <= 0) return nslm_argmax(logits, n);
    int32_t mx = logits[nslm_argmax(logits, n)];
    int64_t k = ((int64_t)94548 << 8) / p->temp_q8; /* log2(e) / T, Q16 */
    /* z = (l - max) * log2(e) / T in Q16; min-p keeps z >= minp_q16 */
    int32_t zmin = p->minp_q16 < 0 ? p->minp_q16 : INT32_MIN;
    uint64_t sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t z = sat32(rshr((int64_t)(logits[i] - mx) * k, 16));
        if (z >= zmin) sum += (uint32_t)exp2_q30(z);
    }
    uint32_t x = *rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *rng = x;
    uint64_t target = ((sum >> 16) * (uint64_t)x) >> 16;
    uint64_t acc = 0;
    int last = 0;
    for (int i = 0; i < n; i++) {
        int32_t z = sat32(rshr((int64_t)(logits[i] - mx) * k, 16));
        if (z < zmin) continue;
        last = i;
        acc += (uint32_t)exp2_q30(z);
        if (acc > target) return i;
    }
    return last;
}
