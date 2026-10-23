/* Integer-only port of llama2.c's BPE tokenizer (merge scores replaced by ranks). */
#include <string.h>

#include "nslm.h"

static int bind_bl(nslm_tok *t, const uint32_t *h, size_t len) {
    t->kind = NSLM_TOK_BL;
    t->n = (int)h[1];
    t->n_merges = (int)h[2];
    t->n_model = (int)h[3];
    t->max_len = 0;
    if (t->n <= 0 || t->n > 65536 || t->n_model <= 0 || t->n_model > t->n || t->n_merges < 0) return -1;
    size_t o = 16; /* byte offsets, checked against len before every use */
#define TAKE(dst, type, count, align)                                         \
    do {                                                                      \
        if (o + (size_t)(count) * sizeof(type) > len) return -1;              \
        dst = (const type *)(const void *)((const uint8_t *)h + o);           \
        o += (size_t)(count) * sizeof(type);                                  \
        o = (o + (align)-1) & ~(size_t)((align)-1);                           \
    } while (0)
    TAKE(t->byte_tok, int32_t, 256, 4);
    TAKE(t->mkey, uint32_t, t->n_merges, 4);
    TAKE(t->mrank, uint16_t, t->n_merges, 2);
    TAKE(t->mout, uint16_t, t->n_merges, 4);
    TAKE(t->poff, uint32_t, t->n + 1, 4);
    TAKE(t->piece, uint16_t, t->poff[t->n], 4);
    TAKE(t->soff, uint32_t, t->n_model + 1, 1);
    TAKE(t->sbytes, uint8_t, t->soff[t->n_model], 1);
#undef TAKE
    for (int b = 0; b < 256; b++)
        if (t->byte_tok[b] >= t->n) return -1;
    for (int i = 0; i < t->n_merges; i++)
        if (t->mout[i] >= t->n) return -1;
    for (int i = 0; i < t->n; i++)
        if (t->poff[i] > t->poff[i + 1]) return -1;
    for (uint32_t i = 0; i < t->poff[t->n]; i++)
        if (t->piece[i] >= t->n_model) return -1;
    for (int i = 0; i < t->n_model; i++)
        if (t->soff[i] > t->soff[i + 1]) return -1;
    return 0;
}

int nslm_tok_bind(nslm_tok *t, const void *image, size_t len) {
    const uint32_t *h = (const uint32_t *)image;
    memset(t, 0, sizeof *t);
    if (((uintptr_t)image & 3) || len < 16) return -1;
    if (h[0] == 0x424C534Eu) return bind_bl(t, h, len);
    if (h[0] != 0x544C534Eu) return -1;
    t->kind = NSLM_TOK_SP;
    t->n = (int)h[1];
    t->max_len = (int)h[2];
    t->n_model = (int)h[3];
    size_t need = 16 + (size_t)t->n * 16 + (size_t)t->n_model * 4 + 4;
    if (len < need || t->n_model < 259 || t->n_model > t->n) return -1;
    const uint32_t *p = h + 4;
    t->rank = (const int32_t *)p;
    t->off = p + t->n;
    t->sorted = p + 2 * t->n;
    t->old2new = (const int32_t *)(p + 3 * t->n);
    t->new2old = (const int32_t *)(p + 4 * t->n);
    uint32_t slen = p[4 * t->n + t->n_model];
    if (len < need + slen) return -1;
    t->str = (const char *)(p + 4 * t->n + t->n_model + 1);
    return 0;
}

static int lookup(const nslm_tok *t, const char *s) {
    int lo = 0, hi = t->n - 1, found = -1;
    while (lo <= hi) { /* first id among equal strings, like Python's setdefault */
        int mid = (lo + hi) / 2;
        int c = strcmp(t->str + t->off[t->sorted[mid]], s);
        if (c < 0) {
            lo = mid + 1;
        } else {
            if (c == 0) found = (int)t->sorted[mid];
            hi = mid - 1;
        }
    }
    return found;
}

/* ---- byte-level BPE (compiler/bltok.py is the reference; tests/test_bltok.py checks both against HF) */

static int bl_cls(unsigned c) { /* s: space, n: digit, l: letter (bytes >= 0x80 too), o: other */
    if (c == ' ' || (c >= 9 && c <= 13)) return 's';
    if (c >= '0' && c <= '9') return 'n';
    if ((c | 32) >= 'a' && (c | 32) <= 'z') return 'l';
    return c >= 0x80 ? 'l' : 'o';
}

static int bl_merge(const nslm_tok *t, int a, int b, int *out_id) {
    uint32_t key = ((uint32_t)a << 16) | (uint32_t)b;
    int lo = 0, hi = t->n_merges - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (t->mkey[mid] < key) lo = mid + 1;
        else if (t->mkey[mid] > key) hi = mid - 1;
        else {
            *out_id = t->mout[mid];
            return t->mrank[mid];
        }
    }
    return -1;
}

/* BPE of one pre-token s[0..len) into out (full ids); returns the count (<= len). */
static int bl_word(const nslm_tok *t, const unsigned char *s, int len, int *out) {
    int n = 0;
    for (int i = 0; i < len; i++) out[n++] = t->byte_tok[s[i]];
    while (n > 1) {
        int best = 0x7fffffff, bi = -1, bid = 0;
        for (int i = 0; i + 1 < n; i++) {
            int id, r = bl_merge(t, out[i], out[i + 1], &id);
            if (r >= 0 && r < best) {
                best = r;
                bi = i;
                bid = id;
            }
        }
        if (bi < 0) break;
        out[bi] = bid;
        memmove(out + bi + 1, out + bi + 2, (size_t)(n - bi - 2) * sizeof *out);
        n--;
    }
    return n;
}

static const char *const bl_contr[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

static int encode_bl(const nslm_tok *t, const char *text, int *out) {
    const unsigned char *s = (const unsigned char *)text;
    int total = (int)strlen(text), n = 0;
    for (int seg = 0; seg < total;) {
        /* Digits(individual_digits): a digit is its own segment */
        int end = seg;
        if (bl_cls(s[seg]) == 'n') end = seg + 1;
        else
            while (end < total && bl_cls(s[end]) != 'n') end++;
        for (int i = seg; i < end;) {
            int c = bl_cls(s[i]), j = -1;
            if (s[i] == '\'') {
                for (unsigned k = 0; k < sizeof bl_contr / sizeof *bl_contr; k++) {
                    int L = (int)strlen(bl_contr[k]);
                    if (i + L <= end && !memcmp(s + i, bl_contr[k], (size_t)L)) {
                        j = i + L;
                        break;
                    }
                }
            }
            if (j < 0) {
                if (c != 's' || (s[i] == ' ' && i + 1 < end && bl_cls(s[i + 1]) != 's')) {
                    int k = c == 's' ? i + 1 : i, cl = bl_cls(s[k]);
                    j = k + 1;
                    while (j < end && bl_cls(s[j]) == cl) j++;
                } else {
                    j = i;
                    while (j < end && bl_cls(s[j]) == 's') j++;
                    if (j < total && bl_cls(s[j]) != 's' && j - i >= 2) j--; /* \s+(?!\S) sees past the segment */
                }
            }
            n += bl_word(t, s + i, j - i, out + n);
            i = j;
        }
        seg = end;
    }
    /* full ids -> model pieces, in place from the back (every piece covers >= 1 byte, so it fits) */
    int m = 0;
    for (int i = 0; i < n; i++) m += (int)(t->poff[out[i] + 1] - t->poff[out[i]]);
    for (int i = n - 1, w = m; i >= 0; i--) {
        int id = out[i];
        for (int k = (int)t->poff[id + 1] - 1; k >= (int)t->poff[id]; k--) out[--w] = t->piece[k];
    }
    return m;
}

int nslm_tok_special(const nslm_tok *t, int full_id) {
    if (t->kind != NSLM_TOK_BL || full_id < 0 || full_id >= t->n || t->poff[full_id + 1] != t->poff[full_id] + 1)
        return -1;
    return t->piece[t->poff[full_id]];
}

int nslm_tok_encode(const nslm_tok *t, const char *text, int *out, char *buf) {
    if (t->kind == NSLM_TOK_BL) return encode_bl(t, text, out);
    int n = 0;
    out[n++] = 1; /* BOS */
    if (text[0]) out[n++] = lookup(t, " ");
    for (const char *c = text; *c;) {
        int len = 1;
        while (c[len] && ((unsigned char)c[len] & 0xC0) == 0x80 && len < 4) len++;
        memcpy(buf, c, (size_t)len);
        buf[len] = 0;
        int id = lookup(t, buf);
        if (id >= 0) {
            out[n++] = id;
        } else {
            for (int i = 0; i < len; i++) out[n++] = (unsigned char)c[i] + 3;
        }
        c += len;
    }
    for (;;) {
        int best = -1, best_id = -1, best_i = -1;
        for (int i = 0; i + 1 < n; i++) {
            const char *a = t->str + t->off[out[i]], *b = t->str + t->off[out[i + 1]];
            size_t la = strlen(a), lb = strlen(b);
            if ((int)(la + lb) > t->max_len) continue;
            memcpy(buf, a, la);
            memcpy(buf + la, b, lb + 1);
            int id = lookup(t, buf);
            if (id >= 0 && t->rank[id] > best) {
                best = t->rank[id];
                best_id = id;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        out[best_i] = best_id;
        memmove(out + best_i + 1, out + best_i + 2, (size_t)(n - best_i - 2) * sizeof *out);
        n--;
    }
    /* Map to model ids. A compiled-out piece is re-segmented in place by longest match over kept
     * pieces (bytes are always kept). Every model token covers at least one byte, so n never exceeds
     * strlen(text) + 2 and the caller's out[] bound holds throughout. */
    for (int i = 0; i < n; i++) {
        int id = out[i];
        if (t->old2new[id] >= 0) {
            out[i] = t->old2new[id];
            continue;
        }
        int pend[64], np = 0; /* a piece is at most max_len (<= 64 in practice) bytes */
        const char *src = t->str + t->off[id];
        int L = (int)strlen(src);
        for (int k = 0; k < L && np < 64;) {
            int done = 0;
            for (int len = (L - k < t->max_len ? L - k : t->max_len); len > 0 && !done; len--) {
                memcpy(buf, src + k, (size_t)len);
                buf[len] = 0;
                int tid = lookup(t, buf);
                if (tid >= 0 && t->old2new[tid] >= 0) {
                    pend[np++] = t->old2new[tid];
                    k += len;
                    done = 1;
                }
            }
            if (!done) pend[np++] = t->old2new[(unsigned char)src[k++] + 3];
        }
        memmove(out + i + np, out + i + 1, (size_t)(n - i - 1) * sizeof *out);
        memcpy(out + i, pend, (size_t)np * sizeof *out);
        n += np - 1;
        i += np - 1;
    }
    return n;
}

const char *nslm_tok_piece(const nslm_tok *t, int prev, int tok, int *n, char byte_buf[2]) {
    if (t->kind == NSLM_TOK_BL) {
        *n = (int)(t->soff[tok + 1] - t->soff[tok]);
        return (const char *)t->sbytes + t->soff[tok];
    }
    const char *s = t->str + t->off[t->new2old[tok]];
    if (prev == 1 && s[0] == ' ') s++;
    size_t len = strlen(s);
    if (len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
        int v = 0;
        for (int i = 3; i < 5; i++) {
            char c = s[i];
            v = v * 16 + (c <= '9' ? c - '0' : (c | 32) - 'a' + 10);
        }
        byte_buf[0] = (char)v;
        byte_buf[1] = 0;
        *n = 1;
        return byte_buf;
    }
    *n = (int)len;
    return s;
}
