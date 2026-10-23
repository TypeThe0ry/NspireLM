/* Integer-only port of llama2.c's BPE tokenizer (merge scores replaced by ranks). */
#include <string.h>

#include "nslm.h"

int nslm_tok_bind(nslm_tok *t, const void *image, size_t len) {
    const uint32_t *h = (const uint32_t *)image;
    if (((uintptr_t)image & 3) || len < 16 || h[0] != 0x544C534Eu) return -1;
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

int nslm_tok_encode(const nslm_tok *t, const char *text, int *out, char *buf) {
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
