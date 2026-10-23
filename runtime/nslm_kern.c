/* Portable reference kernels. The ARM assembly kernels must match these bit for bit. */
#include "nslm.h"

static inline int64_t rshr64(int64_t v, int s) {
    if (s > 0) return (v + ((int64_t)1 << (s - 1))) >> s;
    return v * ((int64_t)1 << -s);
}

#ifndef NSLM_HAVE_ASM_MATMUL
/* out[r] = sum_g m[r][g] * (sum_{i in g} q[r][i] * a[i]), scaled to Q16.
 * Per group the int32 partial cannot overflow: gs * 127 * 32768 < 2^31 for gs <= 512. */
void nslm_matmul_q8(int32_t *out, const nslm_q8 *w, const int16_t *a, int ea, int rows, int cols, int gs) {
    const int ng = cols / gs;
    for (int r = 0; r < rows; r++) {
        const int8_t *q = w->q + (size_t)r * cols;
        const uint16_t *m = w->m + (size_t)r * ng;
        int64_t acc = 0;
        for (int g = 0; g < ng; g++) {
            int32_t d = 0;
            for (int i = 0; i < gs; i++) d += (int32_t)q[g * gs + i] * a[g * gs + i];
            acc += (int64_t)m[g] * d;
        }
        int64_t v = rshr64(acc, w->e[r] + ea - 16);
        out[r] = v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v;
    }
}
#endif
