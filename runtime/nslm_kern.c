/* Portable reference kernels. The ARM assembly kernels must match these bit for bit. */
#include "nslm.h"

static inline int64_t rshr64(int64_t v, int s) {
    if (s > 0) return (v + ((int64_t)1 << (s - 1))) >> s;
    return v * ((int64_t)1 << -s);
}

#ifdef NSLM_HAVE_ASM_MATMUL
void nslm_q8_dot_x4(int32_t *acc, const uint8_t *q, const uint16_t *m, const int16_t *a, const int32_t *asum,
                    int row_blocks, int groups, int chunks_per_group);

void nslm_matmul_q8(int32_t *out, const nslm_q8 *w, const int16_t *a, const int32_t *asum, int ea,
                    int rows, int cols, int gs) {
    nslm_q8_dot_x4(out, w->q, w->m, a, asum, rows / 4, cols / gs, gs / 8);
    for (int r = 0; r < rows; r++) {
        int64_t v = rshr64(out[r], w->e[r] + ea - 32);
        out[r] = v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v;
    }
}
#else
void nslm_matmul_q8(int32_t *out, const nslm_q8 *w, const int16_t *a, const int32_t *asum, int ea,
                    int rows, int cols, int gs) {
    const int ng = cols / gs, nc = cols / 8;
    for (int rb = 0; rb < rows / 4; rb++) {
        const uint8_t *blk = w->q + (size_t)rb * 32 * nc;
        const uint16_t *mb = w->m + (size_t)rb * 4 * ng;
        for (int j = 0; j < 4; j++) {
            int64_t acc = 0;
            for (int g = 0; g < ng; g++) {
                int32_t d = 0; /* sum of u * a, then the offset-binary correction */
                for (int i = g * gs; i < (g + 1) * gs; i++) d += (int32_t)blk[(i >> 3) * 32 + j * 8 + (i & 7)] * a[i];
                d -= asum[g] * 128;
                acc += ((int64_t)d * mb[g * 4 + j]) >> 16; /* SMLAWB: floor */
            }
            int r = rb * 4 + j;
            int64_t v = rshr64(acc, w->e[r] + ea - 32);
            out[r] = v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v;
        }
    }
}
#endif
