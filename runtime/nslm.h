/* NspireLM runtime: integer-only Llama inference, specialised at compile time.
 *
 * The model's shapes and constants come from the generated nslm_model.h (compiler/nslmc.py),
 * so every loop bound below is a compile-time constant. Numeric formats:
 *   fx32   int32, Q16 (real = v / 65536): residual stream, matmul outputs, logits
 *   act16  int16 values + one exponent E per vector (real = v * 2^-E): matmul inputs
 *   q8     int8 weights, group-wise uint16 mantissa scales, one exponent per row
 *   kv     int8 per (position, kv head) with a Q24 scale
 */
#ifndef NSLM_H
#define NSLM_H

#include <stddef.h>
#include <stdint.h>

#include "nslm_model.h"

#if NSLM_SEQ > 1024
#error "attention accumulators are sized for SEQ <= 1024"
#endif

#define NSLM_MAXV (NSLM_HIDDEN > NSLM_DIM ? NSLM_HIDDEN : NSLM_DIM)
#define NSLM_SEQP ((NSLM_SEQ + 7) & ~7) /* value rows padded for 8-wide dot kernels */
#if NSLM_HEAD_DIM % 8
#error "attention dot kernels need head_dim % 8 == 0"
#endif

/* x4 layout (compiler/nslmc.py pack_x4): q is offset-binary uint8 in blocks of 4 rows x 8 columns
 * (one 32-byte cache line); m is [rows/4][groups][4]; e is per row. int4 matrices use the same struct
 * with the x4n layout (pack_x4n: 4 rows x 16 columns per line). */
typedef struct {
    const uint8_t *q;
    const uint16_t *m;
    const int8_t *e;
} nslm_q8;

typedef struct {
    const int32_t *rms_att, *rms_ffn;
    nslm_q8 wq, wk, wv, wo, w1, w2, w3;
} nslm_layer;

typedef struct {
    int32_t x[NSLM_DIM], xb[NSLM_DIM], q[NSLM_DIM];
    int32_t k[NSLM_KV_DIM], v[NSLM_KV_DIM];
    int32_t h1[NSLM_HIDDEN], h3[NSLM_HIDDEN];
    int16_t a[NSLM_MAXV];
    int32_t asum[NSLM_MAXV / 8]; /* per-group sums of a, for the offset-binary correction */
    int16_t qh[NSLM_HEAD_DIM];
    int32_t z[NSLM_SEQ];
    int64_t w[NSLM_SEQ];
    int16_t w16[NSLM_SEQP];
    int32_t logits[NSLM_VOCAB_ROWS]; /* only the first NSLM_VOCAB are real tokens */
    /* KV cache, offset-binary (q + 128), laid out so both attention products are contiguous dots:
     * keys [head][pos][dim], values transposed [head][dim][pos] */
    uint8_t kc[NSLM_LAYERS][NSLM_KV_HEADS][NSLM_SEQ][NSLM_HEAD_DIM];
    uint8_t vt[NSLM_LAYERS][NSLM_KV_HEADS][NSLM_HEAD_DIM][NSLM_SEQP];
    int32_t ks[NSLM_LAYERS][NSLM_KV_HEADS][NSLM_SEQ];
    int32_t vs[NSLM_LAYERS][NSLM_KV_HEADS][NSLM_SEQ];
} nslm_state;

typedef struct {
    nslm_q8 emb[NSLM_EMB_PARTS], cls[NSLM_EMB_PARTS]; /* row parts of NSLM_EMB_PART_ROWS rows */
    nslm_layer layer[NSLM_LAYERS];
    const int32_t *rms_final;
    const int32_t *rope; /* [SEQ][HEAD_DIM/2][2] cos, sin in Q30 */
    nslm_state *s;
} nslm_model;

/* Binds a loaded model.nslm image (must stay alive and be 32-byte aligned). 0 on success. */
int nslm_bind(nslm_model *m, const void *image, size_t len, nslm_state *state);
/* Same, with the file split at nslm_seg_off[] into NSLM_NSEG separately allocated, 32-byte aligned
 * segments (segs[k] holds file bytes nslm_seg_off[k] .. nslm_seg_off[k+1]). */
int nslm_bind_segments(nslm_model *m, const void *const *segs, nslm_state *state);

/* One decoding step. Leaves Q16 logits in m->s->logits. */
void nslm_forward(nslm_model *m, int token, int pos);

int nslm_argmax(const int32_t *logits, int n);

/* temp_q8: temperature in Q8 (256 = 1.0); 0 = greedy. rng: xorshift32 state. */
int nslm_sample(const int32_t *logits, int n, int temp_q8, uint32_t *rng);

/* Decoding policy, all integer:
 *   temp_q8   temperature in Q8 (0 = greedy)
 *   minp_q16  min-p in the log2 domain: keep tokens whose probability is >= 2^(minp_q16/65536) times the
 *             best one (e.g. -4.32 * 65536 for min-p 0.05); 0 disables
 *   rep_q16   presence penalty subtracted from the logit (Q16) of every token in the last `window` tokens
 *   window    how many recent tokens the penalty looks at (<= NSLM_REP_MAX) */
#define NSLM_REP_MAX 128
typedef struct {
    int temp_q8, minp_q16, rep_q16, window;
} nslm_policy;
/* logits is modified in place (penalty). hist: previous tokens, newest last. */
int nslm_decode(int32_t *logits, int n, const nslm_policy *p, const int *hist, int nhist, uint32_t *rng);

/* Kernels (portable C in nslm_kern.c; ARM assembly versions replace them on device).
 * out[r] = sat32(rshr(sum_g (d_rg * m_rg) >> 16, e_r + ea - 32)), d_rg = sum_{i in g} q_ri * a_i
 * asum[g] = sum_{i in g} a_i. rows % 4 == 0, cols % 8 == 0, gs % 8 == 0. */
void nslm_matmul_q8(int32_t *out, const nslm_q8 *w, const int16_t *a, const int32_t *asum, int ea,
                    int rows, int cols, int gs);
/* Same contract for int4 weights in the x4n layout (codes u = q + 8). cols % 16 == 0, gs % 16 == 0. */
void nslm_matmul_q4(int32_t *out, const nslm_q8 *w, const int16_t *a, const int32_t *asum, int ea,
                    int rows, int cols, int gs);
/* sum_i a[i] * u[i] for n % 8 == 0 (attention: int16 query/weights against offset-binary KV bytes) */
int32_t nslm_dot_s16_u8(const int16_t *a, const uint8_t *u, int n);
/* out[r] = sum_i a[i] * u[r * n + i] for r < rows (all attention scores of one head) */
void nslm_dots_s16_u8(int32_t *out, const int16_t *a, const uint8_t *u, int n, int rows);

/* Tokenizer (tok.nslt). */
typedef struct {
    int n, max_len, n_model;           /* full BPE vocab, longest piece, model vocab */
    const int32_t *rank;
    const uint32_t *off, *sorted;
    const int32_t *old2new, *new2old; /* BPE id <-> model id (-1: compiled out) */
    const char *str;
} nslm_tok;

int nslm_tok_bind(nslm_tok *t, const void *image, size_t len);
/* Returns the number of model-vocabulary tokens written (BOS first); out must hold strlen(text) + 3
 * entries and scratch 2 * max_len + 8 bytes. Pieces the model was compiled without are re-segmented. */
int nslm_tok_encode(const nslm_tok *t, const char *text, int *out, char *scratch);
/* Decoded bytes of `tok` following `prev`; *n receives the length. */
const char *nslm_tok_piece(const nslm_tok *t, int prev, int tok, int *n, char byte_buf[2]);

#endif
