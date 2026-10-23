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

typedef struct {
    const int8_t *q;   /* [rows][cols] */
    const uint16_t *m; /* [rows][cols / gs] */
    const int8_t *e;   /* [rows] */
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
    int16_t qh[NSLM_HEAD_DIM];
    int32_t z[NSLM_SEQ];
    int64_t w[NSLM_SEQ];
    int16_t w16[NSLM_SEQ];
    int32_t logits[NSLM_VOCAB];
    int8_t kc[NSLM_LAYERS][NSLM_SEQ][NSLM_KV_DIM];
    int8_t vc[NSLM_LAYERS][NSLM_SEQ][NSLM_KV_DIM];
    int32_t ks[NSLM_LAYERS][NSLM_SEQ][NSLM_KV_HEADS];
    int32_t vs[NSLM_LAYERS][NSLM_SEQ][NSLM_KV_HEADS];
} nslm_state;

typedef struct {
    nslm_q8 emb, cls;
    nslm_layer layer[NSLM_LAYERS];
    const int32_t *rms_final;
    const int32_t *rope; /* [SEQ][HEAD_DIM/2][2] cos, sin in Q30 */
    nslm_state *s;
} nslm_model;

/* Binds a loaded model.nslm image (must stay alive and be 4-byte aligned). 0 on success. */
int nslm_bind(nslm_model *m, const void *image, size_t len, nslm_state *state);

/* One decoding step. Leaves Q16 logits in m->s->logits. */
void nslm_forward(nslm_model *m, int token, int pos);

int nslm_argmax(const int32_t *logits, int n);

/* temp_q8: temperature in Q8 (256 = 1.0); 0 = greedy. rng: xorshift32 state. */
int nslm_sample(const int32_t *logits, int n, int temp_q8, uint32_t *rng);

/* Kernels (portable C in nslm_kern.c; ARM assembly versions replace them on device). */
void nslm_matmul_q8(int32_t *out, const nslm_q8 *w, const int16_t *a, int ea, int rows, int cols, int gs);

/* Tokenizer (tok.nslt). */
typedef struct {
    int n, max_len;
    const int32_t *rank;
    const uint32_t *off, *sorted;
    const char *str;
} nslm_tok;

int nslm_tok_bind(nslm_tok *t, const void *image, size_t len);
/* Returns the number of tokens written (BOS first). buf must hold strlen(text) + 3 entries. */
int nslm_tok_encode(const nslm_tok *t, const char *text, int *out, char *scratch);
/* Decoded bytes of `tok` following `prev`; *n receives the length. */
const char *nslm_tok_piece(const nslm_tok *t, int prev, int tok, int *n, char byte_buf[2]);

#endif
