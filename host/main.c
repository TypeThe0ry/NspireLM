/* Host driver for the NspireLM runtime: generation, logit dumps and timing.
 *
 *   nslm_host MODEL.nslm TOK.nslt [-p prompt] [-n steps] [-t temp_q8] [-s seed]
 *   nslm_host MODEL.nslm TOK.nslt --encode TEXT      prints token ids
 *   nslm_host MODEL.nslm TOK.nslt --bench POS        forward at pos 8 and POS between perf marks
 *   nslm_host MODEL.nslm TOK.nslt --eval TOKENS.i32 LOGITS.i32
 *       runs every token of TOKENS (int32) and writes Q16 logits per position
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nslm.h"

static void *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    /* 32-byte aligned: each x4 weight chunk must sit in exactly one cache line. The model image has a
     * 32-byte header, so tensor offsets (multiples of 32) stay line aligned. */
    void *p = 0;
    if (posix_memalign(&p, 32, *len + 32)) p = 0;
    if (!p || fread(p, 1, *len, f) != *len) {
        fprintf(stderr, "read failed: %s\n", path);
        exit(1);
    }
    fclose(f);
    return p;
}

/* The cycle model (perf/arm926.c) prints counters each time this function's block executes. */
__attribute__((noinline)) void nslm_perf_mark(int id) {
    __asm__ volatile("" ::: "memory");
    (void)id;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.nslm TOK.nslt [-p prompt] [-n steps] [-t temp_q8] [-s seed]"
                        " | --eval TOKENS.i32 LOGITS.i32\n", argv[0]);
        return 2;
    }
    size_t mlen, tlen;
    void *mimg = slurp(argv[1], &mlen), *timg = slurp(argv[2], &tlen);
    static nslm_model model;
    nslm_state *state = malloc(sizeof *state);
    nslm_tok tok;
    if (nslm_bind(&model, mimg, mlen, state) || nslm_tok_bind(&tok, timg, tlen)) {
        fprintf(stderr, "model/tokenizer image does not match this build\n");
        return 1;
    }
    const char *prompt = "";
    int steps = 256, temp = 0;
    uint32_t seed = 42;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--encode") && i + 1 < argc) {
            const char *text = argv[i + 1];
            int *ids = malloc((strlen(text) + 3) * sizeof *ids);
            char *scratch = malloc((size_t)tok.max_len * 2 + 8);
            int n = nslm_tok_encode(&tok, text, ids, scratch);
            for (int k = 0; k < n; k++) printf("%d%c", ids[k], k + 1 < n ? ' ' : '\n');
            return 0;
        }
        if (!strcmp(argv[i], "--bench") && i + 1 < argc) {
            /* timing only: cost depends on shape and position, not on cache contents */
            int far = atoi(argv[i + 1]);
            if (far >= NSLM_SEQ) far = NSLM_SEQ - 1;
            nslm_forward(&model, 1, 7);
            nslm_perf_mark(0);
            nslm_forward(&model, 300 % NSLM_VOCAB, 8);
            nslm_perf_mark(1);
            nslm_forward(&model, 301 % NSLM_VOCAB, far);
            nslm_perf_mark(2);
            return 0;
        }
        if (!strcmp(argv[i], "--eval") && i + 2 < argc) {
            size_t n;
            int32_t *toks = slurp(argv[i + 1], &n);
            n /= 4;
            FILE *out = fopen(argv[i + 2], "wb");
            double t0 = now();
            for (size_t p = 0; p < n && p < NSLM_SEQ; p++) {
                nslm_forward(&model, toks[p], (int)p);
                fwrite(state->logits, 4, NSLM_VOCAB, out);
            }
            fclose(out);
            fprintf(stderr, "eval: %zu positions, %.1f tok/s\n", n, n / (now() - t0));
            return 0;
        } else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            temp = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            seed = (uint32_t)strtoul(argv[++i], 0, 10);
        }
    }
    if (steps > NSLM_SEQ) steps = NSLM_SEQ;
    int *ptoks = malloc((strlen(prompt) + 3) * sizeof *ptoks);
    char *scratch = malloc((size_t)tok.max_len * 2 + 8);
    int np = nslm_tok_encode(&tok, prompt, ptoks, scratch);
    int cur = ptoks[0], prev = 0;
    double t0 = now();
    int pos = 0;
    for (; pos < steps; pos++) {
        nslm_forward(&model, cur, pos);
        int next = pos + 1 < np ? ptoks[pos + 1] : nslm_sample(state->logits, NSLM_VOCAB, temp, &seed);
        if (next == 1) break; /* BOS ends a story */
        int len;
        char bb[2];
        const char *piece = nslm_tok_piece(&tok, cur, next, &len, bb);
        fwrite(piece, 1, (size_t)len, stdout);
        fflush(stdout);
        prev = cur;
        cur = next;
    }
    (void)prev;
    printf("\n");
    fprintf(stderr, "%d tokens, %.1f tok/s\n", pos, pos / (now() - t0));
    return 0;
}
