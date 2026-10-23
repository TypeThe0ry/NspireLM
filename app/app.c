/* NspireLM app: prompt editor, streamed generation and an on-device self-test.
 * Platform independent; see plat.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nslm.h"
#include "plat.h"

#if !defined(NSLM_NO_TEST_H) && __has_include("nslm_test.h")
#include "nslm_test.h" /* generated: expected self-test hash from the host build */
#endif
#ifndef NSLM_TEST_HASH
#define NSLM_TEST_HASH 0u
#endif
#define NSLM_TEST_PROMPT "Once upon a time, there was a little robot who loved numbers."
#define NSLM_TEST_NTOK 24

extern const uint8_t nslm_font8x8[128][8];

#define COLS (SCR_W / 8)
#define ROWS (SCR_H / 8)
#define TEXT_ROWS (ROWS - 3)
#define C_BG 0x0841
#define C_FG 0xE71C
#define C_PROMPT 0x5E7F
#define C_DIM 0x8410
#define C_BAR 0x2124
#define C_ACCENT 0xFD20

static uint16_t *fb;
static char text[16384]; /* the conversation shown in the text area */
static int text_len;
static char input[160];
static int input_len;
static char status[COLS + 1];
static int prompt_start; /* text offset where the prompt echo begins (drawn in another colour) */
static int prompt_end;

static nslm_model model;
static nslm_state *state;
static nslm_tok tok;
/* Decoding presets chosen by tools/sample_eval.py (judge: stories42M float perplexity of the continuation,
 * plus 4-gram repetition; real TinyStories text scores ~2.12):
 *   precise   greedy + presence penalty 1.0        judge 1.84, rep4 1.4%  (deterministic)
 *   balanced  T 0.6, min-p 0.05, penalty 0.5       judge 2.11, rep4 1.8%
 *   creative  T 0.8, min-p 0.05, penalty 1.0       judge 2.54, rep4 1.6%, most diverse
 * Plain T 0.8 sampling scores 3.64 and plain greedy loops (rep4 5.9%). */
static const struct {
    const char *name;
    nslm_policy p;
} presets[] = {
    {"balanced", {154, -283242, 32768, 64}},
    {"precise", {0, 0, 65536, 64}},
    {"creative", {205, -283242, 65536, 64}},
};
static int preset = 0;
static uint32_t rng = 0x9E3779B9u;

/* ------------------------------------------------------------------ drawing */

static void fill(int x, int y, int w, int h, uint16_t c) {
    for (int j = y; j < y + h; j++)
        for (int i = x; i < x + w; i++) fb[j * SCR_W + i] = c;
}

static void glyph(int cx, int cy, unsigned char ch, uint16_t fg, uint16_t bg) {
    const uint8_t *g = nslm_font8x8[ch < 128 ? ch : '?'];
    for (int y = 0; y < 8; y++) {
        uint16_t *row = fb + (cy * 8 + y) * SCR_W + cx * 8;
        for (int x = 0; x < 8; x++) row[x] = (g[y] >> x) & 1 ? fg : bg;
    }
}

static void line_text(int row, const char *s, uint16_t fg, uint16_t bg) {
    int i = 0;
    for (; s[i] && i < COLS; i++) glyph(i, row, (unsigned char)s[i], fg, bg);
    for (; i < COLS; i++) glyph(i, row, ' ', fg, bg);
}

/* Word-wrapped layout of text[]; draws the last TEXT_ROWS lines. */
static void draw_text(void) {
    static int starts[4096];
    int nlines = 0, i = 0;
    while (i < text_len && nlines < 4096) {
        starts[nlines++] = i;
        int j = i, last_space = -1;
        while (j < text_len && text[j] != '\n' && j - i < COLS) {
            if (text[j] == ' ') last_space = j;
            j++;
        }
        if (j < text_len && text[j] == '\n') {
            i = j + 1;
        } else if (j - i >= COLS && last_space > i && j < text_len && text[j] != ' ') {
            i = last_space + 1;
        } else {
            i = j;
            while (i < text_len && text[i] == ' ' && i - starts[nlines - 1] >= COLS) i++;
        }
    }
    int first = nlines > TEXT_ROWS ? nlines - TEXT_ROWS : 0;
    fill(0, 0, SCR_W, TEXT_ROWS * 8, C_BG);
    for (int l = first; l < nlines; l++) {
        int s = starts[l], e = l + 1 < nlines ? starts[l + 1] : text_len;
        for (int k = s, c = 0; k < e && c < COLS; k++, c++) {
            if (text[k] == '\n') break;
            uint16_t col = (k >= prompt_start && k < prompt_end) ? C_PROMPT : C_FG;
            glyph(c, l - first, (unsigned char)text[k], col, C_BG);
        }
    }
}

static void draw(void) {
    draw_text();
    fill(0, TEXT_ROWS * 8, SCR_W, 8, C_BAR);
    char in[COLS + 1];
    int off = input_len > COLS - 3 ? input_len - (COLS - 3) : 0;
    snprintf(in, sizeof in, "> %s_", input + off);
    line_text(ROWS - 2, in, C_ACCENT, C_BG);
    line_text(ROWS - 1, status, C_DIM, C_BAR);
    plat_present(fb);
}

static void append(const char *s, int n) {
    if (text_len + n >= (int)sizeof text) { /* drop the oldest half */
        int cut = text_len / 2;
        memmove(text, text + cut, (size_t)(text_len - cut));
        text_len -= cut;
        prompt_start = prompt_start > cut ? prompt_start - cut : 0;
        prompt_end = prompt_end > cut ? prompt_end - cut : 0;
    }
    memcpy(text + text_len, s, (size_t)n);
    text_len += n;
}

static void appends(const char *s) { append(s, (int)strlen(s)); }

static void set_status(const char *s) { /* truncates to the screen width on purpose */
    size_t n = strlen(s);
    if (n > COLS) n = COLS;
    memcpy(status, s, n);
    status[n] = 0;
}

static void idle_status(void) {
    char s[96];
    snprintf(s, sizeof s, "enter:go tab:%s menu:test esc", presets[preset].name);
    set_status(s);
}

/* ------------------------------------------------------------------ generation */

static void generate(const char *prompt) {
    static int ptoks[NSLM_SEQ + 8], hist[NSLM_SEQ + 8];
    static char scratch[512];
    char buf[200];
    size_t plen = strlen(prompt);
    if (plen + 3 > NSLM_SEQ) plen = NSLM_SEQ - 3;
    memcpy(buf, prompt, plen);
    buf[plen] = 0;
    int np = nslm_tok_encode(&tok, buf, ptoks, scratch);
    if (text_len) appends("\n\n");
    prompt_start = text_len;
    appends(buf);
    prompt_end = text_len;

    int cur = ptoks[0], pos = 0, generated = 0;
    uint64_t t0 = plat_us(), tgen = 0;
    for (; pos < NSLM_SEQ; pos++) {
        if (plat_abort()) break;
        nslm_forward(&model, cur, pos);
        hist[pos] = cur;
        int next;
        if (pos + 1 < np) {
            next = ptoks[pos + 1];
        } else {
            if (!tgen) tgen = plat_us();
            next = nslm_decode(state->logits, NSLM_VOCAB, &presets[preset].p, hist, pos + 1, &rng);
            if (next == 1 || next == 2) break; /* BOS/EOS: the story is over */
            int len;
            char bb[2];
            const char *piece = nslm_tok_piece(&tok, cur, next, &len, bb);
            append(piece, len);
            generated++;
        }
        cur = next;
        uint64_t now = plat_us();
        char s[96];
        if (generated > 1) {
            uint64_t dt = now - tgen;
            unsigned tps100 = (unsigned)((uint64_t)(generated - 1) * 100000000ull / (dt ? dt : 1));
            snprintf(s, sizeof s, "%d tok  %u.%02u tok/s  esc:stop", generated, tps100 / 100, tps100 % 100);
        } else {
            snprintf(s, sizeof s, "reading prompt %d/%d  esc:stop", pos + 1, np);
        }
        set_status(s);
        draw();
    }
    (void)t0;
    idle_status();
}

/* Chat mode (byte-level tokenizer, ChatML, train/distill.py): every turn continues the same KV cache:
 *   <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n
 * When the next turn would not fit in NSLM_SEQ positions, the conversation restarts. Enter on an empty
 * line also starts a new conversation. */
static int chat_pos;
static int chat_last; /* the last token fed (its logits are not needed) */
static int chat_hist[NSLM_SEQ + 8];

static int chat_feed(const int *ids, int n) {
    for (int i = 0; i < n; i++) {
        if (chat_pos >= NSLM_SEQ) return -1;
        nslm_forward(&model, ids[i], chat_pos);
        chat_hist[chat_pos++] = ids[i];
        chat_last = ids[i];
    }
    return 0;
}

static int chat_text_ids(const char *s, int *out) {
    static char scratch[512];
    return nslm_tok_encode(&tok, s, out, scratch);
}

static void chat_reset(void) {
    chat_pos = 0;
    memset(state, 0, sizeof *state);
}

static void chat(const char *question) {
    static int ids[sizeof input + NSLM_SEQ + 16];
    const int im_start = nslm_tok_special(&tok, 1), im_end = nslm_tok_special(&tok, 2),
              eot = nslm_tok_special(&tok, 0);
    if (!question[0]) {
        chat_reset();
        if (text_len) appends("\n\n");
        appends("(new conversation)");
        idle_status();
        return;
    }
    int n = 0;
    ids[n++] = im_start;
    n += chat_text_ids("user\n", ids + n);
    n += chat_text_ids(question, ids + n);
    ids[n++] = im_end;
    n += chat_text_ids("\n", ids + n);
    ids[n++] = im_start;
    n += chat_text_ids("assistant\n", ids + n);
    /* keep room for a useful answer; otherwise start over */
    if (chat_pos + n + 48 > NSLM_SEQ) {
        int fresh = chat_pos > 0;
        chat_reset();
        if (n + 16 > NSLM_SEQ) {
            appends("\n\n(question too long)");
            idle_status();
            return;
        }
        if (fresh) appends("\n\n(context full: new conversation)");
    }
    if (text_len) appends("\n\n");
    prompt_start = text_len;
    appends("Q: ");
    appends(question);
    prompt_end = text_len;
    appends("\nA: ");
    set_status("reading the question...  esc:stop");
    draw();
    uint64_t t0 = plat_us();
    for (int i = 0; i < n; i++) {
        if (plat_abort()) break;
        chat_feed(ids + i, 1);
        char st[96];
        snprintf(st, sizeof st, "reading the question %d/%d  esc:stop", i + 1, n);
        set_status(st);
        draw();
    }
    int generated = 0;
    uint64_t tgen = plat_us();
    (void)t0;
    while (chat_pos < NSLM_SEQ && !plat_abort()) {
        int next = nslm_decode(state->logits, NSLM_VOCAB, &presets[preset].p, chat_hist, chat_pos, &rng);
        if (next == im_end || next == eot || next == im_start) break;
        int len;
        char bb[2];
        const char *piece = nslm_tok_piece(&tok, chat_last, next, &len, bb);
        append(piece, len);
        generated++;
        if (chat_feed(&next, 1)) break;
        char st[96];
        uint64_t dt = plat_us() - tgen;
        unsigned tps100 = (unsigned)((uint64_t)generated * 100000000ull / (dt ? dt : 1));
        snprintf(st, sizeof st, "%d tok  %u.%02u tok/s  esc:stop", generated, tps100 / 100, tps100 % 100);
        set_status(st);
        draw();
    }
    /* close the turn so the next question continues the conversation */
    int tail[4], nt = 0;
    tail[nt++] = im_end;
    nt += chat_text_ids("\n", tail + nt);
    if (chat_feed(tail, nt)) chat_reset();
    idle_status();
}

/* ------------------------------------------------------------------ self-test */

static uint32_t fnv(uint32_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
    return h;
}

uint32_t nslm_selftest_hash(void) {
    static int ids[NSLM_SEQ + 8];
    static char scratch[512];
    int n = nslm_tok_encode(&tok, NSLM_TEST_PROMPT, ids, scratch);
    if (n > NSLM_TEST_NTOK) n = NSLM_TEST_NTOK;
    memset(state, 0, sizeof *state);
    uint32_t h = 2166136261u;
    for (int p = 0; p < n; p++) {
        nslm_forward(&model, ids[p], p);
        h = fnv(h, state->logits, sizeof state->logits);
    }
    memset(state, 0, sizeof *state);
    return h;
}

__attribute__((noinline)) static uint32_t sum_seq(const uint32_t *p, uint32_t words) {
    uint32_t acc = 0;
    for (uint32_t i = 0; i < words; i += 8)
        acc += p[i + 0] + p[i + 1] + p[i + 2] + p[i + 3] + p[i + 4] + p[i + 5] + p[i + 6] + p[i + 7];
    return acc;
}

__attribute__((noinline)) static uint32_t sum_stride(const uint32_t *p, uint32_t words) {
    uint32_t acc = 0;
    for (uint32_t i = 0; i < words; i += 8) acc += p[i];
    return acc;
}

static void report(char *out, size_t cap, const char *fmt, long a, long b) {
    size_t n = strlen(out);
    snprintf(out + n, cap - n, fmt, a, b);
}

static void selftest(void) {
    static char r[2048];
    r[0] = 0;
    set_status("self-test running...");
    if (text_len) appends("\n\n");
    appends("SELF-TEST\n");
    draw();
    int mhz = plat_cpu_mhz();
    report(r, sizeof r, "cpu_mhz=%ld battery_ok=%ld\n", mhz, plat_battery_ok());

    uint64_t t0 = plat_us();
    uint32_t h = nslm_selftest_hash();
    uint64_t t1 = plat_us();
    {
        size_t n = strlen(r);
        snprintf(r + n, sizeof r - n, "hash=%08lx expect=%08lx %s (%ld ms)\n", (unsigned long)h,
                 (unsigned long)NSLM_TEST_HASH,
                 NSLM_TEST_HASH == 0 ? "UNSET" : h == NSLM_TEST_HASH ? "BITEXACT" : "MISMATCH",
                 (long)((t1 - t0) / 1000));
    }

    /* per-token time: shape and position decide it, not cache contents */
    memset(state, 0, sizeof *state);
    nslm_forward(&model, 1, 7);
    t0 = plat_us();
    nslm_forward(&model, 300 % NSLM_VOCAB, 8);
    t1 = plat_us();
    nslm_forward(&model, 301 % NSLM_VOCAB, NSLM_SEQ - 1);
    uint64_t t2 = plat_us();
    memset(state, 0, sizeof *state);
    report(r, sizeof r, "tok_us_pos8=%ld tok_us_last=%ld\n", (long)(t1 - t0), (long)(t2 - t1));

    /* memory system: sequential (LDM, whole line) vs one word per line -> linefill shape */
    const uint32_t bytes = 4u << 20;
    uint32_t *buf = plat_aligned_alloc(bytes);
    if (buf) {
        for (uint32_t i = 0; i < bytes / 4; i++) buf[i] = i;
        volatile uint32_t sink = 0;
        t0 = plat_us();
        sink += sum_seq(buf, bytes / 4);
        t1 = plat_us();
        sink += sum_stride(buf, bytes / 4);
        t2 = plat_us();
        (void)sink;
        long lines = (long)(bytes / 32);
        long seq_cpl = mhz ? (long)((t1 - t0) * (uint64_t)mhz / (uint64_t)lines) : 0;
        long str_cpl = mhz ? (long)((t2 - t1) * (uint64_t)mhz / (uint64_t)lines) : 0;
        report(r, sizeof r, "seq_us=%ld stride_us=%ld (4 MB)\n", (long)(t1 - t0), (long)(t2 - t1));
        report(r, sizeof r, "cycles_per_line seq=%ld stride=%ld\n", seq_cpl, str_cpl);
    } else {
        report(r, sizeof r, "membench: alloc failed\n%.0ld%.0ld", 0, 0);
    }
    plat_aligned_free(buf);
#ifdef _TINSPIRE
    {
        size_t lo = 0, hi = 64u << 20;
        while (hi - lo > 65536) {
            size_t mid = lo + (hi - lo) / 2;
            void *p = malloc(mid);
            if (p) {
                free(p);
                lo = mid;
            } else {
                hi = mid;
            }
        }
        report(r, sizeof r, "free_largest_kb=%ld state_kb=%ld\n", (long)(lo >> 10), (long)(sizeof *state >> 10));
    }
#endif
    report(r, sizeof r, "model=%ld params_q8_kb=%ld\n", (long)NSLM_DIM, (long)(NSLM_BLOB_BYTES >> 10));
    appends(r);
    chat_reset(); /* the self-test overwrote the KV cache */
    int ok = plat_write("nslm_selftest.txt", r);
    appends(ok == 0 ? "saved nslm_selftest.txt\n" : "could not save the report\n");
    idle_status();
}

/* ------------------------------------------------------------------ main loop */

/* Model file -> NSLM_NSEG separately allocated segments (the Ndless heap holds ~21.5 MiB in one block
 * but ~27 MiB in pieces). 64 KB reads: the read path proven on the device. */
static int load_model(const void **segs, void (*prog)(int)) {
    char path[256];
    FILE *f = fopen(plat_path("model.nslm", path, sizeof path), "rb");
    if (!f) return -1;
    uint32_t total = nslm_seg_off[NSLM_NSEG], done = 0;
    int last = -1;
    for (int k = 0; k < NSLM_NSEG; k++) {
        uint32_t n = nslm_seg_off[k + 1] - nslm_seg_off[k];
        unsigned char *p = plat_aligned_alloc(n);
        if (!p) {
            fclose(f);
            return -2;
        }
        for (uint32_t got = 0; got < n;) {
            size_t want = n - got > 65536u ? 65536u : n - got;
            size_t r = fread(p + got, 1, want, f);
            if (!r) {
                fclose(f);
                return -3;
            }
            got += (uint32_t)r;
            done += (uint32_t)r;
            int pct = (int)((uint64_t)done * 100u / total);
            if (prog && pct != last) prog(pct);
            last = pct;
        }
        segs[k] = p;
    }
    fclose(f);
    return 0;
}

static void progress(int pct) {
    char s[96];
    snprintf(s, sizeof s, "loading model... %d%%", pct);
    set_status(s);
    fill(0, (ROWS - 1) * 8 - 4, SCR_W * pct / 100, 3, C_ACCENT);
    line_text(ROWS - 1, status, C_DIM, C_BAR);
    plat_present(fb);
}

/* hash_only: load, print the self-test hash (used by the build to bake the expected value), exit */
int nslm_app_main(int argc, char **argv, int hash_only) {
    if (plat_init(argc, argv)) return 1;
    fb = plat_aligned_alloc(SCR_W * SCR_H * 2);
    state = plat_aligned_alloc(sizeof *state);
    if (!fb || !state) return 2;
    fill(0, 0, SCR_W, SCR_H, C_BG);
    appends("NspireLM - an offline language model\nrunning on this calculator's own CPU.\n");
    draw();

    static const void *segs[NSLM_NSEG];
    size_t tlen = 0, mlen = nslm_seg_off[NSLM_NSEG];
    int lerr = load_model(segs, progress);
    void *timg = plat_load("tok.nslt", &tlen, 0);
    if (lerr || !timg || nslm_bind_segments(&model, segs, state) || nslm_tok_bind(&tok, timg, tlen)) {
        appends(lerr == -2 ? "\nNot enough memory for the model. Reset the calculator and try again.\n"
                : lerr || !timg ? "\nmodel.nslm.tns / tok.nslt.tns not found next to this program.\n"
                                : "\nThe model file does not match this program build.\n");
        set_status("esc: quit");
        draw();
        while (plat_key() != K_ESC) {
        }
        plat_exit();
        return 3;
    }
    if (hash_only) {
        printf("%08lx\n", (unsigned long)nslm_selftest_hash());
        plat_exit();
        return 0;
    }
    char info[120];
    snprintf(info, sizeof info, "model: %d layers, dim %d, %d tokens, %ld KB\n", NSLM_LAYERS, NSLM_DIM,
             NSLM_VOCAB, (long)(mlen >> 10));
    appends(info);
    appends(tok.kind == NSLM_TOK_BL ? "Ask a question and press enter.\nEnter on an empty line: new conversation.\n"
                                    : "Type the start of a story and press enter.\n");
    idle_status();
    draw();

    for (;;) {
        int k = plat_key();
        if (k == K_NONE) continue;
        if (k == K_ESC) break;
        if (k == K_ENTER) {
            if (tok.kind == NSLM_TOK_BL) chat(input);
            else generate(input);
            input_len = 0;
            input[0] = 0;
        } else if (k == K_DEL) {
            if (input_len) input[--input_len] = 0;
        } else if (k == K_TAB) {
            preset = (preset + 1) % (int)(sizeof presets / sizeof presets[0]);
            idle_status();
        } else if (k == K_MENU) {
            selftest();
        } else if (k >= 32 && k < 127 && input_len + 1 < (int)sizeof input) {
            input[input_len++] = (char)k;
            input[input_len] = 0;
        }
        draw();
    }
    plat_exit();
    return 0;
}
