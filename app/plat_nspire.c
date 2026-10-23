/* TI-Nspire CX II platform (Ndless). File loading, timing and clock decoding follow what ChatTLM
 * measured on this hardware (github.com/alex-messerlian/nspire-tlm, src/nspire.c):
 *   - plain fopen() + 64 KB fread() chunks is the read path that works (fdopen hangs, small reads
 *     are minutes slower on the 1.5 MB/s NAND)
 *   - SP804 timer 0 at 0x90010000 runs from the 99 MHz APB clock
 *   - CPU clock = 2 x the AHB clock decoded from the PMU at 0x90140000 (396 MHz on battery,
 *     288 MHz while USB is attached) */
#include <keys.h>
#include <libndls.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plat.h"

#define MMIO32(a) (*(volatile uint32_t *)(uintptr_t)(a))
#define TIMER_BASE 0x90010000u
#define PMU_CLK_MAIN 0x90140030u
#define PMU_CLK_DIV2 0x90140020u
#define PMU_CLK_GATE 0x90140810u
#define TIMER_HZ 99000000u

static uint32_t last_raw;
static uint64_t ticks;

int plat_init(int argc, char **argv) {
    (void)argc;
    enable_relative_paths(argv);
    uint32_t ctrl = MMIO32(TIMER_BASE + 0x08);
    const uint32_t want = (1u << 7) | (1u << 1); /* enabled, 32-bit, free running */
    if ((ctrl & want) != want) {
        MMIO32(TIMER_BASE + 0x08) = 0;
        MMIO32(TIMER_BASE + 0x00) = 0xFFFFFFFFu;
        MMIO32(TIMER_BASE + 0x08) = want;
    }
    last_raw = MMIO32(TIMER_BASE + 0x04);
    if (!lcd_init(SCR_320x240_565)) return -1;
    return 0;
}

void plat_exit(void) { lcd_init(SCR_TYPE_INVALID); }

uint64_t plat_us(void) {
    uint32_t raw = MMIO32(TIMER_BASE + 0x04);
    ticks += (uint32_t)(last_raw - raw); /* down counter, wraps */
    last_raw = raw;
    return ticks / (TIMER_HZ / 1000000u);
}

int plat_cpu_mhz(void) {
    uint32_t main = MMIO32(PMU_CLK_MAIN);
    uint32_t mult = (main >> 24) & 0xFFu, div1 = (main >> 16) & 0x3Fu;
    if (!div1 || !mult) return 0;
    uint32_t hz = (uint32_t)(((uint64_t)12000000u * mult) / div1);
    if (((MMIO32(PMU_CLK_GATE) >> 4) & 1u) && !((main >> 4) & 1u))
        hz /= (((MMIO32(PMU_CLK_DIV2) >> 20) & 0xFu) + 1u);
    return (int)(hz * 2u / 1000000u);
}

int plat_battery_ok(void) {
    int mhz = plat_cpu_mhz();
    return mhz ? mhz >= 390 : -1;
}

/* ------------------------------------------------------------------ keys */

static const struct {
    const t_key *key;
    int plain, shifted;
} keymap[] = {
    {&KEY_NSPIRE_A, 'a', 'A'}, {&KEY_NSPIRE_B, 'b', 'B'}, {&KEY_NSPIRE_C, 'c', 'C'},
    {&KEY_NSPIRE_D, 'd', 'D'}, {&KEY_NSPIRE_E, 'e', 'E'}, {&KEY_NSPIRE_F, 'f', 'F'},
    {&KEY_NSPIRE_G, 'g', 'G'}, {&KEY_NSPIRE_H, 'h', 'H'}, {&KEY_NSPIRE_I, 'i', 'I'},
    {&KEY_NSPIRE_J, 'j', 'J'}, {&KEY_NSPIRE_K, 'k', 'K'}, {&KEY_NSPIRE_L, 'l', 'L'},
    {&KEY_NSPIRE_M, 'm', 'M'}, {&KEY_NSPIRE_N, 'n', 'N'}, {&KEY_NSPIRE_O, 'o', 'O'},
    {&KEY_NSPIRE_P, 'p', 'P'}, {&KEY_NSPIRE_Q, 'q', 'Q'}, {&KEY_NSPIRE_R, 'r', 'R'},
    {&KEY_NSPIRE_S, 's', 'S'}, {&KEY_NSPIRE_T, 't', 'T'}, {&KEY_NSPIRE_U, 'u', 'U'},
    {&KEY_NSPIRE_V, 'v', 'V'}, {&KEY_NSPIRE_W, 'w', 'W'}, {&KEY_NSPIRE_X, 'x', 'X'},
    {&KEY_NSPIRE_Y, 'y', 'Y'}, {&KEY_NSPIRE_Z, 'z', 'Z'},
    {&KEY_NSPIRE_0, '0', ')'}, {&KEY_NSPIRE_1, '1', '!'}, {&KEY_NSPIRE_2, '2', '@'},
    {&KEY_NSPIRE_3, '3', '#'}, {&KEY_NSPIRE_4, '4', '$'}, {&KEY_NSPIRE_5, '5', '%'},
    {&KEY_NSPIRE_6, '6', '^'}, {&KEY_NSPIRE_7, '7', '&'}, {&KEY_NSPIRE_8, '8', '*'},
    {&KEY_NSPIRE_9, '9', '('},
    {&KEY_NSPIRE_SPACE, ' ', ' '}, {&KEY_NSPIRE_PERIOD, '.', '>'}, {&KEY_NSPIRE_COMMA, ',', '<'},
    {&KEY_NSPIRE_QUESEXCL, '?', '!'}, {&KEY_NSPIRE_APOSTROPHE, '\'', '"'}, {&KEY_NSPIRE_QUOTE, '"', '"'},
    {&KEY_NSPIRE_COLON, ':', ';'}, {&KEY_NSPIRE_MINUS, '-', '_'}, {&KEY_NSPIRE_NEGATIVE, '-', '_'},
    {&KEY_NSPIRE_PLUS, '+', '+'}, {&KEY_NSPIRE_MULTIPLY, '*', '*'}, {&KEY_NSPIRE_DIVIDE, '/', '/'},
    {&KEY_NSPIRE_EQU, '=', '='}, {&KEY_NSPIRE_LP, '(', '('}, {&KEY_NSPIRE_RP, ')', ')'},
    {&KEY_NSPIRE_ENTER, K_ENTER, K_ENTER}, {&KEY_NSPIRE_RET, K_ENTER, K_ENTER},
    {&KEY_NSPIRE_DEL, K_DEL, K_DEL}, {&KEY_NSPIRE_ESC, K_ESC, K_ESC}, {&KEY_NSPIRE_MENU, K_MENU, K_MENU},
    {&KEY_NSPIRE_TAB, K_TAB, K_TAB}, {&KEY_NSPIRE_LEFT, K_LEFT, K_LEFT},
    {&KEY_NSPIRE_RIGHT, K_RIGHT, K_RIGHT}, {&KEY_NSPIRE_UP, K_UP, K_UP}, {&KEY_NSPIRE_DOWN, K_DOWN, K_DOWN},
};
#define NKEYS (int)(sizeof keymap / sizeof keymap[0])
static uint8_t was_down[NKEYS];

int plat_key(void) {
    int shift = isKeyPressed(KEY_NSPIRE_SHIFT) || isKeyPressed(KEY_NSPIRE_CTRL);
    int hit = K_NONE;
    for (int i = 0; i < NKEYS; i++) {
        int down = isKeyPressed(*keymap[i].key) ? 1 : 0;
        if (down && !was_down[i] && hit == K_NONE) hit = shift ? keymap[i].shifted : keymap[i].plain;
        was_down[i] = (uint8_t)down;
    }
    if (hit == K_NONE) msleep(10);
    return hit;
}

int plat_abort(void) { return isKeyPressed(KEY_NSPIRE_ESC) ? 1 : 0; }

void plat_present(const uint16_t *fb) { lcd_blit((void *)fb, SCR_320x240_565); }

/* ------------------------------------------------------------------ memory and files */

/* The OS malloc only guarantees 8-byte alignment; keep the raw pointer just below the aligned block. */
void *plat_aligned_alloc(size_t bytes) {
    unsigned char *raw = malloc(bytes + 32 + sizeof(void *));
    if (!raw) return 0;
    unsigned char *a = (unsigned char *)(((uintptr_t)raw + sizeof(void *) + 31) & ~(uintptr_t)31);
    ((void **)a)[-1] = raw;
    memset(a, 0, bytes);
    return a;
}

void plat_aligned_free(void *p) {
    if (p) free(((void **)p)[-1]);
}

void *plat_load(const char *name, size_t *len, void (*progress)(int pct)) {
    char path[256];
    snprintf(path, sizeof path, "%s.tns", name); /* the OS only stores .tns files */
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) {
        fclose(f);
        return 0;
    }
    unsigned char *p = plat_aligned_alloc((size_t)n);
    size_t done = 0;
    int last = -1;
    while (p && done < (size_t)n) {
        size_t want = (size_t)n - done > 65536u ? 65536u : (size_t)n - done;
        size_t got = fread(p + done, 1, want, f);
        if (!got) break;
        done += got;
        int pct = (int)((uint64_t)done * 100u / (uint64_t)n);
        if (progress && pct / 2 != last / 2) progress(pct);
        last = pct;
    }
    fclose(f);
    if (!p || done != (size_t)n) return 0;
    *len = done;
    return p;
}

const char *plat_path(const char *name, char *buf, size_t n) {
    snprintf(buf, n, "%s.tns", name);
    return buf;
}

int plat_write(const char *name, const char *text) {
    char path[256];
    snprintf(path, sizeof path, "%s.tns", name);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fputs(text, f);
    fclose(f);
    return 0;
}

int nslm_app_main(int argc, char **argv, int hash_only);

int main(int argc, char **argv) { return nslm_app_main(argc, argv, 0); }
