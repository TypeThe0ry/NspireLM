/* Headless host platform: keys come from $NSLM_KEYS, files from $NSLM_DIR, and every {SHOT}
 * in the key script saves the current frame as shot_N.ppm in $NSLM_OUT. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "plat.h"

static const char *keys = "";
static const char *dir = ".", *outdir = ".";
static const uint16_t *last_fb;
static int shots;

int plat_init(int argc, char **argv) {
    (void)argc;
    (void)argv;
    if (getenv("NSLM_KEYS")) keys = getenv("NSLM_KEYS");
    if (getenv("NSLM_DIR")) dir = getenv("NSLM_DIR");
    if (getenv("NSLM_OUT")) outdir = getenv("NSLM_OUT");
    return 0;
}

void plat_exit(void) {}

uint64_t plat_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

int plat_cpu_mhz(void) { return 0; }
int plat_battery_ok(void) { return -1; }
/* NSLM_ABORT_AFTER=N: behave as if ESC were held once a generation has polled N times (keeps scripted
 * runs of models that never stop short) */
static int abort_polls;
int plat_abort(void) {
    const char *e = getenv("NSLM_ABORT_AFTER");
    return e && ++abort_polls > atoi(e);
}

static void shot(void) {
    if (!last_fb) return;
    char path[512];
    snprintf(path, sizeof path, "%s/shot_%d.ppm", outdir, shots++);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", SCR_W, SCR_H);
    for (int i = 0; i < SCR_W * SCR_H; i++) {
        uint16_t c = last_fb[i];
        unsigned char rgb[3] = {(unsigned char)((c >> 11) << 3), (unsigned char)(((c >> 5) & 63) << 2),
                                (unsigned char)((c & 31) << 3)};
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

int plat_key(void) {
    abort_polls = 0;
    static const struct { const char *name; int key; } named[] = {
        {"{ENTER}", K_ENTER}, {"{ESC}", K_ESC}, {"{MENU}", K_MENU}, {"{TAB}", K_TAB}, {"{DEL}", K_DEL}};
    for (;;) {
        if (!*keys) return K_ESC; /* script finished: quit */
        if (!strncmp(keys, "{SHOT}", 6)) {
            keys += 6;
            shot();
            continue;
        }
        for (size_t i = 0; i < sizeof named / sizeof named[0]; i++) {
            size_t n = strlen(named[i].name);
            if (!strncmp(keys, named[i].name, n)) {
                keys += n;
                return named[i].key;
            }
        }
        return (unsigned char)*keys++;
    }
}

void plat_present(const uint16_t *fb) { last_fb = fb; }

void *plat_aligned_alloc(size_t bytes) {
    void *p = 0;
    if (posix_memalign(&p, 32, bytes ? bytes : 32)) return 0;
    memset(p, 0, bytes);
    return p;
}

void plat_aligned_free(void *p) { free(p); }

void *plat_load(const char *name, size_t *len, void (*progress)(int pct)) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *p = plat_aligned_alloc(*len);
    size_t done = 0;
    while (p && done < *len) {
        size_t got = fread(p + done, 1, *len - done > 65536 ? 65536 : *len - done, f);
        if (!got) break;
        done += got;
        if (progress) progress((int)(done * 100 / *len));
    }
    fclose(f);
    return done == *len ? p : 0;
}

const char *plat_path(const char *name, char *buf, size_t n) {
    snprintf(buf, n, "%s/%s", dir, name);
    return buf;
}

int plat_write(const char *name, const char *text) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", outdir, name);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fputs(text, f);
    fclose(f);
    return 0;
}

int nslm_app_main(int argc, char **argv, int hash_only);

int main(int argc, char **argv) {
    return nslm_app_main(argc, argv, argc > 1 && !strcmp(argv[1], "--hash"));
}
