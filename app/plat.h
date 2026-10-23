/* Platform seam for the NspireLM app: the calculator (plat_nspire.c) and a headless host
 * build (plat_host.c) that replays scripted keys and saves frames, so the UI is testable. */
#ifndef NSLM_PLAT_H
#define NSLM_PLAT_H

#include <stddef.h>
#include <stdint.h>

#define SCR_W 320
#define SCR_H 240

enum {
    K_NONE = 0,
    K_DEL = 8,
    K_TAB = 9,
    K_ENTER = 10,
    K_ESC = 27,
    K_MENU = 0x100,
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_DOWN,
};

int plat_init(int argc, char **argv);
void plat_exit(void);
uint64_t plat_us(void);          /* monotonic microseconds */
int plat_cpu_mhz(void);          /* 0 if unknown */
int plat_key(void);              /* next key press (edge), K_NONE if none */
int plat_abort(void);            /* polled while generating: 1 if ESC is held */
void plat_present(const uint16_t *fb);
/* Loads a whole file into a 32-byte aligned buffer (never freed: lives until exit). */
void *plat_load(const char *name, size_t *len, void (*progress)(int pct));
int plat_write(const char *name, const char *text);
/* Real path of a data file next to the program ("model.nslm" -> "model.nslm.tns" on the device). */
const char *plat_path(const char *name, char *buf, size_t n);
void *plat_aligned_alloc(size_t bytes); /* 32-byte aligned, zeroed */
void plat_aligned_free(void *p);
int plat_battery_ok(void);              /* 1 = full speed (not on USB), 0 = throttled, -1 unknown */

#endif
