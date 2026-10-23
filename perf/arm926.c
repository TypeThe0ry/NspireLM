/* arm926: a qemu TCG plugin that estimates ARM926EJ-S cycle counts for ARM-mode code.
 *
 * Model (ARM9EJ-S TRM DDI0222B table 8-2; ARM926EJ-S cache geometry):
 *   - per-instruction issue cycles by instruction class, decided at translation time
 *   - load-use and multiply-result interlocks with the next one/two instructions in the block
 *     (the multiply-accumulate chain forwarding case is exempt, as on the core)
 *   - taken branches cost 3 cycles, untaken conditional branches 1 (resolved dynamically)
 *   - 16 KB 4-way I-cache and 8 KB 4-way D-cache, 32-byte lines, round-robin replacement,
 *     write-back, read-allocate (store misses go to the write buffer)
 *   - memory penalties are parameters, calibrated against on-device measurements
 *
 * Arguments: mark=0xADDR (each execution of the block at ADDR prints counters),
 *            rmiss=N (D read-miss cycles), imiss=N, wmiss=N (store miss), wb=N (dirty eviction)
 *            lc=N beat=N hum=0|1 wbus=N: timed linefill model (replaces rmiss/imiss)
 *            prow=N rowbits=N banks=N: DRAM open-row model (closed-row fills pay prow)
 *            prof=FILE: write "block_addr cycles" for the span between mark 0 and mark 1
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static uint64_t cycles, insns, dmiss, imiss_n, dacc, wmiss_n, wb_n, marks;
static uint64_t mark_addr;
static int p_rmiss = 24, p_imiss = 24, p_wmiss = 1, p_wb = 8;
/* timed line-fill model (enabled with lc=N): critical word after lc cycles, then one word every
 * beat cycles (WRAP8 burst); the bus is busy until the fill ends. hum=0: any other D-cache access
 * during a fill waits for it to finish (blocking cache); hum=1: hits proceed under the miss. */
static int p_lc = 0, p_beat = 0, p_hum = 0, p_wbus = 0;
static uint64_t tb_t0, tb_stall, bus_free, fill_t0, fill_done;
static uint32_t fill_line = 0xFFFFFFFFu, fill_crit;
/* DRAM row buffers: rowbits = log2(row bytes), banks open rows; a fill to a closed row pays prow */
static int p_rowbits = 11, p_banks = 4, p_prow = 0;
static uint32_t open_row[16];
static uint64_t rowmiss_n;
static uint64_t st_word, st_block, st_crit, st_bus; /* stall breakdown */

static uint64_t dram_extra(uint32_t addr) {
    if (!p_prow) return 0;
    uint32_t row = addr >> p_rowbits;
    uint32_t bank = row % (uint32_t)p_banks;
    if (open_row[bank] == row + 1) return 0;
    open_row[bank] = row + 1;
    rowmiss_n++;
    return (uint64_t)p_prow;
}

/* ------------------------------------------------------------------ caches */

typedef struct {
    int sets, ways;
    uint32_t *tag; /* sets*ways, 0 = invalid (tags stored as line|1) */
    uint8_t *dirty;
    uint8_t *rr;
} cache_t;

static cache_t dc, ic;

static void cache_init(cache_t *c, int bytes, int ways) {
    c->ways = ways;
    c->sets = bytes / 32 / ways;
    c->tag = calloc((size_t)c->sets * ways, sizeof *c->tag);
    c->dirty = calloc((size_t)c->sets * ways, 1);
    c->rr = calloc((size_t)c->sets, 1);
}

/* returns: 0 hit, 1 miss (allocated), 2 miss with dirty eviction; alloc=0 -> no allocate */
static int cache_access(cache_t *c, uint32_t addr, int write, int alloc) {
    uint32_t line = addr >> 5;
    int set = (int)(line % (uint32_t)c->sets);
    uint32_t t = (line << 1) | 1;
    uint32_t *row = c->tag + set * c->ways;
    for (int w = 0; w < c->ways; w++) {
        if (row[w] == t) {
            if (write) c->dirty[set * c->ways + w] = 1;
            return 0;
        }
    }
    if (!alloc) return 1;
    int v = c->rr[set];
    c->rr[set] = (uint8_t)((v + 1) % c->ways);
    int ev = c->dirty[set * c->ways + v] && row[v];
    row[v] = t;
    c->dirty[set * c->ways + v] = (uint8_t)write;
    return ev ? 2 : 1;
}

/* ------------------------------------------------------------------ decode */

enum { K_OTHER, K_LOADW, K_LOADB, K_LOADD, K_LDM, K_MUL, K_MULACC, K_BRANCH };

typedef struct {
    int cost;      /* issue cycles */
    int kind;
    int dst[2];    /* result registers (-1 none) */
    int is_branch; /* ends the block with a possible change of flow */
    uint32_t raw;
} dec_t;

static int popc(uint32_t v) { return __builtin_popcount(v); }

static dec_t decode(uint32_t i) {
    dec_t d = {1, K_OTHER, {-1, -1}, 0, i};
    uint32_t cond = i >> 28, op = (i >> 25) & 7;
    int rd = (int)((i >> 12) & 15), rn = (int)((i >> 16) & 15);
    if (cond == 0xF) {
        if ((i & 0x0E000000) == 0x0A000000) { d.cost = 3; d.is_branch = 1; } /* BLX imm */
        return d; /* PLD and friends: 1 */
    }
    switch (op) {
    case 0:
    case 1:
        if ((i & 0x0FC000F0) == 0x00000090) { /* MUL/MLA */
            d.cost = (i >> 20 & 1) ? 4 : 2;
            d.kind = (i >> 21 & 1) ? K_MULACC : K_MUL;
            d.dst[0] = rn;
            return d;
        }
        if ((i & 0x0F8000F0) == 0x00800090) { /* long multiply */
            d.cost = (i >> 20 & 1) ? 5 : 3;
            d.kind = (i >> 21 & 1) ? K_MULACC : K_MUL;
            d.dst[0] = rn;
            d.dst[1] = rd;
            return d;
        }
        if ((i & 0x0F900090) == 0x01000080) { /* SMLAxy / SMLAWy / SMULWy / SMLALxy / SMULxy */
            int o = (int)((i >> 21) & 3);
            d.cost = o == 2 ? 2 : 1;
            d.kind = (o == 0 || o == 2 || (o == 1 && !(i & 0x20))) ? K_MULACC : K_MUL;
            d.dst[0] = rn;
            if (o == 2) d.dst[1] = rd;
            return d;
        }
        if ((i & 0x0F9000F0) == 0x01000050) { d.kind = K_MUL; d.dst[0] = rd; return d; } /* Q ops */
        if ((i & 0x0FFFFFD0) == 0x012FFF10) { d.cost = 3; d.is_branch = 1; return d; }  /* BX/BLX */
        if ((i & 0x0FB00FF0) == 0x01000090) { d.cost = 2; d.kind = K_LOADW; d.dst[0] = rd; return d; }
        if ((i & 0x0E000090) == 0x00000090 && (i & 0x60)) { /* extra load/store */
            int L = (int)(i >> 20 & 1), sh = (int)((i >> 5) & 3);
            if (L) { d.kind = K_LOADB; d.dst[0] = rd; }
            else if (sh == 2) { d.cost = 2; d.kind = K_LOADD; d.dst[0] = rd; d.dst[1] = rd + 1; }
            else if (sh == 3) { d.cost = 2; }
            return d;
        }
        if ((i & 0x0FB000F0) == 0x01000000 || (i & 0x0DB0F000) == 0x0120F000) return d; /* MRS/MSR */
        {
            int opc = (int)((i >> 21) & 15);
            int cmp = opc >= 8 && opc <= 11;
            if (op == 0 && (i & 0x10)) d.cost = 2; /* register-specified shift */
            if (!cmp) d.dst[0] = rd;
            if (!cmp && rd == 15) { d.cost += 2; d.is_branch = 1; }
        }
        return d;
    case 2:
    case 3: {
        if (op == 3 && (i & 0x10)) return d; /* media / undefined on v5 */
        int L = (int)(i >> 20 & 1), B = (int)(i >> 22 & 1);
        if (op == 3 && (i & 0xFF0)) d.cost = 2; /* scaled register offset */
        if (L) {
            d.kind = B ? K_LOADB : K_LOADW;
            d.dst[0] = rd;
            if (rd == 15) { d.cost = 5; d.is_branch = 1; }
        }
        return d;
    }
    case 4: {
        int n = popc(i & 0xFFFF);
        d.cost = n > 1 ? n : 1;
        if (i >> 20 & 1) {
            d.kind = K_LDM;
            int last = 31 - __builtin_clz(i & 0xFFFF);
            d.dst[0] = last;
            if (i & 0x8000) { d.cost += 4; d.is_branch = 1; }
        }
        return d;
    }
    case 5:
        d.is_branch = 1; /* +2 if taken, resolved at run time */
        d.kind = K_BRANCH;
        return d;
    default:
        return d;
    }
}

/* Does instruction i read register r (as an operand needed early)? acc_only: set if the only use
 * is the accumulator of a multiply-accumulate (forwarded without a stall on ARM9E). */
static int reads(uint32_t i, int r, int *acc_only) {
    uint32_t op = (i >> 25) & 7;
    int rm = (int)(i & 15), rs = (int)((i >> 8) & 15), rd = (int)((i >> 12) & 15), rn = (int)((i >> 16) & 15);
    *acc_only = 0;
    if ((i >> 28) == 0xF) return (i & 0x0D700000) == 0x05500000 && rn == r; /* PLD base */
    if (op == 0 || op == 1) {
        if ((i & 0x0FC000F0) == 0x00000090) {
            if (rm == r || rs == r) return 1;
            if ((i >> 21 & 1) && rd == r) { *acc_only = 1; return 1; }
            return 0;
        }
        if ((i & 0x0F8000F0) == 0x00800090) {
            if (rm == r || rs == r) return 1;
            if ((i >> 21 & 1) && (rd == r || rn == r)) { *acc_only = 1; return 1; }
            return 0;
        }
        if ((i & 0x0F900090) == 0x01000080) {
            int o = (int)((i >> 21) & 3);
            if (rm == r || rs == r) return 1;
            if ((o == 0 || (o == 1 && !(i & 0x20))) && rd == r) { *acc_only = 1; return 1; }
            if (o == 2 && (rd == r || rn == r)) { *acc_only = 1; return 1; }
            return 0;
        }
        if ((i & 0x0F9000F0) == 0x01000050) return rm == r || rn == r;
        if ((i & 0x0FFFFFD0) == 0x012FFF10) return rm == r;
        if ((i & 0x0E000090) == 0x00000090 && (i & 0x60)) {
            int L = (int)(i >> 20 & 1), sh = (int)((i >> 5) & 3);
            if (rn == r || (!(i >> 22 & 1) && rm == r)) return 1;
            if (!L && sh != 2 && (rd == r || (sh == 3 && rd + 1 == r))) return 1;
            return 0;
        }
        int opc = (int)((i >> 21) & 15);
        if (opc != 13 && opc != 15 && rn == r) return 1;
        if (!(i >> 25 & 1)) {
            if (rm == r) return 1;
            if ((i & 0x10) && rs == r) return 1;
        }
        return 0;
    }
    if (op == 2 || op == 3) {
        if (rn == r) return 1;
        if (op == 3 && rm == r) return 1;
        if (!(i >> 20 & 1) && rd == r) return 1;
        return 0;
    }
    if (op == 4) {
        if (rn == r) return 1;
        if (!(i >> 20 & 1) && (i >> r & 1)) return 1;
        return 0;
    }
    return 0;
}

static int interlock(const dec_t *p, uint32_t next, uint32_t next2) {
    int a, extra = 0;
    for (int k = 0; k < 2; k++) {
        int r = p->dst[k];
        if (r < 0) continue;
        switch (p->kind) {
        case K_LOADW:
        case K_LDM:
            if (reads(next, r, &a)) extra = extra > 1 ? extra : 1;
            break;
        case K_LOADB:
            if (reads(next, r, &a)) extra = 2;
            else if (reads(next2, r, &a)) extra = extra > 1 ? extra : 1;
            break;
        case K_LOADD:
            if (k == 1 && reads(next, r, &a)) extra = extra > 1 ? extra : 1;
            break;
        case K_MUL:
        case K_MULACC:
            if (reads(next, r, &a) && !a) extra = extra > 1 ? extra : 1;
            break;
        default:
            break;
        }
    }
    return extra;
}

/* ------------------------------------------------------------------ plugin glue */

typedef struct tbinfo {
    uint64_t start, fall;
    uint32_t first_line, last_line;
    int cost, n, ends_branch;
    uint64_t prof; /* cycles attributed to this block while profiling */
    struct tbinfo *next;
} tbinfo_t;

static const tbinfo_t *prev_tb;
static tbinfo_t *all_tbs, *cur_tb;
static int p_prof, profiling;
static uint64_t prof_t0;
static const char *p_proffile;

static void tb_exec(unsigned int vcpu, void *ud) {
    (void)vcpu;
    tbinfo_t *t = ud;
    if (profiling && cur_tb) cur_tb->prof += cycles - prof_t0;
    cur_tb = t;
    prof_t0 = cycles;
    if (prev_tb && prev_tb->ends_branch && t->start != prev_tb->fall) cycles += 2;
    prev_tb = t;
    for (uint32_t l = t->first_line; l <= t->last_line; l++)
        if (cache_access(&ic, l << 5, 0, 1)) {
            imiss_n++;
            if (p_lc) { /* instruction fetch waits for the bus, then the whole line */
                uint64_t start = (cycles > bus_free ? cycles : bus_free) + dram_extra(l << 5);
                uint64_t done = start + (uint64_t)p_lc + 7u * (uint64_t)p_beat;
                bus_free = done;
                cycles = done;
            } else {
                cycles += (uint64_t)p_imiss;
            }
        }
    tb_t0 = cycles;
    tb_stall = 0;
    cycles += (uint64_t)t->cost;
    insns += (uint64_t)t->n;
    if (t->start == mark_addr) {
        /* profile between the first and second mark */
        if (p_prof) profiling = marks == 0;
        char buf[512];
        snprintf(buf, sizeof buf, "MARK %" PRIu64 " cycles=%" PRIu64 " insns=%" PRIu64 " dacc=%" PRIu64
                 " dmiss=%" PRIu64 " imiss=%" PRIu64 " wmiss=%" PRIu64 " wb=%" PRIu64 " rowmiss=%" PRIu64 " st_word=%" PRIu64
                 " st_block=%" PRIu64 " st_crit=%" PRIu64 " st_bus=%" PRIu64 "\n",
                 marks++, cycles, insns, dacc, dmiss, imiss_n, wmiss_n, wb_n, rowmiss_n, st_word, st_block,
                 st_crit, st_bus);
        qemu_plugin_outs(buf);
    }
}

static void stall(uint64_t c) {
    cycles += c;
    tb_stall += c;
}

static void mem_cb(unsigned int vcpu, qemu_plugin_meminfo_t info, uint64_t vaddr, void *ud) {
    (void)vcpu;
    dacc++;
    int store = qemu_plugin_mem_is_store(info);
    if (!p_lc) { /* flat penalties */
        if (store) {
            if (cache_access(&dc, (uint32_t)vaddr, 1, 0)) { wmiss_n++; cycles += (uint64_t)p_wmiss; }
        } else {
            int r = cache_access(&dc, (uint32_t)vaddr, 0, 1);
            if (r) { dmiss++; cycles += (uint64_t)p_rmiss; }
            if (r == 2) { wb_n++; cycles += (uint64_t)p_wb; }
        }
        return;
    }
    uint64_t t = tb_t0 + (uint64_t)(uintptr_t)ud + tb_stall;
    uint32_t line = (uint32_t)vaddr >> 5;
    if (t < fill_done) { /* a linefill is in flight */
        if (line == fill_line) {
            uint32_t w = ((uint32_t)vaddr >> 2) & 7, k = (w - fill_crit) & 7;
            uint64_t avail = fill_t0 + (uint64_t)p_lc + (uint64_t)k * (uint64_t)p_beat;
            if (avail > t) { stall(avail - t); st_word += avail - t; }
            return;
        }
        if (!p_hum) { stall(fill_done - t); st_block += fill_done - t; t = fill_done; }
    }
    if (store) {
        if (cache_access(&dc, (uint32_t)vaddr, 1, 0)) {
            wmiss_n++;
            cycles += (uint64_t)p_wmiss;
            uint64_t start = t > bus_free ? t : bus_free;
            bus_free = start + (uint64_t)p_wbus;
        }
        return;
    }
    int r = cache_access(&dc, (uint32_t)vaddr, 0, 1);
    if (!r) return;
    dmiss++;
    uint64_t start = t > bus_free ? t : bus_free;
    if (r == 2) { wb_n++; start += (uint64_t)p_wb; }
    start += dram_extra((uint32_t)vaddr);
    fill_line = line;
    fill_crit = ((uint32_t)vaddr >> 2) & 7;
    fill_t0 = start;
    fill_done = start + (uint64_t)p_lc + 7u * (uint64_t)p_beat;
    bus_free = fill_done;
    stall(start - t + (uint64_t)p_lc);
    st_bus += start - t;
    st_crit += (uint64_t)p_lc;
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
    (void)id;
    size_t n = qemu_plugin_tb_n_insns(tb);
    tbinfo_t *t = calloc(1, sizeof *t);
    uint32_t *raw = calloc(n + 2, sizeof *raw);
    for (size_t k = 0; k < n; k++) {
        struct qemu_plugin_insn *in = qemu_plugin_tb_get_insn(tb, k);
        uint32_t w = 0;
        if (qemu_plugin_insn_size(in) == 4) qemu_plugin_insn_data(in, &w, 4);
        raw[k] = w;
    }
    int cost = 0;
    dec_t last = {0};
    for (size_t k = 0; k < n; k++) {
        dec_t d = decode(raw[k]);
        qemu_plugin_register_vcpu_mem_cb(qemu_plugin_tb_get_insn(tb, k), mem_cb, QEMU_PLUGIN_CB_NO_REGS,
                                         QEMU_PLUGIN_MEM_RW, (void *)(uintptr_t)cost);
        cost += d.cost + interlock(&d, raw[k + 1], raw[k + 2]);
        last = d;
    }
    t->start = qemu_plugin_tb_vaddr(tb);
    uint64_t lastpc = qemu_plugin_insn_vaddr(qemu_plugin_tb_get_insn(tb, n - 1));
    t->fall = lastpc + 4;
    t->first_line = (uint32_t)(t->start >> 5);
    t->last_line = (uint32_t)(lastpc >> 5);
    t->cost = cost;
    t->n = (int)n;
    t->ends_branch = last.kind == K_BRANCH; /* other flow changes are costed statically */
    t->next = all_tbs;
    all_tbs = t;
    free(raw);
    qemu_plugin_register_vcpu_tb_exec_cb(tb, tb_exec, QEMU_PLUGIN_CB_NO_REGS, t);
}

static void at_exit(qemu_plugin_id_t id, void *p) {
    (void)id;
    (void)p;
    char buf[512];
    snprintf(buf, sizeof buf, "TOTAL cycles=%" PRIu64 " insns=%" PRIu64 " dacc=%" PRIu64 " dmiss=%" PRIu64
             " imiss=%" PRIu64 " wmiss=%" PRIu64 " wb=%" PRIu64 "\n", cycles, insns, dacc, dmiss, imiss_n,
             wmiss_n, wb_n);
    qemu_plugin_outs(buf);
    if (p_prof && p_proffile) {
        FILE *f = fopen(p_proffile, "w");
        for (tbinfo_t *t = all_tbs; t && f; t = t->next)
            if (t->prof) fprintf(f, "%" PRIx64 " %" PRIu64 "\n", t->start, t->prof);
        if (f) fclose(f);
    }
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc,
                                           char **argv) {
    (void)info;
    for (int k = 0; k < argc; k++) {
        char *v = strchr(argv[k], '=');
        if (!v) continue;
        *v++ = 0;
        if (!strcmp(argv[k], "mark")) mark_addr = strtoull(v, 0, 0);
        else if (!strcmp(argv[k], "rmiss")) p_rmiss = atoi(v);
        else if (!strcmp(argv[k], "imiss")) p_imiss = atoi(v);
        else if (!strcmp(argv[k], "wmiss")) p_wmiss = atoi(v);
        else if (!strcmp(argv[k], "wb")) p_wb = atoi(v);
        else if (!strcmp(argv[k], "lc")) p_lc = atoi(v);
        else if (!strcmp(argv[k], "beat")) p_beat = atoi(v);
        else if (!strcmp(argv[k], "hum")) p_hum = atoi(v);
        else if (!strcmp(argv[k], "wbus")) p_wbus = atoi(v);
        else if (!strcmp(argv[k], "prow")) p_prow = atoi(v);
        else if (!strcmp(argv[k], "prof")) { p_prof = 1; p_proffile = strdup(v); }
        else if (!strcmp(argv[k], "rowbits")) p_rowbits = atoi(v);
        else if (!strcmp(argv[k], "banks")) p_banks = atoi(v);
    }
    cache_init(&dc, 8192, 4);
    cache_init(&ic, 16384, 4);
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
