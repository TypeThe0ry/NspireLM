/* calibration driver: includes ChatTLM's engine verbatim */
#define main tlm_main
#include CHATTLM_SRC
#undef main
__attribute__((noinline)) void perf_mark(int id){ __asm__ volatile("" ::: "memory"); (void)id; }
int main(int argc, char **argv) {
    static Transformer t;
    build_transformer(&t, argv[1]);
    (void)argc;
    /* cost depends on geometry and position only; warm once, then time pos 8 and pos 256 */
    forward(&t, 5, 7);
    perf_mark(0); forward(&t, 17, 8);
    perf_mark(1); forward(&t, 23, 256);
    perf_mark(2);
    return 0;
}
