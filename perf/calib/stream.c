#include <stdint.h>
#include <stdlib.h>
#include <string.h>
__attribute__((noinline)) void perf_mark(int id){ __asm__ volatile("" ::: "memory"); (void)id; }
static uint32_t stream_read_sum(const uint32_t *p, uint32_t words) {
    uint32_t acc = 0;
    for (uint32_t i = 0; i < words; i += 8)
        acc += p[i+0] + p[i+1] + p[i+2] + p[i+3] + p[i+4] + p[i+5] + p[i+6] + p[i+7];
    return acc;
}
int main(void){
    uint32_t n = 4096*1024; uint32_t *buf = malloc(n);
    for (uint32_t i=0;i<n/4;i++) buf[i]=i;
    perf_mark(0); uint32_t acc=0;
    for (int r=0;r<4;r++) acc += stream_read_sum(buf, n/4);
    perf_mark(1);
    __asm__ volatile("" :: "r"(acc) : "memory");
    for (int r=0;r<4;r++) memcpy((char*)buf+n/2, buf, n/2);
    perf_mark(2);
    return (int)(acc&1);
}
