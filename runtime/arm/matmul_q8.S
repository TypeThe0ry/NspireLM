@ NspireLM int8 x int16 matrix-vector kernel for ARM926EJ-S (ARMv5TE), x4 weight layout.
@
@ void nslm_q8_dot_x4(int32_t *acc, const uint8_t *q, const uint16_t *m, const int16_t *a,
@                     const int32_t *asum, int row_blocks, int groups, int chunks_per_group);
@
@ For every row r: acc[r] = sum_g ((d_rg * m_rg) >> 16), with d_rg = sum_{i in g} (q_ri - 128) * a_i,
@ exactly as the C reference in nslm_kern.c computes it.
@
@ One weight cache line = 8 columns x 4 rows, offset-binary bytes. The 8 activations for those columns
@ are loaded into registers (LDM, cache hits) BEFORE the weight line is touched. Everything after the
@ first weight load is then register work: on a linefill that delivers the critical word first, the
@ unpack + multiply work overlaps the arrival of the rest of the line. One AND with 0x00FF00FF puts two
@ bytes into the two 16-bit lanes that SMLAxy reads. Compute is 60 issue cycles per 32 MACs.
@
@ Register use in the inner loop:
@   r0 weights   r1 activations   r2 0x00FF00FF   r3 chunk counter
@   r4-r7 per-row group sums d0..d3   r8-r11 activations a0..a7   ip weight word   lr unpacked lanes

        .section .note.GNU-stack, "", %progbits
        .syntax unified
        .arm
        .text
        .align  5
        .global nslm_q8_dot_x4
        .type   nslm_q8_dot_x4, %function

        .equ    ACC0, 0
        .equ    OUTP, 16
        .equ    MP, 20
        .equ    ABASE, 24
        .equ    ASUMP, 28
        .equ    GLEFT, 32
        .equ    RBLEFT, 36
        .equ    FRAME, 48
        .equ    ARG_ASUM, FRAME + 36
        .equ    ARG_NRB, FRAME + 40
        .equ    ARG_NG, FRAME + 44
        .equ    ARG_CPG, FRAME + 48

@ one row of one chunk: two weight words against a0..a7, accumulating into \d
        .macro  ROW d
        ldr     ip, [r0], #4
        and     lr, r2, ip
        smlabb  \d, lr, r8, \d
        smlatb  \d, lr, r9, \d
        and     lr, r2, ip, lsr #8
        smlabt  \d, lr, r8, \d
        smlatt  \d, lr, r9, \d
        ldr     ip, [r0], #4
        and     lr, r2, ip
        smlabb  \d, lr, r10, \d
        smlatb  \d, lr, r11, \d
        and     lr, r2, ip, lsr #8
        smlabt  \d, lr, r10, \d
        smlatt  \d, lr, r11, \d
        .endm

nslm_q8_dot_x4:
        push    {r4-r11, lr}
        sub     sp, sp, #FRAME
        str     r0, [sp, #OUTP]
        str     r2, [sp, #MP]
        str     r3, [sp, #ABASE]
        ldr     ip, [sp, #ARG_NRB]
        str     ip, [sp, #RBLEFT]
        mov     r0, r1
        ldr     r2, =0x00FF00FF

.Lrowblock:
        mov     r4, #0
        mov     r5, #0
        mov     r6, #0
        mov     r7, #0
        str     r4, [sp, #ACC0]
        str     r4, [sp, #ACC0 + 4]
        str     r4, [sp, #ACC0 + 8]
        str     r4, [sp, #ACC0 + 12]
        ldr     r1, [sp, #ABASE]
        ldr     ip, [sp, #ARG_ASUM]
        str     ip, [sp, #ASUMP]
        ldr     ip, [sp, #ARG_NG]
        str     ip, [sp, #GLEFT]

.Lgroup:
        ldr     r3, [sp, #ARG_CPG]
.Lchunk:
        ldmia   r1!, {r8-r11}
        ROW     r4
        ROW     r5
        ROW     r6
        ROW     r7
        subs    r3, r3, #1
        bne     .Lchunk

        @ group end: offset-binary correction, then acc_j += (d_j * m_j) >> 16
        ldr     r8, [sp, #ASUMP]
        ldr     ip, [r8], #4
        str     r8, [sp, #ASUMP]
        sub     r4, r4, ip, lsl #7
        sub     r5, r5, ip, lsl #7
        sub     r6, r6, ip, lsl #7
        sub     r7, r7, ip, lsl #7
        ldr     r8, [sp, #MP]
        ldrh    r9, [r8], #2
        ldrh    r10, [r8], #2
        ldrh    r11, [r8], #2
        ldrh    lr, [r8], #2
        str     r8, [sp, #MP]
        ldr     ip, [sp, #ACC0]
        smlawb  ip, r4, r9, ip
        str     ip, [sp, #ACC0]
        ldr     ip, [sp, #ACC0 + 4]
        smlawb  ip, r5, r10, ip
        str     ip, [sp, #ACC0 + 4]
        ldr     ip, [sp, #ACC0 + 8]
        smlawb  ip, r6, r11, ip
        str     ip, [sp, #ACC0 + 8]
        ldr     ip, [sp, #ACC0 + 12]
        smlawb  ip, r7, lr, ip
        str     ip, [sp, #ACC0 + 12]
        mov     r4, #0
        mov     r5, #0
        mov     r6, #0
        mov     r7, #0
        ldr     ip, [sp, #GLEFT]
        subs    ip, ip, #1
        str     ip, [sp, #GLEFT]
        bne     .Lgroup

        ldr     r8, [sp, #OUTP]
        ldmia   sp, {r9-r11, lr}
        stmia   r8!, {r9-r11, lr}
        str     r8, [sp, #OUTP]
        ldr     ip, [sp, #RBLEFT]
        subs    ip, ip, #1
        str     ip, [sp, #RBLEFT]
        bne     .Lrowblock

        add     sp, sp, #FRAME
        pop     {r4-r11, pc}
        .ltorg
        .size   nslm_q8_dot_x4, . - nslm_q8_dot_x4
