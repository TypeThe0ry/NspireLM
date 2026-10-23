@ NspireLM int4 x int16 matrix-vector kernel for ARM926EJ-S (ARMv5TE), x4n weight layout.
@
@ void nslm_q4_dot_x4(int32_t *acc, const uint8_t *q, const uint16_t *m, const int16_t *a,
@                     const int32_t *asum, int row_blocks, int groups, int lines_per_group);
@
@ acc[r] = sum_g ((d_rg * m_rg) >> 16), d_rg = sum_{i in g} (u_ri - 8) * a_i, u = 4-bit offset code.
@ One 32-byte cache line = 16 columns x 4 rows: words 0-3 hold columns 0-7 of rows 0-3 (column j in
@ bits 4j..4j+3), words 4-7 columns 8-15. (W >> 4k) & 0x000F000F puts columns k and k+4 into the two
@ 16-bit lanes SMLAxy reads, so 8 MACs cost 1 load, 4 ANDs and 8 single-cycle multiplies.

        .section .note.GNU-stack, "", %progbits
        .syntax unified
        .arm
        .text
        .align  5
        .global nslm_q4_dot_x4
        .type   nslm_q4_dot_x4, %function

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
        .equ    ARG_LPG, FRAME + 48

@ one weight word (8 columns of one row) against a0..a7 in r8-r11, accumulating into \d
        .macro  WORD d
        ldr     ip, [r0], #4
        and     lr, r2, ip
        smlabb  \d, lr, r8, \d
        smlatb  \d, lr, r10, \d
        and     lr, r2, ip, lsr #4
        smlabt  \d, lr, r8, \d
        smlatt  \d, lr, r10, \d
        and     lr, r2, ip, lsr #8
        smlabb  \d, lr, r9, \d
        smlatb  \d, lr, r11, \d
        and     lr, r2, ip, lsr #12
        smlabt  \d, lr, r9, \d
        smlatt  \d, lr, r11, \d
        .endm

nslm_q4_dot_x4:
        push    {r4-r11, lr}
        sub     sp, sp, #FRAME
        str     r0, [sp, #OUTP]
        str     r2, [sp, #MP]
        str     r3, [sp, #ABASE]
        ldr     ip, [sp, #ARG_NRB]
        str     ip, [sp, #RBLEFT]
        mov     r0, r1
        ldr     r2, =0x000F000F

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
        ldr     r3, [sp, #ARG_LPG]
.Lline:
        ldmia   r1!, {r8-r11}           @ columns 0-7 (cache hits, before the weight line)
        WORD    r4
        WORD    r5
        WORD    r6
        WORD    r7
        ldmia   r1!, {r8-r11}           @ columns 8-15
        WORD    r4
        WORD    r5
        WORD    r6
        WORD    r7
        subs    r3, r3, #1
        bne     .Lline

        ldr     r8, [sp, #ASUMP]
        ldr     ip, [r8], #4
        str     r8, [sp, #ASUMP]
        sub     r4, r4, ip, lsl #3
        sub     r5, r5, ip, lsl #3
        sub     r6, r6, ip, lsl #3
        sub     r7, r7, ip, lsl #3
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
        .size   nslm_q4_dot_x4, . - nslm_q4_dot_x4
