@ int32_t nslm_dot_s16_u8(const int16_t *a, const uint8_t *u, int n)   n % 8 == 0
@ Attention inner product: int16 query (or attention weights) against offset-binary KV bytes.
@ 8 elements per iteration: one LDM of activations, one LDM of bytes, 4 ANDs that drop byte pairs into
@ the 16-bit lanes SMLAxy reads, 8 single-cycle multiplies.
        .section .note.GNU-stack, "", %progbits
        .syntax unified
        .arm
        .text
        .align  5
        .global nslm_dot_s16_u8
        .type   nslm_dot_s16_u8, %function
nslm_dot_s16_u8:
        push    {r4-r9, lr}
        mov     r2, r2, lsr #3
        ldr     lr, =0x00FF00FF
        mov     ip, #0
        cmp     r2, #0
        beq     2f
1:      ldmia   r0!, {r4-r7}            @ a0..a7
        ldmia   r1!, {r8, r9}           @ u0..u7
        and     r3, lr, r8              @ (u0, u2)
        smlabb  ip, r3, r4, ip
        smlatb  ip, r3, r5, ip
        and     r3, lr, r8, lsr #8      @ (u1, u3)
        smlabt  ip, r3, r4, ip
        smlatt  ip, r3, r5, ip
        and     r3, lr, r9              @ (u4, u6)
        smlabb  ip, r3, r6, ip
        smlatb  ip, r3, r7, ip
        and     r3, lr, r9, lsr #8      @ (u5, u7)
        smlabt  ip, r3, r6, ip
        smlatt  ip, r3, r7, ip
        subs    r2, r2, #1
        bne     1b
2:      mov     r0, ip
        pop     {r4-r9, pc}
        .ltorg
        .size   nslm_dot_s16_u8, . - nslm_dot_s16_u8


@ void nslm_dots_s16_u8(int32_t *out, const int16_t *a, const uint8_t *u, int n, int rows)
@ out[r] = sum_i a[i] * u[r * n + i]: all attention scores of one head in one call (n % 8 == 0).
@ r0 out  r1 a  r2 u (runs through all rows)  r3 n/8  r4-r7 a lanes  r8 u word  r9 inner count
@ r10 a cursor  r11 unpacked lanes  ip accumulator  lr 0x00FF00FF  [sp] rows left
        .align  5
        .global nslm_dots_s16_u8
        .type   nslm_dots_s16_u8, %function
nslm_dots_s16_u8:
        push    {r4-r11, lr}
        ldr     ip, [sp, #36]           @ rows
        cmp     ip, #0
        beq     4f
        sub     sp, sp, #8
        str     ip, [sp]
        ldr     lr, =0x00FF00FF
        mov     r3, r3, lsr #3
3:      mov     ip, #0
        mov     r10, r1
        mov     r9, r3
1:      ldmia   r10!, {r4-r7}
        ldmia   r2!, {r8}
        and     r11, lr, r8
        smlabb  ip, r11, r4, ip
        smlatb  ip, r11, r5, ip
        and     r11, lr, r8, lsr #8
        smlabt  ip, r11, r4, ip
        smlatt  ip, r11, r5, ip
        ldmia   r2!, {r8}
        and     r11, lr, r8
        smlabb  ip, r11, r6, ip
        smlatb  ip, r11, r7, ip
        and     r11, lr, r8, lsr #8
        smlabt  ip, r11, r6, ip
        smlatt  ip, r11, r7, ip
        subs    r9, r9, #1
        bne     1b
        str     ip, [r0], #4
        ldr     ip, [sp]
        subs    ip, ip, #1
        str     ip, [sp]
        bne     3b
        add     sp, sp, #8
4:      pop     {r4-r11, pc}
        .ltorg
        .size   nslm_dots_s16_u8, . - nslm_dots_s16_u8
