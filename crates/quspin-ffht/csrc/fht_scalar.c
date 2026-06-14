/*
 * fht_scalar.c
 *
 * Pure C scalar implementation of the Fast Hadamard Transform.
 * No SIMD intrinsics or inline assembly — works on any architecture
 * including ARM64 (Apple Silicon), WASM, RISC-V, etc.
 *
 * When compiled with -O3 -ffast-math, the compiler's auto-vectorizer
 * will emit NEON instructions on ARM64, giving near-optimal performance
 * without hand-written intrinsics.
 *
 * Included by fht_impl.h when neither AVX nor SSE2 is available.
 */

#include "fht.h"

#ifdef FHT_HEADER_ONLY
#  define _STORAGE_ static inline
#else
#  define _STORAGE_
#endif

/* ── float ── */

_STORAGE_ int fht_float(float *buf, int log_n) {
    int n = 1 << log_n;

    /* Iterative Cooley-Tukey butterfly.
     * Outer loop halves the stride each pass; inner loops do the butterfly.
     * The compiler can auto-vectorize the innermost loop with NEON/SSE. */
    for (int stride = n >> 1; stride >= 1; stride >>= 1) {
        for (int base = 0; base < n; base += stride << 1) {
            for (int i = 0; i < stride; ++i) {
                float u = buf[base + i];
                float v = buf[base + i + stride];
                buf[base + i]          = u + v;
                buf[base + i + stride] = u - v;
            }
        }
    }
    return 0;
}

/* ── double ── */

_STORAGE_ int fht_double(double *buf, int log_n) {
    int n = 1 << log_n;

    for (int stride = n >> 1; stride >= 1; stride >>= 1) {
        for (int base = 0; base < n; base += stride << 1) {
            for (int i = 0; i < stride; ++i) {
                double u = buf[base + i];
                double v = buf[base + i + stride];
                buf[base + i]          = u + v;
                buf[base + i + stride] = u - v;
            }
        }
    }
    return 0;
}

#ifdef FHT_HEADER_ONLY
#  undef _STORAGE_
#endif