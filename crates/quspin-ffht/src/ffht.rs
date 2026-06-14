use std::os::raw::c_int;

mod ffi {
    use std::os::raw::c_int;

    unsafe extern "C" {
        pub fn fht_float_scalar(buf: *mut f32, log_n: c_int) -> c_int;
        pub fn fht_double_scalar(buf: *mut f64, log_n: c_int) -> c_int;
        pub fn fht_float_oop_scalar(input: *mut f32, out: *mut f32, log_n: c_int) -> c_int;
        pub fn fht_double_oop_scalar(input: *mut f64, out: *mut f64, log_n: c_int) -> c_int;
    }

    // Only present when build.rs actually compiled the SSE/AVX2 variants
    // (x86/x86_64 with a non-MSVC compiler — see build.rs).
    #[cfg(fht_have_simd_variants)]
    unsafe extern "C" {
        pub fn fht_float_sse(buf: *mut f32, log_n: c_int) -> c_int;
        pub fn fht_double_sse(buf: *mut f64, log_n: c_int) -> c_int;
        pub fn fht_float_oop_sse(input: *mut f32, out: *mut f32, log_n: c_int) -> c_int;
        pub fn fht_double_oop_sse(input: *mut f64, out: *mut f64, log_n: c_int) -> c_int;

        pub fn fht_float_avx2(buf: *mut f32, log_n: c_int) -> c_int;
        pub fn fht_double_avx2(buf: *mut f64, log_n: c_int) -> c_int;
        pub fn fht_float_oop_avx2(input: *mut f32, out: *mut f32, log_n: c_int) -> c_int;
        pub fn fht_double_oop_avx2(input: *mut f64, out: *mut f64, log_n: c_int) -> c_int;
    }
}

#[inline]
fn log_n_of(len: usize) -> c_int {
    let log_n = len.trailing_zeros();
    assert_eq!(1usize << log_n, len, "buffer length must be a power of two");
    log_n as c_int
}

/// In-place Fast Hadamard Transform for f32. `buf.len()` must be a power of two.
pub fn fht_f32(buf: &mut [f32]) {
    let log_n = log_n_of(buf.len());
    unsafe {
        #[cfg(fht_have_simd_variants)]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                ffi::fht_float_avx2(buf.as_mut_ptr(), log_n);
                return;
            }
            if is_x86_feature_detected!("sse2") {
                ffi::fht_float_sse(buf.as_mut_ptr(), log_n);
                return;
            }
        }
        ffi::fht_float_scalar(buf.as_mut_ptr(), log_n);
    }
}

/// In-place Fast Hadamard Transform for f64. `buf.len()` must be a power of two.
pub fn fht_f64(buf: &mut [f64]) {
    let log_n = log_n_of(buf.len());
    unsafe {
        #[cfg(fht_have_simd_variants)]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                ffi::fht_double_avx2(buf.as_mut_ptr(), log_n);
                return;
            }
            if is_x86_feature_detected!("sse2") {
                ffi::fht_double_sse(buf.as_mut_ptr(), log_n);
                return;
            }
        }
        ffi::fht_double_scalar(buf.as_mut_ptr(), log_n);
    }
}

/// Out-of-place Fast Hadamard Transform for f32. Both slices must be the same
/// power-of-two length.
pub fn fht_f32_oop(input: &mut [f32], out: &mut [f32]) {
    assert_eq!(input.len(), out.len());
    let log_n = log_n_of(input.len());
    unsafe {
        #[cfg(fht_have_simd_variants)]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                ffi::fht_float_oop_avx2(input.as_mut_ptr(), out.as_mut_ptr(), log_n);
                return;
            }
            if is_x86_feature_detected!("sse2") {
                ffi::fht_float_oop_sse(input.as_mut_ptr(), out.as_mut_ptr(), log_n);
                return;
            }
        }
        ffi::fht_float_oop_scalar(input.as_mut_ptr(), out.as_mut_ptr(), log_n);
    }
}

/// Out-of-place Fast Hadamard Transform for f64. Both slices must be the same
/// power-of-two length.
pub fn fht_f64_oop(input: &mut [f64], out: &mut [f64]) {
    assert_eq!(input.len(), out.len());
    let log_n = log_n_of(input.len());
    unsafe {
        #[cfg(fht_have_simd_variants)]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                ffi::fht_double_oop_avx2(input.as_mut_ptr(), out.as_mut_ptr(), log_n);
                return;
            }
            if is_x86_feature_detected!("sse2") {
                ffi::fht_double_oop_sse(input.as_mut_ptr(), out.as_mut_ptr(), log_n);
                return;
            }
        }
        ffi::fht_double_oop_scalar(input.as_mut_ptr(), out.as_mut_ptr(), log_n);
    }
}
