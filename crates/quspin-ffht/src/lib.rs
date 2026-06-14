//! FFI bindings to the FHT (Fast Hadamard Transform) C library, with
//! runtime SIMD dispatch (scalar / SSE2 / AVX2+FMA).
//!
//! The underlying C implementation (`fht_scalar.c`, `fht_sse.c`,
//! `fht_avx.c`, `fast_copy.c`) lives unmodified in `csrc/`. `build.rs`
//! compiles each variant into a separate static library with renamed
//! symbols so all three can be linked into one binary; `fht::fht_f32`
//! and friends pick the fastest variant the running CPU supports via
//! `is_x86_feature_detected!`.

pub mod ffht;

pub use ffht::{fht_f32, fht_f32_oop, fht_f64, fht_f64_oop};
