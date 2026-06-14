//! Integration tests for `quspin_ffht::fht`.
//!
//! Run with: `cargo test -p quspin-ffht`

use quspin_ffht::ffht::{fht_f32, fht_f32_oop, fht_f64, fht_f64_oop};

const EPS_F32: f32 = 1e-5;
const EPS_F64: f64 = 1e-12;

// ---------------------------------------------------------------------
// Correctness: applying FHT twice returns n * original (involution up
// to scale), for both f32 and f64, across several sizes.
// ---------------------------------------------------------------------

#[test]
fn fht_f32_involution() {
    for log_n in 0..=8 {
        let n = 1usize << log_n;
        let original: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();

        let mut buf = original.clone();
        fht_f32(&mut buf);
        fht_f32(&mut buf);

        for (a, b) in buf.iter().zip(original.iter()) {
            let expected = b * n as f32;
            assert!(
                (a - expected).abs() < EPS_F32 * expected.abs().max(1.0),
                "n={n}: got {a}, expected {expected}"
            );
        }
    }
}

#[test]
fn fht_f64_involution() {
    for log_n in 0..=8 {
        let n = 1usize << log_n;
        let original: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();

        let mut buf = original.clone();
        fht_f64(&mut buf);
        fht_f64(&mut buf);

        for (a, b) in buf.iter().zip(original.iter()) {
            let expected = b * n as f64;
            assert!(
                (a - expected).abs() < EPS_F64 * expected.abs().max(1.0),
                "n={n}: got {a}, expected {expected}"
            );
        }
    }
}

// ---------------------------------------------------------------------
// Known value: FHT of [1,2,3,4,5,6,7,8] (n=8) is a fixed reference
// vector. Cross-checks the C implementation against a known result.
// ---------------------------------------------------------------------

#[test]
fn fht_f32_known_value_n8() {
    let mut buf: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    fht_f32(&mut buf);
    let expected = [36.0f32, -4.0, -8.0, 0.0, -16.0, 0.0, 0.0, 0.0];
    for (a, b) in buf.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < EPS_F32,
            "got {buf:?}, expected {expected:?}"
        );
    }
}

#[test]
fn fht_f64_known_value_n4() {
    let mut buf: Vec<f64> = (1..=4).map(|x| x as f64).collect();
    fht_f64(&mut buf);
    let expected = [10.0f64, -2.0, -4.0, 0.0];
    for (a, b) in buf.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < EPS_F64,
            "got {buf:?}, expected {expected:?}"
        );
    }
}

// ---------------------------------------------------------------------
// Out-of-place agrees with in-place, and leaves input untouched.
// ---------------------------------------------------------------------

#[test]
fn fht_f32_oop_matches_in_place() {
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();

    let mut in_place = input.clone();
    fht_f32(&mut in_place);

    let mut input_copy = input.clone();
    let mut out = vec![0f32; input.len()];
    fht_f32_oop(&mut input_copy, &mut out);

    assert_eq!(
        input_copy, input,
        "input must be unchanged by oop transform"
    );
    for (a, b) in out.iter().zip(in_place.iter()) {
        assert!(
            (a - b).abs() < EPS_F32,
            "oop {out:?} != in-place {in_place:?}"
        );
    }
}

#[test]
fn fht_f64_oop_matches_in_place() {
    let input: Vec<f64> = (1..=4).map(|x| x as f64).collect();

    let mut in_place = input.clone();
    fht_f64(&mut in_place);

    let mut input_copy = input.clone();
    let mut out = vec![0f64; input.len()];
    fht_f64_oop(&mut input_copy, &mut out);

    assert_eq!(
        input_copy, input,
        "input must be unchanged by oop transform"
    );
    for (a, b) in out.iter().zip(in_place.iter()) {
        assert!(
            (a - b).abs() < EPS_F64,
            "oop {out:?} != in-place {in_place:?}"
        );
    }
}

// ---------------------------------------------------------------------
// Edge case: n = 1 (log_n = 0) is a no-op.
// ---------------------------------------------------------------------

#[test]
fn fht_f32_n1_is_noop() {
    let mut buf = vec![42.0f32];
    fht_f32(&mut buf);
    assert!((buf[0] - 42.0).abs() < EPS_F32);
}

// ---------------------------------------------------------------------
// Contract violations: non-power-of-two length panics (this is the
// documented Rust-level contract — Python bindings convert this to a
// ValueError before calling in, see quspin-py's ffht test).
// ---------------------------------------------------------------------

#[test]
#[should_panic(expected = "power of two")]
fn fht_f32_non_power_of_two_panics() {
    let mut buf = vec![1.0f32, 2.0, 3.0];
    fht_f32(&mut buf);
}

#[test]
#[should_panic(expected = "power of two")]
fn fht_f64_oop_mismatched_lengths_or_non_pow2_panics() {
    let mut input = vec![1.0f64, 2.0, 3.0]; // not a power of two
    let mut out = vec![0.0f64; 3];
    fht_f64_oop(&mut input, &mut out);
}

// ---------------------------------------------------------------------
// Larger size sanity check (exercises AVX2/SSE/scalar dispatch paths
// equally well at n=1024; the runtime dispatch itself is exercised by
// whichever variant the host CPU supports).
// ---------------------------------------------------------------------

#[test]
fn fht_f32_large_n_involution() {
    let n = 1024;
    let original: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) - 3.0).collect();

    let mut buf = original.clone();
    fht_f32(&mut buf);
    fht_f32(&mut buf);

    for (a, b) in buf.iter().zip(original.iter()) {
        let expected = b * n as f32;
        assert!(
            (a - expected).abs() < 1e-2 * expected.abs().max(1.0),
            "got {a}, expected {expected}"
        );
    }
}
