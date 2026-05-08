//! Equivalence tests for `csr_slab_pauli_*` against `QMatrixInner::materialize`.

use num_complex::Complex;
use quspin_core::basis::{SpaceKind, SpinBasis};
use quspin_matrix::csr_slab::csr_slab_pauli_generic;
use quspin_matrix::qmatrix::QMatrixInner;
use quspin_operator::pauli::{HardcoreOp, HardcoreOperator, HardcoreOperatorInner, OpEntry};
use quspin_types::ValueDType;
use smallvec::smallvec;

/// 4-site XX + ZZ chain — small enough for fast tests, has 2 cindices.
fn xx_zz_op() -> HardcoreOperatorInner {
    let mut entries = Vec::new();
    for i in 0..3u32 {
        entries.push(OpEntry::new(
            0u8,
            Complex::new(1.0, 0.0),
            smallvec![(HardcoreOp::X, i), (HardcoreOp::X, i + 1)],
        ));
        entries.push(OpEntry::new(
            1u8,
            Complex::new(1.0, 0.0),
            smallvec![(HardcoreOp::Z, i), (HardcoreOp::Z, i + 1)],
        ));
    }
    HardcoreOperatorInner::Ham8(HardcoreOperator::new(entries))
}

/// 4-site full SpinBasis (lhss=2) — yields a `GenericBasis` we can hand to the slab path.
fn full_4site_basis() -> quspin_basis::dispatch::GenericBasis {
    let basis = SpinBasis::new(4, 2, SpaceKind::Full).unwrap();
    basis.inner
}

#[test]
fn csr_slab_full_range_matches_materialize() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    let dim = 16; // 2^4
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];

    // Reference: build full QMatrix, materialize.
    let qm = QMatrixInner::build_hardcore(&op, &basis, ValueDType::Complex128);
    let (ref_indptr, ref_indices, ref_data) = qm.materialize(&coeffs, true).unwrap();

    // Slab: full range.
    let (slab_indptr, slab_indices, slab_data) =
        csr_slab_pauli_generic(&op, &basis, &coeffs, 0, dim, true).unwrap();

    assert_eq!(slab_indptr, ref_indptr);
    assert_eq!(slab_indices, ref_indices);
    assert_eq!(slab_data.len(), ref_data.len());
    for (a, b) in slab_data.iter().zip(ref_data.iter()) {
        assert!((a - b).norm() < 1e-12, "data mismatch: {a} vs {b}");
    }
}

#[test]
fn csr_slab_partition_concat_matches_materialize() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    let dim = 16;
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];

    let qm = QMatrixInner::build_hardcore(&op, &basis, ValueDType::Complex128);
    let (ref_indptr, ref_indices, ref_data) = qm.materialize(&coeffs, true).unwrap();

    // Try several partitions: 1 chunk (= full), 2, 3, dim chunks (one row each).
    for &k in &[1usize, 2, 3, dim] {
        let bounds: Vec<usize> = (0..=k).map(|i| i * dim / k).collect();
        let mut indptr_concat: Vec<i64> = vec![0];
        let mut indices_concat: Vec<i64> = Vec::new();
        let mut data_concat: Vec<Complex<f64>> = Vec::new();
        for w in bounds.windows(2) {
            let (rs, re) = (w[0], w[1]);
            let (ip, ii, dd) = csr_slab_pauli_generic(&op, &basis, &coeffs, rs, re, true).unwrap();
            // CSR-of-row-blocks merge: shift `ip` by current data length, drop ip[0].
            let off = indices_concat.len() as i64;
            for &p in &ip[1..] {
                indptr_concat.push(p + off);
            }
            indices_concat.extend_from_slice(&ii);
            data_concat.extend_from_slice(&dd);
        }
        assert_eq!(indptr_concat, ref_indptr, "k={k}: indptr");
        assert_eq!(indices_concat, ref_indices, "k={k}: indices");
        assert_eq!(data_concat.len(), ref_data.len(), "k={k}: nnz");
        for (a, b) in data_concat.iter().zip(ref_data.iter()) {
            assert!((a - b).norm() < 1e-12, "k={k}: data mismatch");
        }
    }
}

#[test]
fn csr_slab_empty_range_returns_empty_arrays() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];

    for r in [0usize, 5, 16] {
        let (ip, ii, dd) = csr_slab_pauli_generic(&op, &basis, &coeffs, r, r, true).unwrap();
        assert_eq!(ip, vec![0i64]);
        assert!(ii.is_empty());
        assert!(dd.is_empty());
    }
}

#[test]
fn csr_slab_drop_zeros_false_keeps_zeros() {
    // 4-site H = XX + (-1)*XX on 2 cindices (with both coeffs = 1.0) → all-zero matrix.
    let entries = vec![
        OpEntry::new(
            0u8,
            Complex::new(1.0, 0.0),
            smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)],
        ),
        OpEntry::new(
            1u8,
            Complex::new(-1.0, 0.0),
            smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)],
        ),
    ];
    let op = HardcoreOperatorInner::Ham8(HardcoreOperator::new(entries));
    let basis = full_4site_basis();
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];

    // drop_zeros=true: matrix collapses to nothing.
    let (_, ii_drop, dd_drop) = csr_slab_pauli_generic(&op, &basis, &coeffs, 0, 16, true).unwrap();
    assert!(ii_drop.is_empty());
    assert!(dd_drop.is_empty());

    // drop_zeros=false: every row that XX touches still emits the (cancelled) entry.
    let (_, ii_keep, _) = csr_slab_pauli_generic(&op, &basis, &coeffs, 0, 16, false).unwrap();
    assert!(
        !ii_keep.is_empty(),
        "drop_zeros=false should preserve cancelled entries"
    );
}

#[test]
fn csr_slab_invalid_range_errors() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];

    // row_start > row_end
    assert!(csr_slab_pauli_generic(&op, &basis, &coeffs, 5, 3, true).is_err());
    // row_end > basis.size()
    assert!(csr_slab_pauli_generic(&op, &basis, &coeffs, 0, 99, true).is_err());
}

#[test]
fn csr_slab_wrong_coeffs_len_errors() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    // op has 2 cindices; pass 3 coeffs.
    let coeffs = vec![Complex::new(1.0, 0.0); 3];
    assert!(csr_slab_pauli_generic(&op, &basis, &coeffs, 0, 16, true).is_err());
}

/// Zero-coefficient masking: passing `coeffs = [1.0, 0.0]` against the
/// XX + ZZ operator must yield exactly the same matrix as a single-cindex
/// XX-only operator with `coeffs = [1.0]`.  Exercises the `coeff == 0`
/// fast-skip path inside `process_row`.
#[test]
fn csr_slab_zero_coeff_filter_matches_smaller_op() {
    fn xx_only_op() -> HardcoreOperatorInner {
        let mut entries = Vec::new();
        for i in 0..3u32 {
            entries.push(OpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(HardcoreOp::X, i), (HardcoreOp::X, i + 1)],
            ));
        }
        HardcoreOperatorInner::Ham8(HardcoreOperator::new(entries))
    }

    let basis = full_4site_basis();

    // Two-cindex op masked to keep only XX.
    let masked_op = xx_zz_op();
    let masked_coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    let (mask_ip, mask_ii, mask_dd) =
        csr_slab_pauli_generic(&masked_op, &basis, &masked_coeffs, 0, 16, true).unwrap();

    // Single-cindex XX-only op as the reference.
    let ref_op = xx_only_op();
    let ref_coeffs = vec![Complex::new(1.0, 0.0)];
    let (ref_ip, ref_ii, ref_dd) =
        csr_slab_pauli_generic(&ref_op, &basis, &ref_coeffs, 0, 16, true).unwrap();

    assert_eq!(mask_ip, ref_ip);
    assert_eq!(mask_ii, ref_ii);
    assert_eq!(mask_dd.len(), ref_dd.len());
    for (a, b) in mask_dd.iter().zip(ref_dd.iter()) {
        assert!((a - b).norm() < 1e-12);
    }
}
