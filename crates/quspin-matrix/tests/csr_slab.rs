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
