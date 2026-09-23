//! Tests for `quspin_matrix::expectation`.

use num_complex::Complex;
use quspin_core::basis::{SpaceKind, SpinBasis};
use quspin_matrix::expectation;
use quspin_operator::spin::{SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner};
use smallvec::smallvec;

type C64 = Complex<f64>;

/// Two-site Heisenberg bond `S0·S1` for spin-1/2.
fn heisenberg_bond() -> SpinOperatorInner {
    let one = C64::new(1.0, 0.0);
    let half = C64::new(0.5, 0.0);
    let terms = vec![
        SpinOpEntry::new(0u8, one, smallvec![(SpinOp::Z, 0), (SpinOp::Z, 1)]),
        SpinOpEntry::new(0u8, half, smallvec![(SpinOp::Plus, 0), (SpinOp::Minus, 1)]),
        SpinOpEntry::new(0u8, half, smallvec![(SpinOp::Minus, 0), (SpinOp::Plus, 1)]),
    ];
    SpinOperatorInner::Ham8(SpinOperator::new(terms, 2))
}

#[test]
fn singlet_and_triplet_energies() {
    let basis = SpinBasis::new(2, 2, SpaceKind::Full).unwrap().inner;
    let op = heisenberg_bond();
    let ud = basis.index_of_bytes(&[0, 1]).unwrap();
    let du = basis.index_of_bytes(&[1, 0]).unwrap();
    let uu = basis.index_of_bytes(&[0, 0]).unwrap();
    let r = std::f64::consts::FRAC_1_SQRT_2;

    let mut singlet = vec![C64::default(); 4];
    singlet[ud] = C64::new(r, 0.0);
    singlet[du] = C64::new(-r, 0.0);
    let e = expectation(&op, &basis, &[C64::new(1.0, 0.0)], &singlet).unwrap();
    assert!((e - C64::new(-0.75, 0.0)).norm() < 1e-14, "{e}");

    let mut up = vec![C64::default(); 4];
    up[uu] = C64::new(1.0, 0.0);
    let e = expectation(&op, &basis, &[C64::new(1.0, 0.0)], &up).unwrap();
    assert!((e - C64::new(0.25, 0.0)).norm() < 1e-14, "{e}");
}

#[test]
fn length_mismatch_errors() {
    let basis = SpinBasis::new(2, 2, SpaceKind::Full).unwrap().inner;
    let op = heisenberg_bond();
    let psi = vec![C64::default(); 3];
    assert!(expectation(&op, &basis, &[C64::new(1.0, 0.0)], &psi).is_err());
}
