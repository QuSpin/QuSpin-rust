//! [`OperatorOnBasis`]: matrix-free [`LinearOperator`] over an
//! `(operator, basis)` pair.
//!
//! The alternative — `QMatrix::build_*` followed by
//! [`as_linearoperator`](crate::QMatrix) — materialises every non-zero matrix
//! element up front. For large bases that is the dominant memory cost and
//! often simply not affordable. `OperatorOnBasis` keeps nothing but the
//! operator, the basis, and a coefficient snapshot, and recomputes matrix
//! elements on the fly inside each product.
//!
//! # What is and isn't supported
//!
//! | Method | Status |
//! |--------|--------|
//! | `dim`, `dot`, `dot_many` | full — via [`OperatorDispatch::apply`] |
//! | `dot_transpose` | full — via [`matrix_free::dot_transpose`](crate::matrix_free) |
//! | `trace`, `onenorm` | full — one basis sweep each, no matrix |
//! | `dot_chunk`, `dot_transpose_chunk` | **unsupported** — see below |
//!
//! A row range of `A · x` cannot be produced matrix-free without `A†`: row
//! `r` is `⟨r|A|·⟩`, and the only cheap direction here is applying the
//! operator to a *ket*. Computing one chunk would therefore cost a full
//! sweep, making the chunked path quadratic overall. Both chunk methods
//! return an error and [`parallel_hint`] returns `false`, which is what
//! steers `ExpmOp` onto the sequential path that only ever calls `dot`.
//!
//! # Orientation, and how it differs from `QMatrixOperator`
//!
//! Everything here represents `A` as [`OperatorDispatch::apply`] defines it:
//! `dot` is `A · x`, `dot_transpose` is `Aᵀ · x`, and `onenorm` is the column
//! 1-norm of `A`.
//!
//! [`QMatrixOperator`](crate::QMatrixOperator) currently represents `Aᵀ`
//! instead — `QMatrix::build_*` stores the transpose (issue #121). For the
//! Hermitian real-symmetric Hamiltonians in most use the two agree, but for a
//! non-symmetric operator `OperatorOnBasis::dot` matches
//! `QMatrixOperator::dot_transpose`, not its `dot`. When #121 is fixed the two
//! line up and this note can go.
//!
//! [`parallel_hint`]: LinearOperator::parallel_hint

use std::ops::Range;
use std::sync::Arc;

use ndarray::{ArrayView2, ArrayViewMut2};
use num_complex::Complex;
use quspin_basis::dispatch::{BitBasis, GenericBasis};
use quspin_basis::{BosonBasis, FermionBasis, SpinBasis};
use quspin_types::{ExpmComputation, LinearOperator, QuSpinError};

use crate::dispatch::OperatorDispatch;

type C64 = Complex<f64>;

// ---------------------------------------------------------------------------
// BasisSource
// ---------------------------------------------------------------------------

/// A basis an operator can be applied against.
///
/// Exists so [`OperatorOnBasis`] can accept any of the concrete basis
/// newtypes without knowing which dispatch root each wraps. Exactly one of
/// [`as_generic`](BasisSource::as_generic) / [`as_bit`](BasisSource::as_bit)
/// returns `Some`.
pub trait BasisSource: Send + Sync {
    /// The `GenericBasis` dispatch root, for bases that have one.
    fn as_generic(&self) -> Option<&GenericBasis>;

    /// The `BitBasis` dispatch root, for bases that skip `GenericBasis`.
    fn as_bit(&self) -> Option<&BitBasis>;

    /// Number of basis states.
    fn size(&self) -> usize;
}

impl BasisSource for GenericBasis {
    fn as_generic(&self) -> Option<&GenericBasis> {
        Some(self)
    }
    fn as_bit(&self) -> Option<&BitBasis> {
        None
    }
    fn size(&self) -> usize {
        GenericBasis::size(self)
    }
}

impl BasisSource for SpinBasis {
    fn as_generic(&self) -> Option<&GenericBasis> {
        Some(&self.inner)
    }
    fn as_bit(&self) -> Option<&BitBasis> {
        None
    }
    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl BasisSource for BosonBasis {
    fn as_generic(&self) -> Option<&GenericBasis> {
        Some(&self.inner)
    }
    fn as_bit(&self) -> Option<&BitBasis> {
        None
    }
    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl BasisSource for FermionBasis {
    fn as_generic(&self) -> Option<&GenericBasis> {
        None
    }
    fn as_bit(&self) -> Option<&BitBasis> {
        Some(&self.inner)
    }
    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<T: BasisSource + ?Sized> BasisSource for Arc<T> {
    fn as_generic(&self) -> Option<&GenericBasis> {
        (**self).as_generic()
    }
    fn as_bit(&self) -> Option<&BitBasis> {
        (**self).as_bit()
    }
    fn size(&self) -> usize {
        (**self).size()
    }
}

impl<T: BasisSource + ?Sized> BasisSource for &T {
    fn as_generic(&self) -> Option<&GenericBasis> {
        (**self).as_generic()
    }
    fn as_bit(&self) -> Option<&BitBasis> {
        (**self).as_bit()
    }
    fn size(&self) -> usize {
        (**self).size()
    }
}

// ---------------------------------------------------------------------------
// OperatorOnBasis
// ---------------------------------------------------------------------------

/// Matrix-free [`LinearOperator`] bundling an operator, a basis, and a
/// coefficient snapshot.
///
/// `OP` and `B` are generic so the whole chain stays statically dispatched;
/// pass owned values, `&T`, or `Arc<T>` for either. Coefficients are a
/// snapshot taken at construction, matching
/// [`QMatrixOperator`](crate::QMatrixOperator) — a time-dependent analogue
/// would rebuild the wrapper per time step.
pub struct OperatorOnBasis<OP, B> {
    op: OP,
    basis: B,
    coeffs: Vec<C64>,
}

impl<OP, B> OperatorOnBasis<OP, B>
where
    OP: OperatorDispatch,
    B: BasisSource,
{
    /// Bundle `op` and `basis` with a coefficient snapshot.
    ///
    /// # Errors
    /// Returns `ValueError` when `coeffs.len() != op.num_cindices()`.
    pub fn new(op: OP, basis: B, coeffs: Vec<C64>) -> Result<Self, QuSpinError> {
        let expected = op.num_cindices();
        if coeffs.len() != expected {
            return Err(QuSpinError::ValueError(format!(
                "coeffs.len()={} must equal operator num_cindices={expected}",
                coeffs.len(),
            )));
        }
        Ok(Self { op, basis, coeffs })
    }

    /// Borrow the wrapped operator.
    pub fn operator(&self) -> &OP {
        &self.op
    }

    /// Borrow the wrapped basis.
    pub fn basis(&self) -> &B {
        &self.basis
    }

    /// Borrow the coefficient snapshot.
    pub fn coeffs(&self) -> &[C64] {
        &self.coeffs
    }

    /// Which dispatch root the wrapped basis exposes.
    #[inline]
    fn root(&self) -> Result<Root<'_>, QuSpinError> {
        if let Some(space) = self.basis.as_generic() {
            Ok(Root::Generic(space))
        } else if let Some(space) = self.basis.as_bit() {
            Ok(Root::Bit(space))
        } else {
            Err(QuSpinError::RuntimeError(
                "basis exposes neither a GenericBasis nor a BitBasis dispatch root".into(),
            ))
        }
    }
}

/// The dispatch root a [`BasisSource`] resolved to.
enum Root<'a> {
    Generic(&'a GenericBasis),
    Bit(&'a BitBasis),
}

fn chunked_unsupported<T>(method: &str) -> Result<T, QuSpinError> {
    Err(QuSpinError::RuntimeError(format!(
        "OperatorOnBasis::{method} is not supported: a row range of A·x cannot be \
         computed matrix-free without the adjoint operator. parallel_hint() returns \
         false so the sequential path (which only calls dot) is used instead."
    )))
}

impl<OP, B> LinearOperator<C64> for OperatorOnBasis<OP, B>
where
    OP: OperatorDispatch + Send + Sync,
    B: BasisSource,
{
    fn dim(&self) -> usize {
        self.basis.size()
    }

    fn trace(&self) -> C64 {
        // `LinearOperator::trace` is infallible; the only failure mode here is
        // a coeffs/cindex mismatch, which `new` already rejected.
        match self.root() {
            Ok(Root::Generic(space)) => self.op.trace(space, &self.coeffs),
            Ok(Root::Bit(space)) => self.op.trace_bit(space, &self.coeffs),
            Err(e) => Err(e),
        }
        .unwrap_or_default()
    }

    fn onenorm(&self, shift: C64) -> f64 {
        match self.root() {
            Ok(Root::Generic(space)) => self.op.onenorm(space, &self.coeffs, shift),
            Ok(Root::Bit(space)) => self.op.onenorm_bit(space, &self.coeffs, shift),
            Err(e) => Err(e),
        }
        .unwrap_or(f64::INFINITY)
    }

    /// `apply` already parallelises internally with rayon, and the chunked
    /// methods are unsupported, so the persistent-thread path is off.
    fn parallel_hint(&self) -> bool {
        false
    }

    fn dot(&self, overwrite: bool, input: &[C64], output: &mut [C64]) -> Result<(), QuSpinError> {
        match self.root()? {
            Root::Generic(space) => self.op.apply(space, &self.coeffs, input, output, overwrite),
            Root::Bit(space) => self
                .op
                .apply_bit(space, &self.coeffs, input, output, overwrite),
        }
    }

    fn dot_transpose(
        &self,
        overwrite: bool,
        input: &[C64],
        output: &mut [C64],
    ) -> Result<(), QuSpinError> {
        match self.root()? {
            Root::Generic(space) => {
                self.op
                    .dot_transpose(space, &self.coeffs, input, output, overwrite)
            }
            Root::Bit(space) => {
                self.op
                    .dot_transpose_bit(space, &self.coeffs, input, output, overwrite)
            }
        }
    }

    fn dot_many(
        &self,
        overwrite: bool,
        input: ArrayView2<'_, C64>,
        mut output: ArrayViewMut2<'_, C64>,
    ) -> Result<(), QuSpinError> {
        let n = self.dim();
        if input.nrows() != n || output.nrows() != n {
            return Err(QuSpinError::ValueError(format!(
                "input/output must have {n} rows, got {} and {}",
                input.nrows(),
                output.nrows(),
            )));
        }
        if input.ncols() != output.ncols() {
            return Err(QuSpinError::ValueError(format!(
                "input has {} columns but output has {}",
                input.ncols(),
                output.ncols(),
            )));
        }

        // One `dot` per column. Column-major contiguity is not guaranteed by
        // `ArrayView2`, so each column is copied through a scratch buffer.
        let mut in_col = vec![C64::default(); n];
        let mut out_col = vec![C64::default(); n];
        for k in 0..input.ncols() {
            for (dst, src) in in_col.iter_mut().zip(input.column(k).iter()) {
                *dst = *src;
            }
            self.dot(true, &in_col, &mut out_col)?;
            let mut target = output.column_mut(k);
            for (dst, src) in target.iter_mut().zip(out_col.iter()) {
                if overwrite {
                    *dst = *src;
                } else {
                    *dst += *src;
                }
            }
        }
        Ok(())
    }

    fn dot_chunk(
        &self,
        _overwrite: bool,
        _input: &[C64],
        _output_chunk: &mut [C64],
        _row_start: usize,
    ) -> Result<(), QuSpinError> {
        chunked_unsupported("dot_chunk")
    }

    fn dot_transpose_chunk(
        &self,
        _input: &[C64],
        _output: &[<C64 as ExpmComputation>::Atomic],
        _rows: Range<usize>,
    ) -> Result<(), QuSpinError> {
        chunked_unsupported("dot_transpose_chunk")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use quspin_basis::SpaceKind;
    use quspin_operator::pauli::{HardcoreOp, HardcoreOperator, HardcoreOperatorInner, OpEntry};
    use quspin_operator::spin::{SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner};
    use quspin_types::AtomicAccum;
    use smallvec::smallvec;

    const TOL: f64 = 1e-12;

    fn c(re: f64) -> C64 {
        C64::new(re, 0.0)
    }

    /// `S+_0 S-_1` on two sites — deliberately non-Hermitian, so a stray
    /// transpose anywhere in the chain shows up.
    fn ladder_op(lhss: usize) -> SpinOperatorInner {
        let terms = vec![SpinOpEntry::new(
            0u8,
            c(1.0),
            smallvec![(SpinOp::Plus, 0), (SpinOp::Minus, 1)],
        )];
        SpinOperatorInner::Ham8(SpinOperator::new(terms, lhss))
    }

    /// `Sz_0 Sz_1 + (S+_0 S-_1 + S-_0 S+_1)/2` — Hermitian, two cindices.
    fn heisenberg_op(lhss: usize) -> SpinOperatorInner {
        let terms = vec![
            SpinOpEntry::new(0u8, c(1.0), smallvec![(SpinOp::Z, 0), (SpinOp::Z, 1)]),
            SpinOpEntry::new(
                1u8,
                c(0.5),
                smallvec![(SpinOp::Plus, 0), (SpinOp::Minus, 1)],
            ),
            SpinOpEntry::new(
                1u8,
                c(0.5),
                smallvec![(SpinOp::Minus, 0), (SpinOp::Plus, 1)],
            ),
        ];
        SpinOperatorInner::Ham8(SpinOperator::new(terms, lhss))
    }

    /// `X_0 X_1` for the fermion/bit path.
    fn xx_op() -> HardcoreOperatorInner {
        let terms = vec![OpEntry::new(
            0u8,
            c(1.0),
            smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)],
        )];
        HardcoreOperatorInner::Ham8(HardcoreOperator::new(terms))
    }

    /// Dense matrix of `op` obtained by applying it to each unit vector.
    ///
    /// Built from `dot` alone — the pre-existing, independently tested
    /// `apply` path — so it is a genuine reference for the new kernels.
    fn dense_from_dot(op: &impl LinearOperator<C64>) -> Vec<Vec<C64>> {
        let n = op.dim();
        let mut cols = Vec::with_capacity(n);
        for j in 0..n {
            let mut e = vec![C64::default(); n];
            e[j] = c(1.0);
            let mut out = vec![C64::default(); n];
            op.dot(true, &e, &mut out).unwrap();
            cols.push(out);
        }
        // cols[j][i] == A[i, j]
        cols
    }

    fn spin_basis(n_sites: usize, lhss: usize) -> SpinBasis {
        SpinBasis::new(n_sites, lhss, SpaceKind::Full).unwrap()
    }

    // --- construction ---

    #[test]
    fn rejects_wrong_coeff_count() {
        let basis = spin_basis(2, 3);
        let err = OperatorOnBasis::new(ladder_op(3), basis, vec![c(1.0), c(1.0)]);
        assert!(err.is_err());
    }

    #[test]
    fn dim_matches_basis_size() {
        let basis = spin_basis(2, 3);
        let expected = basis.inner.size();
        let op = OperatorOnBasis::new(ladder_op(3), basis, vec![c(1.0)]).unwrap();
        assert_eq!(op.dim(), expected);
    }

    #[test]
    fn parallel_hint_is_off() {
        let op = OperatorOnBasis::new(ladder_op(3), spin_basis(2, 3), vec![c(1.0)]).unwrap();
        assert!(!op.parallel_hint());
    }

    // --- dot_transpose ---

    #[test]
    fn dot_transpose_matches_dense_transpose() {
        let op = OperatorOnBasis::new(ladder_op(3), spin_basis(2, 3), vec![c(1.0)]).unwrap();
        let a = dense_from_dot(&op);
        let n = op.dim();

        for j in 0..n {
            let mut e = vec![C64::default(); n];
            e[j] = c(1.0);
            let mut got = vec![C64::default(); n];
            op.dot_transpose(true, &e, &mut got).unwrap();
            // (Aᵀ)[:, j] == A[j, :]
            for i in 0..n {
                assert!((got[i] - a[i][j]).norm() < TOL, "i={i} j={j}");
            }
        }
    }

    #[test]
    fn dot_transpose_differs_from_dot_for_non_hermitian() {
        let op = OperatorOnBasis::new(ladder_op(3), spin_basis(2, 3), vec![c(1.0)]).unwrap();
        let n = op.dim();
        let x: Vec<C64> = (0..n).map(|k| c(k as f64 + 1.0)).collect();

        let mut fwd = vec![C64::default(); n];
        let mut rev = vec![C64::default(); n];
        op.dot(true, &x, &mut fwd).unwrap();
        op.dot_transpose(true, &x, &mut rev).unwrap();

        // Guards against the kernels accidentally being the same thing.
        assert!(fwd.iter().zip(&rev).any(|(a, b)| (a - b).norm() > TOL));
    }

    #[test]
    fn dot_transpose_accumulates_when_not_overwriting() {
        let op =
            OperatorOnBasis::new(heisenberg_op(3), spin_basis(2, 3), vec![c(1.0), c(1.0)]).unwrap();
        let n = op.dim();
        let x = vec![c(1.0); n];

        let mut once = vec![C64::default(); n];
        op.dot_transpose(true, &x, &mut once).unwrap();

        let mut twice = vec![C64::default(); n];
        op.dot_transpose(true, &x, &mut twice).unwrap();
        op.dot_transpose(false, &x, &mut twice).unwrap();

        for (a, b) in twice.iter().zip(&once) {
            assert!((a - b * c(2.0)).norm() < TOL);
        }
    }

    // --- trace ---

    #[test]
    fn trace_matches_dense_diagonal() {
        for lhss in [2, 3, 4] {
            let op =
                OperatorOnBasis::new(heisenberg_op(lhss), spin_basis(2, lhss), vec![c(1.0); 2])
                    .unwrap();
            let a = dense_from_dot(&op);
            let want: C64 = (0..op.dim()).map(|i| a[i][i]).sum();
            assert!((op.trace() - want).norm() < TOL, "lhss={lhss}");
        }
    }

    #[test]
    fn trace_of_purely_off_diagonal_operator_is_zero() {
        let op = OperatorOnBasis::new(ladder_op(3), spin_basis(2, 3), vec![c(1.0)]).unwrap();
        assert!(op.trace().norm() < TOL);
    }

    // --- onenorm ---

    /// `max_j Σ_i |A[i, j] − shift·δ_ij|` straight off the dense reference.
    fn dense_onenorm(a: &[Vec<C64>], shift: C64) -> f64 {
        let n = a.len();
        (0..n)
            .map(|j| {
                (0..n)
                    .map(|i| {
                        let v = if i == j { a[i][j] - shift } else { a[i][j] };
                        v.norm()
                    })
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn onenorm_matches_dense_column_norm() {
        for lhss in [2, 3, 4] {
            let op =
                OperatorOnBasis::new(heisenberg_op(lhss), spin_basis(2, lhss), vec![c(1.0); 2])
                    .unwrap();
            let a = dense_from_dot(&op);
            for shift in [c(0.0), c(1.5), C64::new(0.25, -0.75)] {
                let want = dense_onenorm(&a, shift);
                let got = op.onenorm(shift);
                assert!((got - want).abs() < 1e-10, "lhss={lhss} shift={shift}");
            }
        }
    }

    #[test]
    fn onenorm_of_non_hermitian_matches_dense() {
        let op = OperatorOnBasis::new(ladder_op(3), spin_basis(2, 3), vec![c(1.0)]).unwrap();
        let a = dense_from_dot(&op);
        for shift in [c(0.0), c(2.0)] {
            assert!((op.onenorm(shift) - dense_onenorm(&a, shift)).abs() < 1e-10);
        }
    }

    #[test]
    fn onenorm_counts_shift_on_empty_columns() {
        // A column with no stored entries still contributes |shift| once the
        // identity is subtracted.
        let op = OperatorOnBasis::new(ladder_op(3), spin_basis(2, 3), vec![c(1.0)]).unwrap();
        let a = dense_from_dot(&op);
        let shift = c(3.0);
        assert!((op.onenorm(shift) - dense_onenorm(&a, shift)).abs() < 1e-10);
    }

    // --- dot_many ---

    #[test]
    fn dot_many_matches_column_wise_dot() {
        let op = OperatorOnBasis::new(heisenberg_op(3), spin_basis(2, 3), vec![c(1.0); 2]).unwrap();
        let n = op.dim();
        let k = 3;

        let mut input = Array2::<C64>::zeros((n, k));
        for ((i, j), v) in input.indexed_iter_mut() {
            *v = C64::new(i as f64 - j as f64, 0.5 * j as f64);
        }
        let mut output = Array2::<C64>::zeros((n, k));
        op.dot_many(true, input.view(), output.view_mut()).unwrap();

        for j in 0..k {
            let col: Vec<C64> = input.column(j).iter().copied().collect();
            let mut want = vec![C64::default(); n];
            op.dot(true, &col, &mut want).unwrap();
            for i in 0..n {
                assert!((output[[i, j]] - want[i]).norm() < TOL, "i={i} j={j}");
            }
        }
    }

    #[test]
    fn dot_many_rejects_shape_mismatch() {
        let op = OperatorOnBasis::new(ladder_op(3), spin_basis(2, 3), vec![c(1.0)]).unwrap();
        let input = Array2::<C64>::zeros((op.dim() + 1, 2));
        let mut output = Array2::<C64>::zeros((op.dim() + 1, 2));
        assert!(op.dot_many(true, input.view(), output.view_mut()).is_err());
    }

    // --- unsupported chunked methods ---

    #[test]
    fn chunked_methods_report_unsupported() {
        let op = OperatorOnBasis::new(ladder_op(3), spin_basis(2, 3), vec![c(1.0)]).unwrap();
        let n = op.dim();
        let input = vec![c(1.0); n];
        let mut chunk = vec![C64::default(); 2];
        assert!(op.dot_chunk(true, &input, &mut chunk, 0).is_err());

        let atomics: Vec<<C64 as ExpmComputation>::Atomic> =
            (0..n).map(|_| AtomicAccum::zero()).collect();
        assert!(op.dot_transpose_chunk(&input, &atomics, 0..n).is_err());
    }

    // --- bit-family (FermionBasis) path ---

    #[test]
    fn bit_basis_path_works() {
        let basis = FermionBasis::new(2, SpaceKind::Full).unwrap();
        let op = OperatorOnBasis::new(xx_op(), basis, vec![c(1.0)]).unwrap();
        assert_eq!(op.dim(), 4);

        let a = dense_from_dot(&op);
        let want: C64 = (0..op.dim()).map(|i| a[i][i]).sum();
        assert!((op.trace() - want).norm() < TOL);
        assert!((op.onenorm(c(0.0)) - dense_onenorm(&a, c(0.0))).abs() < 1e-10);

        let n = op.dim();
        for j in 0..n {
            let mut e = vec![C64::default(); n];
            e[j] = c(1.0);
            let mut got = vec![C64::default(); n];
            op.dot_transpose(true, &e, &mut got).unwrap();
            for i in 0..n {
                assert!((got[i] - a[i][j]).norm() < TOL);
            }
        }
    }

    // --- end-to-end: drives expm with no QMatrix in sight ---

    #[test]
    fn expm_runs_matrix_free_and_matches_explicit_series() {
        use ndarray::Array1;
        use quspin_expm::ExpmOp;

        let basis = spin_basis(2, 3);
        let op = OperatorOnBasis::new(heisenberg_op(3), basis, vec![c(1.0), c(1.0)]).unwrap();
        let n = op.dim();
        let a = C64::new(0.0, -0.35);

        // Reference: dense Taylor series of exp(a·A)·v, summed to convergence.
        let dense = dense_from_dot(&op);
        let v: Vec<C64> = (0..n).map(|k| c((k as f64 + 1.0) / n as f64)).collect();
        let mut term = v.clone();
        let mut want = v.clone();
        for k in 1..200 {
            let mut next = vec![C64::default(); n];
            for i in 0..n {
                for j in 0..n {
                    next[i] += dense[j][i] * term[j];
                }
            }
            let scale = a / c(k as f64);
            for (t, nx) in term.iter_mut().zip(&next) {
                *t = nx * scale;
            }
            for (w, t) in want.iter_mut().zip(&term) {
                *w += t;
            }
            if term.iter().all(|t| t.norm() < 1e-18) {
                break;
            }
        }

        let expm = ExpmOp::new(op, a).unwrap();
        let mut got = Array1::from(v);
        expm.apply(got.view_mut()).unwrap();

        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).norm() < 1e-10, "got {g} want {w}");
        }
    }

    // --- shared ownership ---

    #[test]
    fn accepts_arc_operands() {
        let basis = Arc::new(spin_basis(2, 3));
        let op = OperatorOnBasis::new(ladder_op(3), Arc::clone(&basis), vec![c(1.0)]).unwrap();
        assert_eq!(op.dim(), basis.inner.size());
    }
}
