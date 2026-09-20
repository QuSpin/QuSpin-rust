//! Matrix-free spectral metadata and transposed application.
//!
//! [`apply_and_project_to_inner`] already gives `y = A · x` without
//! assembling a [`QMatrix`](crate::QMatrix).  The kernels here supply the
//! three remaining pieces a [`LinearOperator`] needs — `trace`, `onenorm`
//! and `dot_transpose` — so an `(operator, basis)` pair can satisfy the
//! trait with no intermediate matrix.
//!
//! # The matrix these kernels describe
//!
//! For a basis `S` and operator `op`, `apply_and_project_to_inner` defines
//!
//! ```text
//! A[j, i] = Σ_{(s, w) ∈ expand(i)} Σ_{op terms} coeffs[c] · op_amp · w · scale(j)
//! ```
//!
//! where `expand(i)` is [`ExpandRefState::expand_ref_state_iter`] and
//! `scale(j)` comes from [`ProjectState::project`].  Every kernel below walks
//! that same chain; they differ only in what they do with each contribution.
//!
//! # Why the transpose is the cheap direction
//!
//! Walking column `i` produces contributions to many rows `j`, so `A · x`
//! is a *scatter* and needs either per-thread buffers or atomics.  `Aᵀ · x`
//! reads those same contributions but accumulates into `out[i]` alone — a
//! pure gather, so it parallelises over `i` with no synchronisation at all.

use num_complex::Complex;
use quspin_basis::expand::ExpandRefState;
use quspin_basis::traits::BasisSpace;
use quspin_bitbasis::BitInt;
use quspin_operator::Operator;
use quspin_types::QuSpinError;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::apply::ProjectState;
use crate::qmatrix::CIndex;

type C64 = Complex<f64>;

/// Minimum basis size before switching to a parallel column sweep.
const PARALLEL_THRESHOLD: usize = 256;

/// Walk every non-zero contribution in column `i` of `A`.
///
/// Calls `visit(j, value)` once per `(expansion, operator term)` pair whose
/// image lands inside `space`; the same `j` may be visited more than once
/// when several terms hit the same row.
#[inline]
fn visit_column<H, B, C, S, F>(op: &H, space: &S, coeffs: &[C64], i: usize, mut visit: F)
where
    H: Operator<C>,
    B: BitInt,
    C: CIndex,
    S: BasisSpace<B> + ExpandRefState<B, C64, C64> + ProjectState<B>,
    F: FnMut(usize, C64),
{
    let unit = C64::new(1.0, 0.0);
    for (state, amp) in space.expand_ref_state_iter(i, &unit) {
        if amp.norm_sqr() == 0.0 {
            continue;
        }
        op.apply(state, |cindex, op_amp, new_state| {
            if let Some((j, scale)) = space.project(new_state) {
                visit(j, coeffs[cindex.as_usize()] * op_amp * amp * scale);
            }
        });
    }
}

/// Check the argument shapes shared by all three kernels.
fn validate<B, S>(
    num_cindices: usize,
    coeffs_len: usize,
    op_lhss: usize,
    op_max_site: usize,
    space: &S,
    vec_lens: &[(usize, &str)],
) -> Result<(), QuSpinError>
where
    B: BitInt,
    S: BasisSpace<B>,
{
    if coeffs_len != num_cindices {
        return Err(QuSpinError::ValueError(format!(
            "coeffs.len() = {coeffs_len} but operator has {num_cindices} cindices"
        )));
    }
    if op_lhss != space.lhss() {
        return Err(QuSpinError::ValueError(format!(
            "operator lhss={op_lhss} does not match basis lhss={}",
            space.lhss()
        )));
    }
    if op_max_site >= space.n_sites() {
        return Err(QuSpinError::ValueError(format!(
            "operator references site {op_max_site} but basis has only {} sites",
            space.n_sites()
        )));
    }
    for &(len, name) in vec_lens {
        if len != space.size() {
            return Err(QuSpinError::ValueError(format!(
                "{name}.len() = {len} but basis.size() = {}",
                space.size()
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// dot_transpose
// ---------------------------------------------------------------------------

/// `output = Aᵀ · input` (or `+=` when `overwrite` is false).
///
/// Note this is the plain transpose, not the Hermitian adjoint.
pub fn dot_transpose_inner<H, B, C, S>(
    op: &H,
    space: &S,
    coeffs: &[C64],
    input: &[C64],
    output: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    C: CIndex,
    S: BasisSpace<B> + ExpandRefState<B, C64, C64> + ProjectState<B> + Sync,
{
    validate(
        op.num_cindices(),
        coeffs.len(),
        op.lhss(),
        op.max_site(),
        space,
        &[(input.len(), "input"), (output.len(), "output")],
    )?;

    // out[i] = Σ_j A[j, i] · input[j] — each i owns its accumulator, so this
    // is embarrassingly parallel with no atomics or per-thread buffers.
    let compute = |i: usize| {
        let mut acc = C64::default();
        visit_column(op, space, coeffs, i, |j, value| {
            acc += value * input[j];
        });
        acc
    };

    if output.len() < PARALLEL_THRESHOLD {
        for (i, out) in output.iter_mut().enumerate() {
            let acc = compute(i);
            if overwrite {
                *out = acc;
            } else {
                *out += acc;
            }
        }
    } else {
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            let acc = compute(i);
            if overwrite {
                *out = acc;
            } else {
                *out += acc;
            }
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// trace
// ---------------------------------------------------------------------------

/// `Σ_i A[i, i]`.
///
/// One sweep over the basis — the same cost as a single matrix-vector
/// product, not the `O(dim)` matvecs that probing with unit vectors needs.
pub fn trace_inner<H, B, C, S>(op: &H, space: &S, coeffs: &[C64]) -> Result<C64, QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    C: CIndex,
    S: BasisSpace<B> + ExpandRefState<B, C64, C64> + ProjectState<B> + Sync,
{
    validate(
        op.num_cindices(),
        coeffs.len(),
        op.lhss(),
        op.max_site(),
        space,
        &[],
    )?;

    let diag = |i: usize| {
        let mut acc = C64::default();
        visit_column(op, space, coeffs, i, |j, value| {
            if j == i {
                acc += value;
            }
        });
        acc
    };

    let n = space.size();
    let total = if n < PARALLEL_THRESHOLD {
        (0..n).map(diag).fold(C64::default(), |a, b| a + b)
    } else {
        (0..n)
            .into_par_iter()
            .map(diag)
            .reduce(C64::default, |a, b| a + b)
    };
    Ok(total)
}

// ---------------------------------------------------------------------------
// onenorm
// ---------------------------------------------------------------------------

/// `‖A − shift·I‖₁ = max_i Σ_j |A[j, i] − shift·δ_{ji}|`.
///
/// Contributions to the same row are summed *before* taking the modulus, so
/// terms that cancel do not inflate the norm. Each column's contributions are
/// gathered into a `SmallVec` and merged in place; operator strings are
/// few-body, so the inline capacity covers the common case.
pub fn onenorm_inner<H, B, C, S>(
    op: &H,
    space: &S,
    coeffs: &[C64],
    shift: C64,
) -> Result<f64, QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    C: CIndex,
    S: BasisSpace<B> + ExpandRefState<B, C64, C64> + ProjectState<B> + Sync,
{
    validate(
        op.num_cindices(),
        coeffs.len(),
        op.lhss(),
        op.max_site(),
        space,
        &[],
    )?;

    let col_sum = |i: usize| {
        let mut entries: SmallVec<[(usize, C64); 16]> = SmallVec::new();
        visit_column(op, space, coeffs, i, |j, value| entries.push((j, value)));

        entries.sort_unstable_by_key(|&(j, _)| j);

        let mut sum = 0.0_f64;
        let mut seen_diagonal = false;
        let mut k = 0;
        while k < entries.len() {
            let j = entries[k].0;
            let mut acc = C64::default();
            while k < entries.len() && entries[k].0 == j {
                acc += entries[k].1;
                k += 1;
            }
            if j == i {
                acc -= shift;
                seen_diagonal = true;
            }
            sum += acc.norm();
        }
        // A zero diagonal entry still contributes |shift|.
        if !seen_diagonal {
            sum += shift.norm();
        }
        sum
    };

    let n = space.size();
    let max = if n < PARALLEL_THRESHOLD {
        (0..n).map(col_sum).fold(0.0_f64, f64::max)
    } else {
        (0..n)
            .into_par_iter()
            .map(col_sum)
            .reduce(|| 0.0_f64, f64::max)
    };
    Ok(max)
}

// ---------------------------------------------------------------------------
// Type-erased dispatch
// ---------------------------------------------------------------------------
//
// Mirrors the layering in `apply.rs`, but single-basis: these kernels are
// square by construction, so there is no cross-width pairing to reject.
//
// As in `apply.rs`, the `LargeInt` match arms carry a
// `#[cfg(feature = "large-int")]` on the enum variant itself, so the match
// stays exhaustive when the feature is off.

pub use quspin_basis::dispatch::{
    BitBasis, BitBasisDefault, DitBasis, DynDitBasis, DynDitBasisDefault, GenericBasis, QuatBasis,
    QuatBasisDefault, TritBasis, TritBasisDefault,
};
#[cfg(feature = "large-int")]
pub use quspin_basis::dispatch::{
    BitBasisLargeInt, DynDitBasisLargeInt, QuatBasisLargeInt, TritBasisLargeInt,
};

/// Dispatch over a per-family `Default` sub-enum.
macro_rules! dispatch_default {
    ($Enum:ident, $space:expr, |$s:ident| $body:expr) => {
        match $space {
            $Enum::Full32($s) => $body,
            $Enum::Full64($s) => $body,
            $Enum::Sub32($s) => $body,
            $Enum::Sub64($s) => $body,
            $Enum::Sub128($s) => $body,
            $Enum::Sub256($s) => $body,
            $Enum::Sym32($s) => $body,
            $Enum::Sym64($s) => $body,
            $Enum::Sym128($s) => $body,
            $Enum::Sym256($s) => $body,
        }
    };
}

/// Dispatch over a per-family `LargeInt` sub-enum (widths 512..8192).
#[cfg(feature = "large-int")]
macro_rules! dispatch_largeint {
    ($Enum:ident, $space:expr, |$s:ident| $body:expr) => {
        match $space {
            $Enum::Sub512($s) => $body,
            $Enum::Sub1024($s) => $body,
            $Enum::Sub2048($s) => $body,
            $Enum::Sub4096($s) => $body,
            $Enum::Sub8192($s) => $body,
            $Enum::Sym512($s) => $body,
            $Enum::Sym1024($s) => $body,
            $Enum::Sym2048($s) => $body,
            $Enum::Sym4096($s) => $body,
            $Enum::Sym8192($s) => $body,
        }
    };
}

/// Generate the `_bit` / `_dit` / generic entry-point triple for one kernel.
macro_rules! entry_points {
    (
        $(#[$meta:meta])*
        $name:ident, $inner:ident, $ret:ty, ($($arg:ident : $ty:ty),* $(,)?)
    ) => {
        paste::paste! {
            $(#[$meta])*
            ///
            /// Bit-family entry point (LHSS = 2) — used by the `FermionBasis`
            /// path so it never instantiates the dit families.
            pub fn [<$name _bit>]<H, C>(
                op: &H,
                space: &BitBasis,
                coeffs: &[C64],
                $($arg: $ty),*
            ) -> Result<$ret, QuSpinError>
            where
                H: Operator<C> + Sync,
                C: CIndex,
            {
                match space {
                    BitBasis::Default(d) => {
                        dispatch_default!(BitBasisDefault, d, |s| $inner(op, s, coeffs $(, $arg)*))
                    }
                    #[cfg(feature = "large-int")]
                    BitBasis::LargeInt(d) => {
                        dispatch_largeint!(BitBasisLargeInt, d, |s| $inner(op, s, coeffs $(, $arg)*))
                    }
                }
            }

            $(#[$meta])*
            ///
            /// Dit-family entry point (LHSS > 2).
            pub fn [<$name _dit>]<H, C>(
                op: &H,
                space: &DitBasis,
                coeffs: &[C64],
                $($arg: $ty),*
            ) -> Result<$ret, QuSpinError>
            where
                H: Operator<C> + Sync,
                C: CIndex,
            {
                match space {
                    DitBasis::Trit(TritBasis::Default(d)) => {
                        dispatch_default!(TritBasisDefault, d, |s| $inner(op, s, coeffs $(, $arg)*))
                    }
                    #[cfg(feature = "large-int")]
                    DitBasis::Trit(TritBasis::LargeInt(d)) => {
                        dispatch_largeint!(TritBasisLargeInt, d, |s| $inner(op, s, coeffs $(, $arg)*))
                    }
                    DitBasis::Quat(QuatBasis::Default(d)) => {
                        dispatch_default!(QuatBasisDefault, d, |s| $inner(op, s, coeffs $(, $arg)*))
                    }
                    #[cfg(feature = "large-int")]
                    DitBasis::Quat(QuatBasis::LargeInt(d)) => {
                        dispatch_largeint!(QuatBasisLargeInt, d, |s| $inner(op, s, coeffs $(, $arg)*))
                    }
                    DitBasis::Dyn(DynDitBasis::Default(d)) => {
                        dispatch_default!(DynDitBasisDefault, d, |s| $inner(op, s, coeffs $(, $arg)*))
                    }
                    #[cfg(feature = "large-int")]
                    DitBasis::Dyn(DynDitBasis::LargeInt(d)) => {
                        dispatch_largeint!(DynDitBasisLargeInt, d, |s| $inner(op, s, coeffs $(, $arg)*))
                    }
                }
            }

            $(#[$meta])*
            ///
            /// Entry point over the full `GenericBasis` dispatch root.
            pub fn $name<H, C>(
                op: &H,
                space: &GenericBasis,
                coeffs: &[C64],
                $($arg: $ty),*
            ) -> Result<$ret, QuSpinError>
            where
                H: Operator<C> + Sync,
                C: CIndex,
            {
                match space {
                    GenericBasis::Bit(b) => [<$name _bit>](op, b, coeffs $(, $arg)*),
                    GenericBasis::Dit(d) => [<$name _dit>](op, d, coeffs $(, $arg)*),
                }
            }
        }
    };
}

entry_points!(
    /// `output = Aᵀ · input` (or `+=` when `overwrite` is false).
    dot_transpose, dot_transpose_inner, (),
    (input: &[C64], output: &mut [C64], overwrite: bool)
);

entry_points!(
    /// `Σ_i A[i, i]`.
    trace, trace_inner, C64, ()
);

entry_points!(
    /// `‖A − shift·I‖₁`.
    onenorm, onenorm_inner, f64, (shift: C64)
);
