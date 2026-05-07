//! Row-range CSR materialisation that bypasses `QMatrix`.
//!
//! Each call walks rows `[row_start, row_end)` of the operator + basis pair
//! directly, returning `(indptr, indices, data)` in the layout petsc4py's
//! `Mat.setValuesCSR(I, J, V)` expects.  Memory is bounded by the slab —
//! useful when each MPI rank only needs its locally-owned rows of a matrix
//! that's otherwise too large to materialise globally.

use num_complex::Complex;
use quspin_basis::BasisSpace;
use quspin_bitbasis::BitInt;
use quspin_operator::Operator;
use quspin_types::QuSpinError;
use rayon::prelude::*;

use crate::qmatrix::CIndex;
use crate::qmatrix::matrix::PARALLEL_DIM_THRESHOLD;

/// Drop-zeros tolerance — matches `QMatrix::to_csr_into` / `materialize`.
const ZERO_TOL: f64 = 4.0 * f64::EPSILON;

/// Output triple of [`csr_slab_kernel`] and the public dispatchers.
///
/// - `indptr` length `row_end - row_start + 1`, zero-based, `indptr[0] == 0`,
///   `indptr[-1] == nnz_local`.
/// - `indices` length `nnz_local`, **global** column indices.
/// - `data` length `nnz_local`, accumulated `Σ_c coeffs[c] * stored_value`.
pub type CsrSlab = (Vec<i64>, Vec<i64>, Vec<Complex<f64>>);

/// Row-range CSR kernel — generic over operator/bitint/cindex/basis-space.
///
/// Validates `row_start <= row_end <= basis.size()` and
/// `coeffs.len() == op.num_cindices()`, then walks each row in parallel
/// (when the slab covers `PARALLEL_DIM_THRESHOLD` or more rows) and emits the
/// merged CSR slab.
pub fn csr_slab_kernel<H, B, C, S>(
    op: &H,
    basis: &S,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    C: CIndex + Copy,
    S: BasisSpace<B> + Sync,
{
    if row_start > row_end {
        return Err(QuSpinError::ValueError(format!(
            "row_start ({row_start}) must be <= row_end ({row_end})"
        )));
    }
    let dim = basis.size();
    if row_end > dim {
        return Err(QuSpinError::ValueError(format!(
            "row_end ({row_end}) must be <= basis.size() ({dim})"
        )));
    }
    if coeffs.len() != op.num_cindices() {
        return Err(QuSpinError::ValueError(format!(
            "coeffs length {} must equal op.num_cindices() {}",
            coeffs.len(),
            op.num_cindices()
        )));
    }

    let n_local = row_end - row_start;

    // Per-row work: collect (col, contribution) entries via op.apply, sort + merge.
    let process_row = |r: usize| -> Vec<(i64, Complex<f64>)> {
        let state = basis.state_at(r);
        let mut entries: Vec<(i64, Complex<f64>)> = Vec::new();
        op.apply(state, |cindex, amp, new_state| {
            if let Some(col) = basis.index(new_state) {
                entries.push((col as i64, amp * coeffs[cindex.as_usize()]));
            }
        });
        // Sort by col, merge same-col, drop_zeros (relative tolerance).
        entries.sort_unstable_by_key(|&(c, _)| c);
        let mut merged: Vec<(i64, Complex<f64>)> = Vec::with_capacity(entries.len());
        let mut i = 0;
        while i < entries.len() {
            let col = entries[i].0;
            let mut acc = Complex::new(0.0, 0.0);
            let mut scale = 0.0f64;
            while i < entries.len() && entries[i].0 == col {
                let v = entries[i].1;
                acc += v;
                scale += v.norm();
                i += 1;
            }
            if !drop_zeros || acc.norm() > scale * ZERO_TOL {
                merged.push((col, acc));
            }
        }
        merged
    };

    let row_results: Vec<Vec<(i64, Complex<f64>)>> = if n_local >= PARALLEL_DIM_THRESHOLD {
        (row_start..row_end)
            .into_par_iter()
            .map(process_row)
            .collect()
    } else {
        (row_start..row_end).map(process_row).collect()
    };

    // Flatten into CSR.
    let total_nnz: usize = row_results.iter().map(|r| r.len()).sum();
    let mut indptr: Vec<i64> = Vec::with_capacity(n_local + 1);
    let mut indices: Vec<i64> = Vec::with_capacity(total_nnz);
    let mut data: Vec<Complex<f64>> = Vec::with_capacity(total_nnz);
    indptr.push(0);
    for row in row_results {
        for (col, val) in row {
            indices.push(col);
            data.push(val);
        }
        indptr.push(indices.len() as i64);
    }
    Ok((indptr, indices, data))
}

// ===========================================================================
// Symmetric-basis kernel (mirrors build_from_symmetric)
// ===========================================================================

use quspin_basis::sym::{NormInt, SymBasis};
use quspin_bitbasis::{DynamicPermDitValues, FermionicBitStateOp, PermDitMask, PermDitValues};
use smallvec::SmallVec;

/// Companion to `csr_slab_kernel` for symmetric bases. `SymBasis<B, L, N>`
/// stores only orbit representatives; the Hamiltonian matrix elements must
/// be weighted by the orbit-norm factor `grp_char * sqrt(new_norm / norm)`,
/// exactly as in `build_from_symmetric`. This kernel applies that weighting
/// within the `[row_start, row_end)` slab.
fn csr_slab_kernel_sym<H, B, L, N, C>(
    op: &H,
    basis: &SymBasis<B, L, N>,
    coeffs: &[num_complex::Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    L: FermionicBitStateOp<B> + Sync,
    N: NormInt,
    C: CIndex + Copy,
{
    if row_start > row_end {
        return Err(QuSpinError::ValueError(format!(
            "row_start ({row_start}) must be <= row_end ({row_end})"
        )));
    }
    let dim = basis.size();
    if row_end > dim {
        return Err(QuSpinError::ValueError(format!(
            "row_end ({row_end}) must be <= basis.size() ({dim})"
        )));
    }
    if coeffs.len() != op.num_cindices() {
        return Err(QuSpinError::ValueError(format!(
            "coeffs length {} must equal op.num_cindices() {}",
            coeffs.len(),
            op.num_cindices()
        )));
    }

    let n_local = row_end - row_start;
    const ROW_CAP: usize = 64;

    let process_row = |r: usize| -> Vec<(i64, num_complex::Complex<f64>)> {
        let (state, norm) = basis.entry(r);
        let mut row_buf: SmallVec<[(C, num_complex::Complex<f64>); ROW_CAP]> = SmallVec::new();
        let mut new_states: SmallVec<[B; ROW_CAP]> = SmallVec::new();
        let mut ref_out: SmallVec<[(B, num_complex::Complex<f64>); ROW_CAP]> = SmallVec::new();

        op.apply(state, |cindex, amp, new_state| {
            row_buf.push((cindex, amp));
            new_states.push(new_state);
        });

        let mut entries: Vec<(i64, num_complex::Complex<f64>)> = Vec::new();

        if !new_states.is_empty() {
            ref_out.resize(
                new_states.len(),
                (new_states[0], num_complex::Complex::new(1.0, 0.0)),
            );
            basis.get_refstate_batch(&new_states, &mut ref_out);

            for ((cindex, amp), (ref_state, grp_char)) in row_buf.iter().zip(ref_out.iter()) {
                let Some(col_idx) = basis.index(*ref_state) else {
                    continue;
                };
                let (_, new_norm) = basis.entry(col_idx);
                let scale = grp_char * (new_norm / norm).sqrt();
                let full_amp = amp * scale;
                let weighted = full_amp * coeffs[cindex.as_usize()];
                entries.push((col_idx as i64, weighted));
            }
        }

        // Sort by col, then merge same-col entries.
        entries.sort_unstable_by_key(|&(c, _)| c);
        let mut merged: Vec<(i64, num_complex::Complex<f64>)> = Vec::with_capacity(entries.len());
        let mut i = 0;
        while i < entries.len() {
            let col = entries[i].0;
            let mut acc = num_complex::Complex::new(0.0, 0.0);
            let mut scale = 0.0f64;
            while i < entries.len() && entries[i].0 == col {
                let v = entries[i].1;
                acc += v;
                scale += v.norm();
                i += 1;
            }
            if !drop_zeros || acc.norm() > scale * ZERO_TOL {
                merged.push((col, acc));
            }
        }
        merged
    };

    let row_results: Vec<Vec<(i64, num_complex::Complex<f64>)>> =
        if n_local >= PARALLEL_DIM_THRESHOLD {
            (row_start..row_end)
                .into_par_iter()
                .map(process_row)
                .collect()
        } else {
            (row_start..row_end).map(process_row).collect()
        };

    // Flatten into CSR.
    let total_nnz: usize = row_results.iter().map(|r| r.len()).sum();
    let mut indptr: Vec<i64> = Vec::with_capacity(n_local + 1);
    let mut indices: Vec<i64> = Vec::with_capacity(total_nnz);
    let mut data: Vec<num_complex::Complex<f64>> = Vec::with_capacity(total_nnz);
    indptr.push(0);
    for row in row_results {
        for (col, val) in row {
            indices.push(col);
            data.push(val);
        }
        indptr.push(indices.len() as i64);
    }
    Ok((indptr, indices, data))
}

// ===========================================================================
// Pauli (HardcoreOperator) dispatchers
// ===========================================================================

use quspin_basis::dispatch::{
    BitBasis, BitBasisDefault, DitBasis, DynDitBasis, DynDitBasisDefault, GenericBasis, QuatBasis,
    QuatBasisDefault, TritBasis, TritBasisDefault,
};
#[cfg(feature = "large-int")]
use quspin_basis::dispatch::{
    BitBasisLargeInt, DynDitBasisLargeInt, QuatBasisLargeInt, TritBasisLargeInt,
};
use quspin_operator::pauli::{HardcoreOperator, HardcoreOperatorInner};

type B128 = ruint::Uint<128, 2>;
type B256 = ruint::Uint<256, 4>;
#[cfg(feature = "large-int")]
type B512 = ruint::Uint<512, 8>;
#[cfg(feature = "large-int")]
type B1024 = ruint::Uint<1024, 16>;
#[cfg(feature = "large-int")]
type B2048 = ruint::Uint<2048, 32>;
#[cfg(feature = "large-int")]
type B4096 = ruint::Uint<4096, 64>;
#[cfg(feature = "large-int")]
type B8192 = ruint::Uint<8192, 128>;

// ---------------------------------------------------------------------------
// Slab variants of the plain_default_arms! / plain_largeint_arms! macros.
// These mirror their counterparts in qmatrix/build.rs but call
// csr_slab_kernel instead of build_from_basis.
// ---------------------------------------------------------------------------

macro_rules! slab_default_arms {
    ($Enum:ident, $self:expr, $op:ident, $C:ty, $coeffs:ident, $rs:ident, $re:ident, $dz:ident) => {
        match $self {
            $Enum::Full32(b) => csr_slab_kernel::<_, u32, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Full64(b) => csr_slab_kernel::<_, u64, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Sub32(b) => csr_slab_kernel::<_, u32, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Sub64(b) => csr_slab_kernel::<_, u64, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Sub128(b) => csr_slab_kernel::<_, B128, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Sub256(b) => csr_slab_kernel::<_, B256, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            _ => unreachable!("only plain (Full*/Sub*) variants reach this branch"),
        }
    };
}

#[cfg(feature = "large-int")]
macro_rules! slab_largeint_arms {
    ($Enum:ident, $self:expr, $op:ident, $C:ty, $coeffs:ident, $rs:ident, $re:ident, $dz:ident) => {
        match $self {
            $Enum::Sub512(b) => csr_slab_kernel::<_, B512, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Sub1024(b) => csr_slab_kernel::<_, B1024, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Sub2048(b) => csr_slab_kernel::<_, B2048, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Sub4096(b) => csr_slab_kernel::<_, B4096, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            $Enum::Sub8192(b) => csr_slab_kernel::<_, B8192, $C, _>($op, b, $coeffs, $rs, $re, $dz),
            _ => unreachable!("only plain (Sub*) variants reach this branch"),
        }
    };
}

// ---------------------------------------------------------------------------
// Public dispatchers
// ---------------------------------------------------------------------------

/// Slab dispatcher for `HardcoreOperatorInner` + `GenericBasis` (the `SpinBasis` path).
pub fn csr_slab_pauli_generic(
    op: &HardcoreOperatorInner,
    basis: &GenericBasis,
    coeffs: &[num_complex::Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError> {
    match basis {
        GenericBasis::Bit(b) => csr_slab_pauli_bit(op, b, coeffs, row_start, row_end, drop_zeros),
        GenericBasis::Dit(b) => csr_slab_pauli_dit(op, b, coeffs, row_start, row_end, drop_zeros),
    }
}

/// Slab dispatcher for `HardcoreOperatorInner` + `BitBasis` (the `FermionBasis` path
/// and the `BitBasis` arm of `GenericBasis`).
pub fn csr_slab_pauli_bit(
    op: &HardcoreOperatorInner,
    basis: &BitBasis,
    coeffs: &[num_complex::Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError> {
    match op {
        HardcoreOperatorInner::Ham8(h) => {
            csr_slab_pauli_bit_dispatch::<u8>(h, basis, coeffs, row_start, row_end, drop_zeros)
        }
        HardcoreOperatorInner::Ham16(h) => {
            csr_slab_pauli_bit_dispatch::<u16>(h, basis, coeffs, row_start, row_end, drop_zeros)
        }
    }
}

fn csr_slab_pauli_dit(
    op: &HardcoreOperatorInner,
    basis: &DitBasis,
    coeffs: &[num_complex::Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError> {
    match op {
        HardcoreOperatorInner::Ham8(h) => {
            csr_slab_pauli_dit_dispatch::<u8>(h, basis, coeffs, row_start, row_end, drop_zeros)
        }
        HardcoreOperatorInner::Ham16(h) => {
            csr_slab_pauli_dit_dispatch::<u16>(h, basis, coeffs, row_start, row_end, drop_zeros)
        }
    }
}

// ---------------------------------------------------------------------------
// Per-cindex-width bit dispatch — mirrors build_from_bit
// ---------------------------------------------------------------------------

fn csr_slab_pauli_bit_dispatch<C: CIndex + Copy>(
    op: &HardcoreOperator<C>,
    space: &BitBasis,
    coeffs: &[num_complex::Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError> {
    let rs = row_start;
    let re = row_end;
    let dz = drop_zeros;
    match space {
        BitBasis::Default(d) => match d {
            BitBasisDefault::Sym32(b) => {
                csr_slab_kernel_sym::<_, u32, PermDitMask<u32>, u8, C>(op, b, coeffs, rs, re, dz)
            }
            BitBasisDefault::Sym64(b) => {
                csr_slab_kernel_sym::<_, u64, PermDitMask<u64>, u16, C>(op, b, coeffs, rs, re, dz)
            }
            BitBasisDefault::Sym128(b) => {
                csr_slab_kernel_sym::<_, B128, PermDitMask<B128>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            BitBasisDefault::Sym256(b) => {
                csr_slab_kernel_sym::<_, B256, PermDitMask<B256>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            other => slab_default_arms!(BitBasisDefault, other, op, C, coeffs, rs, re, dz),
        },
        #[cfg(feature = "large-int")]
        BitBasis::LargeInt(d) => match d {
            BitBasisLargeInt::Sym512(b) => {
                csr_slab_kernel_sym::<_, B512, PermDitMask<B512>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            BitBasisLargeInt::Sym1024(b) => {
                csr_slab_kernel_sym::<_, B1024, PermDitMask<B1024>, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            BitBasisLargeInt::Sym2048(b) => {
                csr_slab_kernel_sym::<_, B2048, PermDitMask<B2048>, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            BitBasisLargeInt::Sym4096(b) => {
                csr_slab_kernel_sym::<_, B4096, PermDitMask<B4096>, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            BitBasisLargeInt::Sym8192(b) => {
                csr_slab_kernel_sym::<_, B8192, PermDitMask<B8192>, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            other => slab_largeint_arms!(BitBasisLargeInt, other, op, C, coeffs, rs, re, dz),
        },
    }
}

// ---------------------------------------------------------------------------
// Per-cindex-width dit dispatch — mirrors build_from_dit
// ---------------------------------------------------------------------------

fn csr_slab_pauli_dit_dispatch<C: CIndex + Copy>(
    op: &HardcoreOperator<C>,
    space: &DitBasis,
    coeffs: &[num_complex::Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError> {
    let rs = row_start;
    let re = row_end;
    let dz = drop_zeros;
    match space {
        DitBasis::Trit(TritBasis::Default(d)) => match d {
            TritBasisDefault::Sym32(b) => {
                csr_slab_kernel_sym::<_, u32, PermDitValues<3>, u8, C>(op, b, coeffs, rs, re, dz)
            }
            TritBasisDefault::Sym64(b) => {
                csr_slab_kernel_sym::<_, u64, PermDitValues<3>, u16, C>(op, b, coeffs, rs, re, dz)
            }
            TritBasisDefault::Sym128(b) => {
                csr_slab_kernel_sym::<_, B128, PermDitValues<3>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            TritBasisDefault::Sym256(b) => {
                csr_slab_kernel_sym::<_, B256, PermDitValues<3>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            other => slab_default_arms!(TritBasisDefault, other, op, C, coeffs, rs, re, dz),
        },
        #[cfg(feature = "large-int")]
        DitBasis::Trit(TritBasis::LargeInt(d)) => match d {
            TritBasisLargeInt::Sym512(b) => {
                csr_slab_kernel_sym::<_, B512, PermDitValues<3>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            TritBasisLargeInt::Sym1024(b) => {
                csr_slab_kernel_sym::<_, B1024, PermDitValues<3>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            TritBasisLargeInt::Sym2048(b) => {
                csr_slab_kernel_sym::<_, B2048, PermDitValues<3>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            TritBasisLargeInt::Sym4096(b) => {
                csr_slab_kernel_sym::<_, B4096, PermDitValues<3>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            TritBasisLargeInt::Sym8192(b) => {
                csr_slab_kernel_sym::<_, B8192, PermDitValues<3>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            other => slab_largeint_arms!(TritBasisLargeInt, other, op, C, coeffs, rs, re, dz),
        },
        DitBasis::Quat(QuatBasis::Default(d)) => match d {
            QuatBasisDefault::Sym32(b) => {
                csr_slab_kernel_sym::<_, u32, PermDitValues<4>, u8, C>(op, b, coeffs, rs, re, dz)
            }
            QuatBasisDefault::Sym64(b) => {
                csr_slab_kernel_sym::<_, u64, PermDitValues<4>, u16, C>(op, b, coeffs, rs, re, dz)
            }
            QuatBasisDefault::Sym128(b) => {
                csr_slab_kernel_sym::<_, B128, PermDitValues<4>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            QuatBasisDefault::Sym256(b) => {
                csr_slab_kernel_sym::<_, B256, PermDitValues<4>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            other => slab_default_arms!(QuatBasisDefault, other, op, C, coeffs, rs, re, dz),
        },
        #[cfg(feature = "large-int")]
        DitBasis::Quat(QuatBasis::LargeInt(d)) => match d {
            QuatBasisLargeInt::Sym512(b) => {
                csr_slab_kernel_sym::<_, B512, PermDitValues<4>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            QuatBasisLargeInt::Sym1024(b) => {
                csr_slab_kernel_sym::<_, B1024, PermDitValues<4>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            QuatBasisLargeInt::Sym2048(b) => {
                csr_slab_kernel_sym::<_, B2048, PermDitValues<4>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            QuatBasisLargeInt::Sym4096(b) => {
                csr_slab_kernel_sym::<_, B4096, PermDitValues<4>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            QuatBasisLargeInt::Sym8192(b) => {
                csr_slab_kernel_sym::<_, B8192, PermDitValues<4>, u32, C>(op, b, coeffs, rs, re, dz)
            }
            other => slab_largeint_arms!(QuatBasisLargeInt, other, op, C, coeffs, rs, re, dz),
        },
        DitBasis::Dyn(DynDitBasis::Default(d)) => match d {
            DynDitBasisDefault::Sym32(b) => {
                csr_slab_kernel_sym::<_, u32, DynamicPermDitValues, u8, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            DynDitBasisDefault::Sym64(b) => {
                csr_slab_kernel_sym::<_, u64, DynamicPermDitValues, u16, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            DynDitBasisDefault::Sym128(b) => {
                csr_slab_kernel_sym::<_, B128, DynamicPermDitValues, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            DynDitBasisDefault::Sym256(b) => {
                csr_slab_kernel_sym::<_, B256, DynamicPermDitValues, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            other => slab_default_arms!(DynDitBasisDefault, other, op, C, coeffs, rs, re, dz),
        },
        #[cfg(feature = "large-int")]
        DitBasis::Dyn(DynDitBasis::LargeInt(d)) => match d {
            DynDitBasisLargeInt::Sym512(b) => {
                csr_slab_kernel_sym::<_, B512, DynamicPermDitValues, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            DynDitBasisLargeInt::Sym1024(b) => {
                csr_slab_kernel_sym::<_, B1024, DynamicPermDitValues, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            DynDitBasisLargeInt::Sym2048(b) => {
                csr_slab_kernel_sym::<_, B2048, DynamicPermDitValues, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            DynDitBasisLargeInt::Sym4096(b) => {
                csr_slab_kernel_sym::<_, B4096, DynamicPermDitValues, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            DynDitBasisLargeInt::Sym8192(b) => {
                csr_slab_kernel_sym::<_, B8192, DynamicPermDitValues, u32, C>(
                    op, b, coeffs, rs, re, dz,
                )
            }
            other => slab_largeint_arms!(DynDitBasisLargeInt, other, op, C, coeffs, rs, re, dz),
        },
    }
}
