//! Type-erased row production for matrix building.
//!
//! # Why
//!
//! `build_from_basis<H, B, M, I, C, S>` carries six type parameters, and the
//! rayon `collect` and the per-row `sort_unstable_by` sit at its innermost
//! level. Every combination therefore gets its own copy of rayon's job
//! machinery and its own pdqsort instantiation. `cargo llvm-lines` on
//! `quspin-matrix` attributes **50.8%** of all LLVM IR to rayon (228,503
//! instantiations) and **17.5%** to `core::slice::sort` (36,376) — against
//! 13.3% for QuSpin's own code. The library code is not incidental; it is
//! what the type-parameter product multiplies.
//!
//! # How
//!
//! The state integer `B` never escapes a row build: `state_at` produces it,
//! `Operator::apply` threads it, and `BasisSpace::index` consumes it into a
//! plain `usize`. `Operator::apply` already fixes the amplitude as
//! `Complex<f64>`, so no element type is involved either. That makes `B`,
//! `S` and `C` erasable behind an object-safe trait whose intermediate is
//! one concrete type, [`RawEntry`].
//!
//! The generic adapter ([`BasisRows`]) keeps the type parameters but holds
//! only the apply loop and a `push`. The expensive parts — the parallel
//! collect and the sort — see `&dyn RowSource` and `Vec<RawEntry>`, so they
//! instantiate **once** instead of once per combination.
//!
//! Costs, stated plainly: one virtual call per *row* (not per non-zero,
//! since entries land in a concrete `Vec`), and a 32-byte intermediate
//! entry where `Entry<f32, i32, u8>` would be 12. Both are build-phase
//! only; `QMatrix::dot` is untouched.

use std::marker::PhantomData;

use num_complex::Complex;
use quspin_basis::BasisSpace;
use quspin_basis::sym::{NormInt, SymBasis};
use quspin_bitbasis::{BitInt, FermionicBitStateOp};
use quspin_operator::Operator;
use rayon::prelude::*;
use smallvec::SmallVec;

use super::matrix::PARALLEL_DIM_THRESHOLD;
use super::{CIndex, Entry, Index, QMatrix};
use quspin_types::Primitive;

/// One intermediate matrix entry, independent of `M`, `I`, `C` and `B`.
///
/// `cindex` is widened to `u32` and narrowed back through
/// [`CIndex::from_usize`] at assembly, which is what removes `C` from the
/// parallel and sorting code paths.
#[derive(Clone, Copy, Debug)]
pub struct RawEntry {
    pub col: usize,
    pub cindex: u32,
    pub amp: Complex<f64>,
}

/// A per-row producer with the state integer and basis space erased.
///
/// Object-safe by construction: every parameter and return is concrete.
pub trait RowSource: Sync {
    /// Number of rows (basis dimension).
    fn dim(&self) -> usize;

    /// Append every non-zero contribution in row `row_idx` to `out`.
    ///
    /// Duplicates are allowed — the driver coalesces them. Implementations
    /// must push in the operator's natural emission order, because that
    /// order fixes the floating-point summation order during coalescing.
    fn row_into(&self, row_idx: usize, out: &mut Vec<RawEntry>);
}

// ---------------------------------------------------------------------------
// Adapter: plain (non-symmetric) basis
// ---------------------------------------------------------------------------

/// Adapts a `Hamiltonian` + non-symmetric basis pair to [`RowSource`].
///
/// This is the only piece that stays generic over `<H, B, C, S>`, and it
/// holds nothing but the apply loop — no rayon, no sort, no `Vec<Entry>`.
pub struct BasisRows<'a, H, B, C, S> {
    ham: &'a H,
    basis: &'a S,
    _pd: PhantomData<fn() -> (B, C)>,
}

impl<'a, H, B, C, S> BasisRows<'a, H, B, C, S> {
    pub fn new(ham: &'a H, basis: &'a S) -> Self {
        BasisRows {
            ham,
            basis,
            _pd: PhantomData,
        }
    }
}

impl<H, B, C, S> RowSource for BasisRows<'_, H, B, C, S>
where
    H: Operator<C> + Sync,
    B: BitInt,
    C: CIndex,
    S: BasisSpace<B> + Sync,
{
    #[inline]
    fn dim(&self) -> usize {
        self.basis.size()
    }

    fn row_into(&self, row_idx: usize, out: &mut Vec<RawEntry>) {
        let state = self.basis.state_at(row_idx);
        self.ham.apply(state, |cindex, amp, new_state| {
            // `B` dies here: `index` turns it into a plain row number.
            let Some(col) = self.basis.index(new_state) else {
                return;
            };
            out.push(RawEntry {
                col,
                cindex: cindex.as_usize() as u32,
                amp,
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Adapter: symmetry-reduced basis
// ---------------------------------------------------------------------------

/// Adapts a `Hamiltonian` + [`SymBasis`] pair to [`RowSource`].
///
/// Same containment argument as [`BasisRows`], with one extra hop: the
/// emitted states go through `get_refstate_batch` to find their orbit
/// representatives before `index` turns them into row numbers. `B` still
/// never leaves the method.
pub struct SymRows<'a, H, B: BitInt, L, N: NormInt, C> {
    ham: &'a H,
    basis: &'a SymBasis<B, L, N>,
    _pd: PhantomData<fn() -> C>,
}

impl<'a, H, B: BitInt, L, N: NormInt, C> SymRows<'a, H, B, L, N, C> {
    pub fn new(ham: &'a H, basis: &'a SymBasis<B, L, N>) -> Self {
        SymRows {
            ham,
            basis,
            _pd: PhantomData,
        }
    }
}

/// Inline capacity for the per-row scratch buffers.
const ROW_CAP: usize = 64;

impl<H, B, L, N, C> RowSource for SymRows<'_, H, B, L, N, C>
where
    H: Operator<C> + Sync,
    B: BitInt,
    L: FermionicBitStateOp<B> + Sync,
    N: NormInt,
    C: CIndex,
{
    #[inline]
    fn dim(&self) -> usize {
        self.basis.size()
    }

    fn row_into(&self, row_idx: usize, out: &mut Vec<RawEntry>) {
        let (state, norm) = self.basis.entry(row_idx);
        let mut row_buf: SmallVec<[(C, Complex<f64>); ROW_CAP]> = SmallVec::new();
        let mut new_states: SmallVec<[B; ROW_CAP]> = SmallVec::new();
        let mut ref_out: SmallVec<[(B, Complex<f64>); ROW_CAP]> = SmallVec::new();

        self.ham.apply(state, |cindex, amp, new_state| {
            row_buf.push((cindex, amp));
            new_states.push(new_state);
        });

        if new_states.is_empty() {
            return;
        }
        ref_out.resize(new_states.len(), (new_states[0], Complex::new(1.0, 0.0)));
        self.basis.get_refstate_batch(&new_states, &mut ref_out);

        for ((cindex, amp), (ref_state, grp_char)) in row_buf.iter().zip(ref_out.iter()) {
            // `B` dies here, same as in the plain path.
            let Some(col) = self.basis.index(*ref_state) else {
                continue;
            };
            let (_, new_norm) = self.basis.entry(col);
            let scale = grp_char * (new_norm / norm).sqrt();
            out.push(RawEntry {
                col,
                cindex: cindex.as_usize() as u32,
                amp: amp * scale,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Erased driver — one rayon instantiation, one sort instantiation
// ---------------------------------------------------------------------------

/// Sort a row by `(col, cindex)` and sum duplicates.
///
/// Uses a **stable** sort so that entries sharing a key keep their emission
/// order, which makes the coalesced sum bit-identical to the previous
/// implementation's accumulate-on-first-match behaviour.
fn coalesce_row(row: &mut Vec<RawEntry>) {
    row.sort_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));
    let mut write = 0usize;
    for read in 0..row.len() {
        let cur = row[read];
        if write > 0 && row[write - 1].col == cur.col && row[write - 1].cindex == cur.cindex {
            row[write - 1].amp += cur.amp;
        } else {
            row[write] = cur;
            write += 1;
        }
    }
    row.truncate(write);
}

/// Build every row through a type-erased source.
///
/// Takes `&dyn RowSource`, so the closure handed to rayon and the slice
/// handed to the sort both have exactly one type no matter how many
/// `(H, B, C, S)` combinations exist upstream.
pub fn build_raw_rows(src: &dyn RowSource) -> Vec<Vec<RawEntry>> {
    let dim = src.dim();
    let build_row = |row_idx: usize| -> Vec<RawEntry> {
        let mut row = Vec::new();
        src.row_into(row_idx, &mut row);
        coalesce_row(&mut row);
        row
    };

    if dim >= PARALLEL_DIM_THRESHOLD {
        (0..dim).into_par_iter().map(build_row).collect()
    } else {
        (0..dim).map(build_row).collect()
    }
}

/// Assemble erased rows into a typed `QMatrix`.
///
/// Generic over `M`, `I` and `C`, but contains no parallelism and no
/// sorting — so this is the only code the element-type axes multiply.
pub fn raw_rows_to_qmatrix<M: Primitive, I: Index, C: CIndex>(
    dim: usize,
    rows: Vec<Vec<RawEntry>>,
) -> QMatrix<M, I, C> {
    let total_nnz: usize = rows.iter().map(|r| r.len()).sum();
    let mut indptr = Vec::with_capacity(dim + 1);
    let mut data: Vec<Entry<M, I, C>> = Vec::with_capacity(total_nnz);
    indptr.push(I::from_usize(0));
    for row in rows {
        for e in row {
            data.push(Entry::new(
                M::from_complex(e.amp),
                I::from_usize(e.col),
                C::from_usize(e.cindex as usize),
            ));
        }
        indptr.push(I::from_usize(data.len()));
    }
    QMatrix::from_csr(indptr, data)
}
