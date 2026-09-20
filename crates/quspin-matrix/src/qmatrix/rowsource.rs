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
    /// Duplicates are allowed: the row is sorted in [`build_raw_rows`] and
    /// duplicates are summed later, in [`raw_rows_to_qmatrix`], where the
    /// element type is known. Implementations must push in the operator's
    /// natural emission order — the sort is stable, so that order survives
    /// and fixes the floating-point summation order.
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

/// Order a row by `(col, cindex)`, leaving duplicates adjacent.
///
/// The sort is **stable**, so entries sharing a key keep the operator's
/// emission order. Duplicates are *not* summed here: see
/// [`raw_rows_to_qmatrix`] for why that has to happen in the typed stage.
fn sort_row(row: &mut [RawEntry]) {
    row.sort_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));
}

/// Build every row through a type-erased source.
///
/// Takes `&dyn RowSource`, so the closure handed to rayon and the slice
/// handed to the sort both have exactly one type no matter how many
/// `(H, B, C, S)` combinations exist upstream. This is where the
/// instantiation count collapses: one rayon copy, one pdqsort copy.
pub fn build_raw_rows(src: &dyn RowSource) -> Vec<Vec<RawEntry>> {
    let dim = src.dim();
    let build_row = |row_idx: usize| -> Vec<RawEntry> {
        let mut row = Vec::new();
        src.row_into(row_idx, &mut row);
        sort_row(&mut row);
        row
    };

    if dim >= PARALLEL_DIM_THRESHOLD {
        (0..dim).into_par_iter().map(build_row).collect()
    } else {
        (0..dim).map(build_row).collect()
    }
}

/// Number of distinct `(col, cindex)` keys in an already-sorted row, i.e.
/// how many entries survive coalescing.
fn distinct_keys(row: &[RawEntry]) -> usize {
    let boundaries = row
        .windows(2)
        .filter(|w| w[0].col != w[1].col || w[0].cindex != w[1].cindex)
        .count();
    boundaries + usize::from(!row.is_empty())
}

/// Assemble sorted erased rows into a typed `QMatrix`, summing duplicates.
///
/// Generic over `M`, `I` and `C`, but contains no parallelism and no sort —
/// those stayed in [`build_raw_rows`], so the element-type axes multiply
/// only this small loop.
///
/// # Why coalescing lives here and not in the erased stage
///
/// Summing in `Complex<f64>` and narrowing once is *not* equivalent to the
/// previous implementation, which narrowed to `M` at every addition. For
/// `M = f32` the contributions `16_777_216`, `1`, `-16_777_216` sum to `1`
/// under deferred narrowing but to `0` under per-addition narrowing,
/// because `16_777_217` is not representable in `f32`; integer `M` can
/// diverge further. Accumulating through `M` here keeps the stored values
/// bit-identical to the pre-refactor behaviour.
pub fn raw_rows_to_qmatrix<M: Primitive, I: Index, C: CIndex>(
    dim: usize,
    rows: Vec<Vec<RawEntry>>,
) -> QMatrix<M, I, C> {
    // Reserve the *coalesced* count, not the emitted one. Rows arrive
    // sorted, so the number of surviving entries is the number of distinct
    // `(col, cindex)` runs; an operator that emits many contributions per
    // key would otherwise leave the returned `QMatrix` holding the raw
    // count as dead capacity for its whole lifetime.
    let total_nnz: usize = rows.iter().map(|row| distinct_keys(row)).sum();
    let mut indptr = Vec::with_capacity(dim + 1);
    let mut data: Vec<Entry<M, I, C>> = Vec::with_capacity(total_nnz);
    indptr.push(I::from_usize(0));
    for row in rows {
        let row_start = data.len();
        for e in row {
            // Rows arrive sorted, so duplicates are adjacent and comparing
            // against the entry just written is enough. `row_start` stops a
            // merge from reaching back into the previous row.
            let col = I::from_usize(e.col);
            let cindex = C::from_usize(e.cindex as usize);
            let merged = if data.len() > row_start {
                let last = data.last_mut().expect("non-empty by the length check");
                let same = last.col == col && last.cindex == cindex;
                if same {
                    last.value = M::from_complex(last.value.to_complex() + e.amp);
                }
                same
            } else {
                false
            };
            if !merged {
                data.push(Entry::new(M::from_complex(e.amp), col, cindex));
            }
        }
        indptr.push(I::from_usize(data.len()));
    }
    QMatrix::from_csr(indptr, data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(col: usize, cindex: u32, re: f64) -> RawEntry {
        RawEntry {
            col,
            cindex,
            amp: Complex::new(re, 0.0),
        }
    }

    /// Regression: coalescing must narrow to `M` at *every* addition, not
    /// sum in `Complex<f64>` and narrow once.
    ///
    /// `16_777_216 + 1` is not representable in `f32`, so per-addition
    /// narrowing rounds it back to `16_777_216` and the third term cancels
    /// to exactly zero. Deferred narrowing would keep the `1` and produce
    /// `1.0` instead. The pre-refactor implementation did the former, so
    /// this pins the stored value to it.
    #[test]
    fn coalescing_narrows_at_every_addition() {
        let row = vec![
            entry(0, 0, 16_777_216.0),
            entry(0, 0, 1.0),
            entry(0, 0, -16_777_216.0),
        ];
        let m: QMatrix<f32, i64, u8> = raw_rows_to_qmatrix(1, vec![row]);
        assert_eq!(m.row(0).len(), 1, "duplicates must coalesce to one entry");
        assert_eq!(
            m.row(0)[0].value,
            0.0f32,
            "f32 must round 16_777_217 back down at each step"
        );
    }

    /// The same inputs in `f64`, where every intermediate *is* representable,
    /// must keep the `1`. Guards against "fixing" the test above by
    /// truncating somewhere it does not belong.
    #[test]
    fn coalescing_keeps_precision_when_the_type_allows() {
        let row = vec![
            entry(0, 0, 16_777_216.0),
            entry(0, 0, 1.0),
            entry(0, 0, -16_777_216.0),
        ];
        let m: QMatrix<f64, i64, u8> = raw_rows_to_qmatrix(1, vec![row]);
        assert_eq!(m.row(0)[0].value, 1.0f64);
    }

    /// Only entries sharing *both* col and cindex may merge.
    #[test]
    fn coalescing_distinguishes_col_and_cindex() {
        let row = vec![
            entry(0, 0, 1.0),
            entry(0, 1, 2.0),
            entry(1, 0, 4.0),
            entry(1, 0, 8.0),
        ];
        let m: QMatrix<f64, i64, u8> = raw_rows_to_qmatrix(1, vec![row]);
        assert_eq!(m.row(0).len(), 3);
        assert_eq!(m.row(0)[0].value, 1.0);
        assert_eq!(m.row(0)[1].value, 2.0);
        assert_eq!(m.row(0)[2].value, 12.0);
    }

    /// A merge must never reach back into the previous row, even when the
    /// last entry of one row and the first of the next share a key.
    #[test]
    fn coalescing_does_not_cross_row_boundaries() {
        let rows = vec![vec![entry(3, 0, 1.0)], vec![entry(3, 0, 2.0)]];
        let m: QMatrix<f64, i64, u8> = raw_rows_to_qmatrix(2, rows);
        assert_eq!(m.row(0).len(), 1);
        assert_eq!(m.row(1).len(), 1);
        assert_eq!(m.row(0)[0].value, 1.0);
        assert_eq!(m.row(1)[0].value, 2.0);
    }

    /// Empty rows must still advance `indptr` so row lookups stay aligned.
    #[test]
    fn empty_rows_keep_indptr_aligned() {
        let rows = vec![vec![], vec![entry(0, 0, 5.0)], vec![]];
        let m: QMatrix<f64, i64, u8> = raw_rows_to_qmatrix(3, rows);
        assert_eq!(m.row(0).len(), 0);
        assert_eq!(m.row(1).len(), 1);
        assert_eq!(m.row(2).len(), 0);
        assert_eq!(m.row(1)[0].value, 5.0);
    }

    /// `distinct_keys` must predict exactly what coalescing produces,
    /// otherwise the reserved capacity is either short (reallocation) or
    /// long (dead memory retained in the returned matrix).
    #[test]
    fn reserved_capacity_matches_coalesced_length() {
        // Ten emitted contributions collapsing to three surviving entries.
        let row = vec![
            entry(0, 0, 1.0),
            entry(0, 0, 1.0),
            entry(0, 0, 1.0),
            entry(0, 1, 1.0),
            entry(0, 1, 1.0),
            entry(5, 0, 1.0),
            entry(5, 0, 1.0),
            entry(5, 0, 1.0),
            entry(5, 0, 1.0),
            entry(5, 0, 1.0),
        ];
        assert_eq!(distinct_keys(&row), 3);

        let m: QMatrix<f64, i64, u8> = raw_rows_to_qmatrix(1, vec![row]);
        assert_eq!(
            m.row(0).len(),
            3,
            "coalesced length must match the estimate"
        );
        assert_eq!(m.row(0)[0].value, 3.0);
        assert_eq!(m.row(0)[1].value, 2.0);
        assert_eq!(m.row(0)[2].value, 5.0);
    }

    #[test]
    fn distinct_keys_handles_edges() {
        assert_eq!(distinct_keys(&[]), 0);
        assert_eq!(distinct_keys(&[entry(0, 0, 1.0)]), 1);
        assert_eq!(distinct_keys(&[entry(0, 0, 1.0), entry(0, 0, 2.0)]), 1);
        assert_eq!(distinct_keys(&[entry(0, 0, 1.0), entry(0, 1, 2.0)]), 2);
        assert_eq!(distinct_keys(&[entry(0, 0, 1.0), entry(1, 0, 2.0)]), 2);
    }

    /// `sort_row` must be stable: equal keys keep emission order, which is
    /// what fixes the summation order during coalescing.
    #[test]
    fn sort_row_is_stable_on_equal_keys() {
        let mut row = vec![
            entry(1, 0, 1.0),
            entry(0, 0, 2.0),
            entry(1, 0, 3.0),
            entry(0, 0, 4.0),
        ];
        sort_row(&mut row);
        let order: Vec<f64> = row.iter().map(|e| e.amp.re).collect();
        assert_eq!(order, vec![2.0, 4.0, 1.0, 3.0]);
    }
}
