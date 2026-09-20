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

/// Narrow a `CIndex` into [`RawEntry`]'s carrier width.
///
/// `CIndex` is sealed to `u8` and `u16` (`impl_cindex!(u8, u16)`), so this
/// is exact today. The assert is the tripwire for a future wider impl:
/// `RawEntry.cindex` round-trips back through `C::from_usize`, so a silent
/// truncation here would reconstruct the wrong operator-string index and
/// mis-assign coefficients in matrix-vector products.
#[inline]
fn narrow_cindex<C: CIndex>(cindex: C) -> u32 {
    let raw = cindex.as_usize();
    debug_assert!(
        raw <= u32::MAX as usize,
        "cindex {raw} exceeds RawEntry's u32 carrier; widen RawEntry.cindex"
    );
    raw as u32
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
    /// duplicates are summed later, in [`append_row`], where the element
    /// type is known. Implementations must push in the operator's
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
                cindex: narrow_cindex(cindex),
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
                cindex: narrow_cindex(*cindex),
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

/// How many rows the erased stage builds before the typed stage drains
/// them. Bounds the uncoalesced intermediate: see [`build_qmatrix`].
const ROW_CHUNK: usize = 4096;

/// Build rows `start..end` through a type-erased source.
///
/// Takes `&dyn RowSource`, so the closure handed to rayon and the slice
/// handed to the sort both have exactly one type no matter how many
/// `(H, B, C, S)` combinations exist upstream. This is where the
/// instantiation count collapses: one rayon copy, one pdqsort copy.
fn build_raw_rows(src: &dyn RowSource, start: usize, end: usize) -> Vec<Vec<RawEntry>> {
    let build_row = |row_idx: usize| -> Vec<RawEntry> {
        let mut row = Vec::new();
        src.row_into(row_idx, &mut row);
        sort_row(&mut row);
        row
    };

    // Threshold on the whole matrix, not the chunk, so chunking does not
    // change which matrices take the parallel path.
    if src.dim() >= PARALLEL_DIM_THRESHOLD {
        (start..end).into_par_iter().map(build_row).collect()
    } else {
        (start..end).map(build_row).collect()
    }
}

/// Append one sorted erased row to `data`, summing duplicate keys.
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
fn append_row<M: Primitive, I: Index, C: CIndex>(data: &mut Vec<Entry<M, I, C>>, row: &[RawEntry]) {
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
}

/// Build a typed `QMatrix` from a type-erased row source.
///
/// Rows are produced in chunks of [`ROW_CHUNK`] and drained into `data`
/// immediately, so the uncoalesced `RawEntry` intermediate never holds more
/// than one chunk. That matters because `row_into` emits *every*
/// contribution while the old implementation deduplicated on insert: a
/// diagonal-heavy operator emits one contribution per term for the same
/// `(col, cindex)`, so the uncoalesced count scales with the term count —
/// measured at 8x/16x/24x the coalesced count for an Ising chain at
/// n = 8/16/24. Materialising the whole matrix uncoalesced would multiply
/// peak memory by that factor on top of `RawEntry` being wider than
/// `Entry`.
///
/// `dim` comes from the source rather than a separate argument, so the row
/// count and the declared dimension cannot disagree.
///
/// Generic over `M`, `I` and `C`, but contains no parallelism and no sort —
/// those stay in [`build_raw_rows`] behind `&dyn RowSource`, so the
/// element-type axes multiply only this small loop.
pub fn build_qmatrix<M: Primitive, I: Index, C: CIndex>(src: &dyn RowSource) -> QMatrix<M, I, C> {
    let dim = src.dim();
    let mut indptr = Vec::with_capacity(dim + 1);
    // One entry per row is the floor for any operator with a non-empty
    // diagonal; growth from here is amortised. Counting exactly would mean
    // a second full pass over every emitted contribution.
    let mut data: Vec<Entry<M, I, C>> = Vec::with_capacity(dim);
    indptr.push(I::from_usize(0));

    let mut start = 0usize;
    while start < dim {
        let end = (start + ROW_CHUNK).min(dim);
        for row in build_raw_rows(src, start, end) {
            append_row(&mut data, &row);
            indptr.push(I::from_usize(data.len()));
        }
        start = end;
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

    /// Serves pre-built rows so the coalescing tests drive the real
    /// `build_qmatrix` path rather than a stand-in for it.
    struct FakeRows(Vec<Vec<RawEntry>>);

    impl RowSource for FakeRows {
        fn dim(&self) -> usize {
            self.0.len()
        }

        fn row_into(&self, row_idx: usize, out: &mut Vec<RawEntry>) {
            out.extend_from_slice(&self.0[row_idx]);
        }
    }

    fn build<M: Primitive>(rows: Vec<Vec<RawEntry>>) -> QMatrix<M, i64, u8> {
        build_qmatrix(&FakeRows(rows))
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
        let m: QMatrix<f32, i64, u8> = build(vec![row]);
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
        let m: QMatrix<f64, i64, u8> = build(vec![row]);
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
        let m: QMatrix<f64, i64, u8> = build(vec![row]);
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
        let m: QMatrix<f64, i64, u8> = build(rows);
        assert_eq!(m.row(0).len(), 1);
        assert_eq!(m.row(1).len(), 1);
        assert_eq!(m.row(0)[0].value, 1.0);
        assert_eq!(m.row(1)[0].value, 2.0);
    }

    /// Empty rows must still advance `indptr` so row lookups stay aligned.
    #[test]
    fn empty_rows_keep_indptr_aligned() {
        let rows = vec![vec![], vec![entry(0, 0, 5.0)], vec![]];
        let m: QMatrix<f64, i64, u8> = build(rows);
        assert_eq!(m.row(0).len(), 0);
        assert_eq!(m.row(1).len(), 1);
        assert_eq!(m.row(2).len(), 0);
        assert_eq!(m.row(1)[0].value, 5.0);
    }

    /// Many contributions on one key must collapse to a single entry.
    #[test]
    fn many_contributions_collapse_to_one_entry() {
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
        let m: QMatrix<f64, i64, u8> = build(vec![row]);
        assert_eq!(
            m.row(0).len(),
            3,
            "ten contributions must collapse to three entries"
        );
        assert_eq!(m.row(0)[0].value, 3.0);
        assert_eq!(m.row(0)[1].value, 2.0);
        assert_eq!(m.row(0)[2].value, 5.0);
    }

    /// Rows must be built in bounded chunks, so the uncoalesced
    /// intermediate never scales with the whole matrix. A diagonal-heavy
    /// operator emits one contribution per term for the same key, so
    /// materialising every row before coalescing would multiply peak memory
    /// by the term count.
    ///
    /// The peak itself is not observable through `RowSource`, so this pins
    /// the two properties that are: the chunk helper never returns more
    /// than one chunk, and driving a matrix several chunks wide still
    /// visits every row exactly once and reproduces every entry.
    #[test]
    fn rows_are_built_in_bounded_chunks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Counting {
            dim: usize,
            calls: AtomicUsize,
        }

        impl RowSource for Counting {
            fn dim(&self) -> usize {
                self.dim
            }

            fn row_into(&self, row_idx: usize, out: &mut Vec<RawEntry>) {
                self.calls.fetch_add(1, Ordering::Relaxed);
                // One entry per row, valued by its index, so a dropped or
                // duplicated row is visible in the result.
                out.push(entry(row_idx, 0, row_idx as f64));
            }
        }

        let dim = ROW_CHUNK * 3 + 7;
        let src = Counting {
            dim,
            calls: AtomicUsize::new(0),
        };

        // The erased stage hands back one chunk at a time, which is what
        // bounds the live `RawEntry` intermediate.
        let chunk = build_raw_rows(&src, 0, ROW_CHUNK.min(dim));
        assert_eq!(chunk.len(), ROW_CHUNK.min(dim));
        src.calls.store(0, Ordering::Relaxed);

        let m: QMatrix<f64, i64, u8> = build_qmatrix(&src);
        assert_eq!(m.dim(), dim, "every row must reach the matrix");
        assert_eq!(
            src.calls.load(Ordering::Relaxed),
            dim,
            "each row must be built exactly once across all chunks"
        );
        // Spot-check rows either side of every chunk boundary.
        for r in [0, 1, ROW_CHUNK - 1, ROW_CHUNK, ROW_CHUNK + 1, dim - 1] {
            assert_eq!(m.row(r).len(), 1, "row {r}");
            assert_eq!(m.row(r)[0].value, r as f64, "row {r} value");
            assert_eq!(m.row(r)[0].col, r as i64, "row {r} col");
        }
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
