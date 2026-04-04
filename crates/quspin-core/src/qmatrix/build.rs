use super::matrix::PARALLEL_DIM_THRESHOLD;
use super::{CIndex, Entry, Index, QMatrix};
use crate::basis::dispatch::SpaceInner;
use crate::basis::{
    BasisSpace,
    sym::{NormInt, SymBasis},
};
use crate::bitbasis::{BitInt, BitStateOp, GenLocalOp};
use crate::operator::Operator;
use crate::primitive::Primitive;
use num_complex::Complex;
use rayon::prelude::*;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// Build from non-symmetric basis (FullSpace or Subspace)
// ---------------------------------------------------------------------------

/// Construct a `QMatrix` from a `Hamiltonian` and a non-symmetric basis.
///
/// For each row (basis state), applies the Hamiltonian and looks up the
/// resulting state in the basis.  Contributions to the same (col, cindex) pair
/// are merged by summing.
///
/// Mirrors `qmatrix::calculate_row` for `space` / `subspace` variants.
///
/// # Type parameters
/// - `H` — Operator type implementing `Operator<C>`
/// - `B` — basis integer type
/// - `M` — matrix element type
/// - `I` — CSR index type
/// - `C` — operator-string index type
pub fn build_from_basis<H, B, M, I, C, S>(ham: &H, basis: &S) -> QMatrix<M, I, C>
where
    H: Operator<C> + Sync,
    B: BitInt,
    M: Primitive,
    I: Index,
    C: CIndex + Copy + Ord,
    S: BasisSpace<B> + Sync,
{
    let dim = basis.size();

    if dim >= PARALLEL_DIM_THRESHOLD {
        let rows: Vec<Vec<Entry<M, I, C>>> = (0..dim)
            .into_par_iter()
            .map(|row_idx| {
                let state = basis.state_at(row_idx);
                let mut entries: Vec<Entry<M, I, C>> = Vec::new();
                ham.apply(state, |cindex, amp, new_state| {
                    let Some(col_idx) = basis.index(new_state) else {
                        return;
                    };
                    let col = I::from_usize(col_idx);
                    let value = M::from_complex(amp);
                    let existing = entries
                        .iter_mut()
                        .find(|e| e.col == col && e.cindex == cindex);
                    if let Some(e) = existing {
                        e.value = M::from_complex(e.value.to_complex() + amp);
                    } else {
                        entries.push(Entry::new(value, col, cindex));
                    }
                });
                entries.sort_unstable_by(|a, b| {
                    a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex))
                });
                entries
            })
            .collect();

        let total_nnz: usize = rows.iter().map(|r| r.len()).sum();
        let mut indptr = Vec::with_capacity(dim + 1);
        let mut data = Vec::with_capacity(total_nnz);
        indptr.push(I::from_usize(0));
        for row in rows {
            data.extend_from_slice(&row);
            indptr.push(I::from_usize(data.len()));
        }
        QMatrix::from_csr(indptr, data)
    } else {
        let mut indptr = Vec::with_capacity(dim + 1);
        let mut data: Vec<Entry<M, I, C>> = Vec::new();
        indptr.push(I::from_usize(0));

        for row_idx in 0..dim {
            let state = basis.state_at(row_idx);
            let row_start = data.len();
            ham.apply(state, |cindex, amp, new_state| {
                let Some(col_idx) = basis.index(new_state) else {
                    return;
                };
                let col = I::from_usize(col_idx);
                let value = M::from_complex(amp);
                let existing = data[row_start..]
                    .iter_mut()
                    .find(|e| e.col == col && e.cindex == cindex);
                if let Some(e) = existing {
                    e.value = M::from_complex(e.value.to_complex() + amp);
                } else {
                    data.push(Entry::new(value, col, cindex));
                }
            });
            data[row_start..]
                .sort_unstable_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));
            indptr.push(I::from_usize(data.len()));
        }
        QMatrix::from_csr(indptr, data)
    }
}

// ---------------------------------------------------------------------------
// Build from symmetric basis
// ---------------------------------------------------------------------------

/// Construct a `QMatrix` from a `Hamiltonian` and a `SymBasis`.
///
/// For each row, the Hamiltonian is applied to the representative state.
/// Each resulting state is mapped to its representative via `check_refstate`,
/// and the matrix element is scaled by the group character and norm ratio.
///
/// Mirrors `qmatrix::calculate_row` for `symmetric_subspace`.
pub fn build_from_symmetric<H, B, L, N, M, I, C>(
    ham: &H,
    basis: &SymBasis<B, L, N>,
) -> QMatrix<M, I, C>
where
    H: Operator<C> + Sync,
    B: BitInt,
    L: BitStateOp<B> + Sync,
    N: NormInt,
    M: Primitive,
    I: Index,
    C: CIndex + Copy + Ord,
{
    let dim = basis.size();
    const ROW_CAP: usize = 64;

    if dim >= PARALLEL_DIM_THRESHOLD {
        let rows: Vec<Vec<Entry<M, I, C>>> = (0..dim)
            .into_par_iter()
            .map(|row_idx| {
                let (state, norm) = basis.entry(row_idx);
                let mut row_buf: SmallVec<[(C, Complex<f64>); ROW_CAP]> = SmallVec::new();
                let mut new_states: SmallVec<[B; ROW_CAP]> = SmallVec::new();
                let mut ref_out: SmallVec<[(B, Complex<f64>); ROW_CAP]> = SmallVec::new();
                let mut entries: Vec<Entry<M, I, C>> = Vec::new();

                ham.apply(state, |cindex, amp, new_state| {
                    row_buf.push((cindex, amp));
                    new_states.push(new_state);
                });

                if !new_states.is_empty() {
                    ref_out.resize(new_states.len(), (new_states[0], Complex::new(1.0, 0.0)));
                    basis.get_refstate_batch(&new_states, &mut ref_out);

                    for ((cindex, amp), (ref_state, grp_char)) in row_buf.iter().zip(ref_out.iter())
                    {
                        let Some(col_idx) = basis.index(*ref_state) else {
                            continue;
                        };
                        let (_, new_norm) = basis.entry(col_idx);
                        let scale = grp_char * (new_norm / norm).sqrt();
                        let full_amp = amp * scale;
                        let col = I::from_usize(col_idx);
                        let value = M::from_complex(full_amp);
                        let existing = entries
                            .iter_mut()
                            .find(|e| e.col == col && e.cindex == *cindex);
                        if let Some(e) = existing {
                            e.value = M::from_complex(e.value.to_complex() + full_amp);
                        } else {
                            entries.push(Entry::new(value, col, *cindex));
                        }
                    }
                }
                entries.sort_unstable_by(|a, b| {
                    a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex))
                });
                entries
            })
            .collect();

        let total_nnz: usize = rows.iter().map(|r| r.len()).sum();
        let mut indptr = Vec::with_capacity(dim + 1);
        let mut data = Vec::with_capacity(total_nnz);
        indptr.push(I::from_usize(0));
        for row in rows {
            data.extend_from_slice(&row);
            indptr.push(I::from_usize(data.len()));
        }
        QMatrix::from_csr(indptr, data)
    } else {
        let mut indptr = Vec::with_capacity(dim + 1);
        let mut data: Vec<Entry<M, I, C>> = Vec::new();
        indptr.push(I::from_usize(0));

        let mut row_buf: SmallVec<[(C, Complex<f64>); ROW_CAP]> = SmallVec::new();
        let mut new_states: SmallVec<[B; ROW_CAP]> = SmallVec::new();
        let mut ref_out: SmallVec<[(B, Complex<f64>); ROW_CAP]> = SmallVec::new();

        for row_idx in 0..dim {
            let (state, norm) = basis.entry(row_idx);
            let row_start = data.len();

            row_buf.clear();
            new_states.clear();
            ham.apply(state, |cindex, amp, new_state| {
                row_buf.push((cindex, amp));
                new_states.push(new_state);
            });

            if !new_states.is_empty() {
                ref_out.resize(new_states.len(), (new_states[0], Complex::new(1.0, 0.0)));
                basis.get_refstate_batch(&new_states, &mut ref_out);

                for ((cindex, amp), (ref_state, grp_char)) in row_buf.iter().zip(ref_out.iter()) {
                    let Some(col_idx) = basis.index(*ref_state) else {
                        continue;
                    };
                    let (_, new_norm) = basis.entry(col_idx);
                    let scale = grp_char * (new_norm / norm).sqrt();
                    let full_amp = amp * scale;
                    let col = I::from_usize(col_idx);
                    let value = M::from_complex(full_amp);
                    let existing = data[row_start..]
                        .iter_mut()
                        .find(|e| e.col == col && e.cindex == *cindex);
                    if let Some(e) = existing {
                        e.value = M::from_complex(e.value.to_complex() + full_amp);
                    } else {
                        data.push(Entry::new(value, col, *cindex));
                    }
                }
            }

            data[row_start..]
                .sort_unstable_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));
            indptr.push(I::from_usize(data.len()));
        }
        QMatrix::from_csr(indptr, data)
    }
}

// ---------------------------------------------------------------------------
// SpaceInner dispatch helpers
// ---------------------------------------------------------------------------

/// Build a `QMatrix<M, i64, C>` from an `Operator<C>` and a `SpaceInner`.
///
/// Dispatches over all 29 `SpaceInner` variants, calling
/// `build_from_basis` for `Full*`/`Sub*` and `build_from_symmetric` for
/// `Sym*`/`DitSym*`.
///
/// This function is used by the Python FFI layer to avoid duplicating the
/// 29-arm match for every `(M, C)` combination.
pub fn build_from_space<H, M, C>(ham: &H, space: &SpaceInner) -> QMatrix<M, i64, C>
where
    H: crate::operator::Operator<C> + Sync,
    M: crate::primitive::Primitive,
    C: CIndex + Copy + Ord,
{
    use crate::basis::dispatch::SpaceInner;
    use crate::bitbasis::{DynamicPermDitValues, PermDitMask};

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

    match space {
        // Non-symmetric
        SpaceInner::Full32(b) => build_from_basis::<H, u32, M, i64, C, _>(ham, b),
        SpaceInner::Full64(b) => build_from_basis::<H, u64, M, i64, C, _>(ham, b),
        SpaceInner::Sub32(b) => build_from_basis::<H, u32, M, i64, C, _>(ham, b),
        SpaceInner::Sub64(b) => build_from_basis::<H, u64, M, i64, C, _>(ham, b),
        SpaceInner::Sub128(b) => build_from_basis::<H, B128, M, i64, C, _>(ham, b),
        SpaceInner::Sub256(b) => build_from_basis::<H, B256, M, i64, C, _>(ham, b),
        #[cfg(feature = "large-int")]
        SpaceInner::Sub512(b) => build_from_basis::<H, B512, M, i64, C, _>(ham, b),
        #[cfg(feature = "large-int")]
        SpaceInner::Sub1024(b) => build_from_basis::<H, B1024, M, i64, C, _>(ham, b),
        #[cfg(feature = "large-int")]
        SpaceInner::Sub2048(b) => build_from_basis::<H, B2048, M, i64, C, _>(ham, b),
        #[cfg(feature = "large-int")]
        SpaceInner::Sub4096(b) => build_from_basis::<H, B4096, M, i64, C, _>(ham, b),
        #[cfg(feature = "large-int")]
        SpaceInner::Sub8192(b) => build_from_basis::<H, B8192, M, i64, C, _>(ham, b),
        // LHSS=2 symmetric
        SpaceInner::Sym32(b) => {
            build_from_symmetric::<H, u32, PermDitMask<u32>, u8, M, i64, C>(ham, b)
        }
        SpaceInner::Sym64(b) => {
            build_from_symmetric::<H, u64, PermDitMask<u64>, u16, M, i64, C>(ham, b)
        }
        SpaceInner::Sym128(b) => {
            build_from_symmetric::<H, B128, PermDitMask<B128>, u32, M, i64, C>(ham, b)
        }
        SpaceInner::Sym256(b) => {
            build_from_symmetric::<H, B256, PermDitMask<B256>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::Sym512(b) => {
            build_from_symmetric::<H, B512, PermDitMask<B512>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::Sym1024(b) => {
            build_from_symmetric::<H, B1024, PermDitMask<B1024>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::Sym2048(b) => {
            build_from_symmetric::<H, B2048, PermDitMask<B2048>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::Sym4096(b) => {
            build_from_symmetric::<H, B4096, PermDitMask<B4096>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::Sym8192(b) => {
            build_from_symmetric::<H, B8192, PermDitMask<B8192>, u32, M, i64, C>(ham, b)
        }
        // LHSS≥3 symmetric (dit)
        SpaceInner::DitSym32(b) => {
            build_from_symmetric::<H, u32, DynamicPermDitValues, u8, M, i64, C>(ham, b)
        }
        SpaceInner::DitSym64(b) => {
            build_from_symmetric::<H, u64, DynamicPermDitValues, u16, M, i64, C>(ham, b)
        }
        SpaceInner::DitSym128(b) => {
            build_from_symmetric::<H, B128, DynamicPermDitValues, u32, M, i64, C>(ham, b)
        }
        SpaceInner::DitSym256(b) => {
            build_from_symmetric::<H, B256, DynamicPermDitValues, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::DitSym512(b) => {
            build_from_symmetric::<H, B512, DynamicPermDitValues, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::DitSym1024(b) => {
            build_from_symmetric::<H, B1024, DynamicPermDitValues, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::DitSym2048(b) => {
            build_from_symmetric::<H, B2048, DynamicPermDitValues, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::DitSym4096(b) => {
            build_from_symmetric::<H, B4096, DynamicPermDitValues, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::DitSym8192(b) => {
            build_from_symmetric::<H, B8192, DynamicPermDitValues, u32, M, i64, C>(ham, b)
        }
        // Any-LHSS generic symmetric
        SpaceInner::GenSym32(b) => {
            build_from_symmetric::<H, u32, GenLocalOp<u32>, u8, M, i64, C>(ham, b)
        }
        SpaceInner::GenSym64(b) => {
            build_from_symmetric::<H, u64, GenLocalOp<u64>, u16, M, i64, C>(ham, b)
        }
        SpaceInner::GenSym128(b) => {
            build_from_symmetric::<H, B128, GenLocalOp<B128>, u32, M, i64, C>(ham, b)
        }
        SpaceInner::GenSym256(b) => {
            build_from_symmetric::<H, B256, GenLocalOp<B256>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::GenSym512(b) => {
            build_from_symmetric::<H, B512, GenLocalOp<B512>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::GenSym1024(b) => {
            build_from_symmetric::<H, B1024, GenLocalOp<B1024>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::GenSym2048(b) => {
            build_from_symmetric::<H, B2048, GenLocalOp<B2048>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::GenSym4096(b) => {
            build_from_symmetric::<H, B4096, GenLocalOp<B4096>, u32, M, i64, C>(ham, b)
        }
        #[cfg(feature = "large-int")]
        SpaceInner::GenSym8192(b) => {
            build_from_symmetric::<H, B8192, GenLocalOp<B8192>, u32, M, i64, C>(ham, b)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::space::{FullSpace, Subspace};
    use crate::operator::pauli::HardcoreOperator;
    use num_complex::Complex;
    use smallvec::smallvec;

    fn xx_ham() -> HardcoreOperator<u8> {
        use crate::operator::pauli::{HardcoreOp, OpEntry};
        // H = Σ_i X_i X_{i+1}, two-site chain
        // Term 0: X_0 X_1, cindex=0, coeff=1
        let ops0 = smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops0)];
        HardcoreOperator::new(terms)
    }

    #[test]
    fn build_fullspace_2site_xx() {
        // 2-site XX: H|00⟩ = |11⟩, H|01⟩ = |10⟩, H|10⟩ = |01⟩, H|11⟩ = |00⟩
        // FullSpace dim=4, states ordered descending: state_at(0)=3,1=2,2=1,3=0
        // H as matrix in this ordering:
        // row 0 (state=3=|11⟩): XX|11⟩ = |00⟩=state_at(3) → col 3
        // row 1 (state=2=|10⟩): XX|10⟩ = |01⟩=state_at(2) → col 2
        // row 2 (state=1=|01⟩): XX|01⟩ = |10⟩=state_at(1) → col 1
        // row 3 (state=0=|00⟩): XX|00⟩ = |11⟩=state_at(0) → col 0
        let ham = xx_ham();
        let basis = FullSpace::<u32>::new(2, 2, false);
        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);

        assert_eq!(mat.dim(), 4);
        assert_eq!(mat.nnz(), 4);

        // Each row has exactly 1 non-zero
        for r in 0..4 {
            assert_eq!(mat.row(r).len(), 1);
        }

        // row 0 → col 3
        assert_eq!(mat.row(0)[0].col, 3i64);
        // row 3 → col 0
        assert_eq!(mat.row(3)[0].col, 0i64);
    }

    #[test]
    fn build_subspace_1particle_2site() {
        // 1-particle sector of 2-site XX: states {|01⟩=1, |10⟩=2}
        // XX connects them: H|01⟩=|10⟩, H|10⟩=|01⟩
        // Subspace sorted ascending: state_at(0)=1, state_at(1)=2
        use crate::operator::pauli::{HardcoreOp, OpEntry};
        let ops = smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops)];
        let ham = HardcoreOperator::new(terms);

        let mut sub = Subspace::<u32>::new(2, 2, false);
        // seed with |01⟩=1
        sub.build(0b01u32, |s| ham.apply_smallvec(s).into_iter());
        assert_eq!(sub.size(), 2);

        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &sub);
        assert_eq!(mat.dim(), 2);
        assert_eq!(mat.nnz(), 2);

        // Matrix should be [[0,1],[1,0]]
        assert!((mat.row(0)[0].value - 1.0).abs() < 1e-12);
        assert_eq!(mat.row(0)[0].col, 1i64);
        assert!((mat.row(1)[0].value - 1.0).abs() < 1e-12);
        assert_eq!(mat.row(1)[0].col, 0i64);
    }

    #[test]
    fn dot_after_build_fullspace() {
        // H|ψ⟩ = |11⟩ = [0,0,0,1] in FullSpace ordering (state_at(3)=0=|00⟩)
        // Wait let's just verify energy with uniform state.
        // For XX on 2-site full space, eigenvalues are ±1 (from XX^2=I).
        // Check: H * H * |ψ⟩ = |ψ⟩ for any |ψ⟩
        let ham = xx_ham();
        let basis = FullSpace::<u32>::new(2, 2, false);
        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
        let coeff = vec![1.0_f64];
        let psi = vec![1.0_f64, 0.0, 0.0, 0.0];
        let mut hpsi = vec![0.0_f64; 4];
        mat.dot(true, &coeff, &psi, &mut hpsi).unwrap();
        let mut h2psi = vec![0.0_f64; 4];
        mat.dot(true, &coeff, &hpsi, &mut h2psi).unwrap();
        // H^2 = I for XX on 2 sites → h2psi == psi
        for i in 0..4 {
            assert!((h2psi[i] - psi[i]).abs() < 1e-12, "h2psi[{i}]={}", h2psi[i]);
        }
    }

    /// Build XX chain Hamiltonian for `n_sites` (periodic boundary).
    fn xx_chain(n_sites: usize) -> HardcoreOperator<u8> {
        use crate::operator::pauli::{HardcoreOp, OpEntry};
        let mut terms = Vec::new();
        for i in 0..n_sites {
            let j = (i + 1) % n_sites;
            let ops = smallvec![(HardcoreOp::X, i as u32), (HardcoreOp::X, j as u32),];
            terms.push(OpEntry::new(0u8, Complex::new(1.0, 0.0), ops));
        }
        HardcoreOperator::new(terms)
    }

    #[test]
    fn build_from_basis_parallel() {
        // 9-site XX chain → FullSpace dim=512, exercises parallel path.
        let ham = xx_chain(9);
        let basis = FullSpace::<u32>::new(2, 9, false);
        assert_eq!(basis.size(), 512);
        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
        assert_eq!(mat.dim(), 512);
        // XX is hermitian: H^2|ψ⟩ should give back a valid result.
        // Check trace(H) = 0 (off-diagonal operator).
        let mut trace = 0.0f64;
        for r in 0..512 {
            for e in mat.row(r) {
                if e.col == r as i64 {
                    trace += e.value;
                }
            }
        }
        assert!(trace.abs() < 1e-12, "trace={trace}");
    }
}
