use super::{CIndex, Entry, Index, QMatrix};
use crate::basis::{BasisSpace, SymmetricSubspace};
use crate::operator::PauliHamiltonian;
use crate::primitive::Primitive;
use bitbasis::BitInt;

// ---------------------------------------------------------------------------
// Build from non-symmetric basis (FullSpace or Subspace)
// ---------------------------------------------------------------------------

/// Construct a `QMatrix` from a `PauliHamiltonian` and a non-symmetric basis.
///
/// For each row (basis state), applies the Hamiltonian and looks up the
/// resulting state in the basis.  Contributions to the same (col, cindex) pair
/// are merged by summing.
///
/// Mirrors `qmatrix::calculate_row` for `space` / `subspace` variants.
///
/// # Type parameters
/// - `B` — basis integer type
/// - `V` — matrix element type
/// - `I` — CSR index type
/// - `C` — operator-string index type (must match `PauliHamiltonian<C>`)
pub fn build_from_basis<B, V, I, C, S>(ham: &PauliHamiltonian<C>, basis: &S) -> QMatrix<V, I, C>
where
    B: BitInt,
    V: Primitive,
    I: Index,
    C: CIndex + Copy + Ord,
    S: BasisSpace<B>,
{
    let dim = basis.size();
    let mut indptr = Vec::with_capacity(dim + 1);
    let mut data: Vec<Entry<V, I, C>> = Vec::new();

    indptr.push(I::from_usize(0));

    for row_idx in 0..dim {
        let state = basis.state_at(row_idx);
        let row_start = data.len();

        for (amp, new_state, cindex) in ham.apply(state) {
            let Some(col_idx) = basis.index(new_state) else {
                continue;
            };
            let col = I::from_usize(col_idx);
            let value = V::from_complex(amp);

            // Merge with existing entry at same (col, cindex) if present.
            let existing = data[row_start..]
                .iter_mut()
                .find(|e| e.col == col && e.cindex == cindex);
            if let Some(e) = existing {
                e.value = V::from_complex(e.value.to_complex() + amp);
            } else {
                data.push(Entry::new(value, col, cindex));
            }
        }

        // Sort this row's entries by (col, cindex).
        data[row_start..]
            .sort_unstable_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));

        indptr.push(I::from_usize(data.len()));
    }

    QMatrix::from_csr(indptr, data)
}

// ---------------------------------------------------------------------------
// Build from symmetric basis
// ---------------------------------------------------------------------------

/// Construct a `QMatrix` from a `PauliHamiltonian` and a `SymmetricSubspace`.
///
/// For each row, the Hamiltonian is applied to the representative state.
/// Each resulting state is mapped to its representative via `check_refstate`,
/// and the matrix element is scaled by the group character and norm ratio.
///
/// Mirrors `qmatrix::calculate_row` for `symmetric_subspace`.
pub fn build_from_symmetric<B, V, I, C>(
    ham: &PauliHamiltonian<C>,
    basis: &SymmetricSubspace<B>,
) -> QMatrix<V, I, C>
where
    B: BitInt,
    V: Primitive,
    I: Index,
    C: CIndex + Copy + Ord,
{
    let dim = basis.size();
    let mut indptr = Vec::with_capacity(dim + 1);
    let mut data: Vec<Entry<V, I, C>> = Vec::new();

    indptr.push(I::from_usize(0));

    for row_idx in 0..dim {
        let (state, norm) = basis.entry(row_idx);
        let row_start = data.len();

        for (amp, new_state, cindex) in ham.apply(state) {
            // Map the output state to its representative.
            let (ref_state, grp_char) = basis.get_refstate(new_state);

            let Some(col_idx) = basis.index(ref_state) else {
                continue;
            };

            let (_, new_norm) = basis.entry(col_idx);

            // Scale: amp * grp_char * sqrt(new_norm / norm)
            let scale = grp_char * (new_norm / norm).sqrt();
            let full_amp = amp * scale;
            let col = I::from_usize(col_idx);
            let value = V::from_complex(full_amp);

            let existing = data[row_start..]
                .iter_mut()
                .find(|e| e.col == col && e.cindex == cindex);
            if let Some(e) = existing {
                e.value = V::from_complex(e.value.to_complex() + full_amp);
            } else {
                data.push(Entry::new(value, col, cindex));
            }
        }

        data[row_start..]
            .sort_unstable_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));

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
    use crate::basis::space::{FullSpace, Subspace};
    use num_complex::Complex;
    use smallvec::smallvec;

    fn xx_ham() -> PauliHamiltonian<u8> {
        use crate::operator::{OpEntry, PauliOp};
        // H = Σ_i X_i X_{i+1}, two-site chain
        // Term 0: X_0 X_1, cindex=0, coeff=1
        let ops0 = smallvec![(PauliOp::X, 0u32), (PauliOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops0)];
        PauliHamiltonian::new(terms, 2)
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
        let basis = FullSpace::<u32>::new(4);
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
        use crate::operator::{OpEntry, PauliOp};
        let ops = smallvec![(PauliOp::X, 0u32), (PauliOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops)];
        let ham = PauliHamiltonian::new(terms, 2);

        let mut sub = Subspace::<u32>::new();
        // seed with |01⟩=1
        sub.build(0b01u32, |s| ham.apply(s).into_iter());
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
        let basis = FullSpace::<u32>::new(4);
        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
        let coeff = vec![Complex::new(1.0, 0.0)];
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
}
