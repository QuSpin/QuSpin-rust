use super::matrix::PARALLEL_DIM_THRESHOLD;
use super::{CIndex, Entry, Index, QMatrix};
use num_complex::Complex;
use quspin_basis::dispatch::{
    BitBasis, BitBasisDefault, DitBasis, DynDitBasis, DynDitBasisDefault, GenericBasis, QuatBasis,
    QuatBasisDefault, TritBasis, TritBasisDefault,
};
#[cfg(feature = "large-int")]
use quspin_basis::dispatch::{
    BitBasisLargeInt, DynDitBasisLargeInt, QuatBasisLargeInt, TritBasisLargeInt,
};
use quspin_basis::{
    BasisSpace,
    sym::{NormInt, SymBasis},
};
use quspin_bitbasis::{
    BitInt, DynamicPermDitValues, FermionicBitStateOp, PermDitMask, PermDitValues,
};
use quspin_operator::Operator;
use quspin_types::Primitive;
use rayon::prelude::*;
use smallvec::SmallVec;

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

/// Assemble a `QMatrix` from per-row entry vectors.
fn rows_to_qmatrix<M: Primitive, I: Index, C: CIndex>(
    dim: usize,
    rows: Vec<Vec<Entry<M, I, C>>>,
) -> QMatrix<M, I, C> {
    let total_nnz: usize = rows.iter().map(|r| r.len()).sum();
    let mut indptr = Vec::with_capacity(dim + 1);
    let mut data = Vec::with_capacity(total_nnz);
    indptr.push(I::from_usize(0));
    for row in rows {
        data.extend_from_slice(&row);
        indptr.push(I::from_usize(data.len()));
    }
    QMatrix::from_csr(indptr, data)
}

// ---------------------------------------------------------------------------
// Build from non-symmetric basis (FullSpace or Subspace)
// ---------------------------------------------------------------------------

/// Construct a `QMatrix` from a `Hamiltonian` and a non-symmetric basis.
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

    let build_row = |row_idx: usize| -> Vec<Entry<M, I, C>> {
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
        entries.sort_unstable_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));
        entries
    };

    let rows: Vec<Vec<Entry<M, I, C>>> = if dim >= PARALLEL_DIM_THRESHOLD {
        (0..dim).into_par_iter().map(build_row).collect()
    } else {
        (0..dim).map(build_row).collect()
    };

    rows_to_qmatrix(dim, rows)
}

// ---------------------------------------------------------------------------
// Build from symmetric basis
// ---------------------------------------------------------------------------

/// Construct a `QMatrix` from a `Hamiltonian` and a `SymBasis`.
pub fn build_from_symmetric<H, B, L, N, M, I, C>(
    ham: &H,
    basis: &SymBasis<B, L, N>,
) -> QMatrix<M, I, C>
where
    H: Operator<C> + Sync,
    B: BitInt,
    L: FermionicBitStateOp<B> + Sync,
    N: NormInt,
    M: Primitive,
    I: Index,
    C: CIndex + Copy + Ord,
{
    let dim = basis.size();
    const ROW_CAP: usize = 64;

    let build_row = |row_idx: usize| -> Vec<Entry<M, I, C>> {
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

            for ((cindex, amp), (ref_state, grp_char)) in row_buf.iter().zip(ref_out.iter()) {
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
        entries.sort_unstable_by(|a, b| a.col.cmp(&b.col).then_with(|| a.cindex.cmp(&b.cindex)));
        entries
    };

    let rows: Vec<Vec<Entry<M, I, C>>> = if dim >= PARALLEL_DIM_THRESHOLD {
        (0..dim).into_par_iter().map(build_row).collect()
    } else {
        (0..dim).map(build_row).collect()
    };

    rows_to_qmatrix(dim, rows)
}

// ---------------------------------------------------------------------------
// Type-erased dispatch — split into _bit / _dit / _generic.
// ---------------------------------------------------------------------------

// Each per-family Default/LargeInt enum has Full*/Sub* variants (built
// via build_from_basis, which doesn't care about L) plus Sym* variants
// (built via build_from_symmetric, which needs the family's L type and
// the right N). The macro generates the per-enum match arms for plain
// (non-symmetric) variants.
macro_rules! plain_default_arms {
    ($Enum:ident, $self:expr, $ham:ident, $H:ty, $M:ty, $C:ty) => {
        match $self {
            $Enum::Full32(b) => build_from_basis::<$H, u32, $M, i64, $C, _>($ham, b),
            $Enum::Full64(b) => build_from_basis::<$H, u64, $M, i64, $C, _>($ham, b),
            $Enum::Sub32(b) => build_from_basis::<$H, u32, $M, i64, $C, _>($ham, b),
            $Enum::Sub64(b) => build_from_basis::<$H, u64, $M, i64, $C, _>($ham, b),
            $Enum::Sub128(b) => build_from_basis::<$H, B128, $M, i64, $C, _>($ham, b),
            $Enum::Sub256(b) => build_from_basis::<$H, B256, $M, i64, $C, _>($ham, b),
            _ => unreachable!("only plain (Full*/Sub*) variants reach this branch"),
        }
    };
}

#[cfg(feature = "large-int")]
macro_rules! plain_largeint_arms {
    ($Enum:ident, $self:expr, $ham:ident, $H:ty, $M:ty, $C:ty) => {
        match $self {
            $Enum::Sub512(b) => build_from_basis::<$H, B512, $M, i64, $C, _>($ham, b),
            $Enum::Sub1024(b) => build_from_basis::<$H, B1024, $M, i64, $C, _>($ham, b),
            $Enum::Sub2048(b) => build_from_basis::<$H, B2048, $M, i64, $C, _>($ham, b),
            $Enum::Sub4096(b) => build_from_basis::<$H, B4096, $M, i64, $C, _>($ham, b),
            $Enum::Sub8192(b) => build_from_basis::<$H, B8192, $M, i64, $C, _>($ham, b),
            _ => unreachable!("only plain (Sub*) variants reach this branch"),
        }
    };
}

/// Build a `QMatrix<M, i64, C>` from a `BitBasis` (LHSS = 2). FermionBasis
/// uses this path directly, avoiding monomorphization of the dit families.
pub fn build_from_bit<H, M, C>(ham: &H, space: &BitBasis) -> QMatrix<M, i64, C>
where
    H: Operator<C> + Sync,
    M: Primitive,
    C: CIndex + Copy + Ord,
{
    match space {
        BitBasis::Default(d) => match d {
            BitBasisDefault::Sym32(b) => {
                build_from_symmetric::<H, u32, PermDitMask<u32>, u8, M, i64, C>(ham, b)
            }
            BitBasisDefault::Sym64(b) => {
                build_from_symmetric::<H, u64, PermDitMask<u64>, u16, M, i64, C>(ham, b)
            }
            BitBasisDefault::Sym128(b) => {
                build_from_symmetric::<H, B128, PermDitMask<B128>, u32, M, i64, C>(ham, b)
            }
            BitBasisDefault::Sym256(b) => {
                build_from_symmetric::<H, B256, PermDitMask<B256>, u32, M, i64, C>(ham, b)
            }
            other => plain_default_arms!(BitBasisDefault, other, ham, H, M, C),
        },
        #[cfg(feature = "large-int")]
        BitBasis::LargeInt(d) => match d {
            BitBasisLargeInt::Sym512(b) => {
                build_from_symmetric::<H, B512, PermDitMask<B512>, u32, M, i64, C>(ham, b)
            }
            BitBasisLargeInt::Sym1024(b) => {
                build_from_symmetric::<H, B1024, PermDitMask<B1024>, u32, M, i64, C>(ham, b)
            }
            BitBasisLargeInt::Sym2048(b) => {
                build_from_symmetric::<H, B2048, PermDitMask<B2048>, u32, M, i64, C>(ham, b)
            }
            BitBasisLargeInt::Sym4096(b) => {
                build_from_symmetric::<H, B4096, PermDitMask<B4096>, u32, M, i64, C>(ham, b)
            }
            BitBasisLargeInt::Sym8192(b) => {
                build_from_symmetric::<H, B8192, PermDitMask<B8192>, u32, M, i64, C>(ham, b)
            }
            other => plain_largeint_arms!(BitBasisLargeInt, other, ham, H, M, C),
        },
    }
}

/// Build a `QMatrix<M, i64, C>` from a `DitBasis` (LHSS > 2).
pub fn build_from_dit<H, M, C>(ham: &H, space: &DitBasis) -> QMatrix<M, i64, C>
where
    H: Operator<C> + Sync,
    M: Primitive,
    C: CIndex + Copy + Ord,
{
    match space {
        DitBasis::Trit(TritBasis::Default(d)) => match d {
            TritBasisDefault::Sym32(b) => {
                build_from_symmetric::<H, u32, PermDitValues<3>, u8, M, i64, C>(ham, b)
            }
            TritBasisDefault::Sym64(b) => {
                build_from_symmetric::<H, u64, PermDitValues<3>, u16, M, i64, C>(ham, b)
            }
            TritBasisDefault::Sym128(b) => {
                build_from_symmetric::<H, B128, PermDitValues<3>, u32, M, i64, C>(ham, b)
            }
            TritBasisDefault::Sym256(b) => {
                build_from_symmetric::<H, B256, PermDitValues<3>, u32, M, i64, C>(ham, b)
            }
            other => plain_default_arms!(TritBasisDefault, other, ham, H, M, C),
        },
        #[cfg(feature = "large-int")]
        DitBasis::Trit(TritBasis::LargeInt(d)) => match d {
            TritBasisLargeInt::Sym512(b) => {
                build_from_symmetric::<H, B512, PermDitValues<3>, u32, M, i64, C>(ham, b)
            }
            TritBasisLargeInt::Sym1024(b) => {
                build_from_symmetric::<H, B1024, PermDitValues<3>, u32, M, i64, C>(ham, b)
            }
            TritBasisLargeInt::Sym2048(b) => {
                build_from_symmetric::<H, B2048, PermDitValues<3>, u32, M, i64, C>(ham, b)
            }
            TritBasisLargeInt::Sym4096(b) => {
                build_from_symmetric::<H, B4096, PermDitValues<3>, u32, M, i64, C>(ham, b)
            }
            TritBasisLargeInt::Sym8192(b) => {
                build_from_symmetric::<H, B8192, PermDitValues<3>, u32, M, i64, C>(ham, b)
            }
            other => plain_largeint_arms!(TritBasisLargeInt, other, ham, H, M, C),
        },
        DitBasis::Quat(QuatBasis::Default(d)) => match d {
            QuatBasisDefault::Sym32(b) => {
                build_from_symmetric::<H, u32, PermDitValues<4>, u8, M, i64, C>(ham, b)
            }
            QuatBasisDefault::Sym64(b) => {
                build_from_symmetric::<H, u64, PermDitValues<4>, u16, M, i64, C>(ham, b)
            }
            QuatBasisDefault::Sym128(b) => {
                build_from_symmetric::<H, B128, PermDitValues<4>, u32, M, i64, C>(ham, b)
            }
            QuatBasisDefault::Sym256(b) => {
                build_from_symmetric::<H, B256, PermDitValues<4>, u32, M, i64, C>(ham, b)
            }
            other => plain_default_arms!(QuatBasisDefault, other, ham, H, M, C),
        },
        #[cfg(feature = "large-int")]
        DitBasis::Quat(QuatBasis::LargeInt(d)) => match d {
            QuatBasisLargeInt::Sym512(b) => {
                build_from_symmetric::<H, B512, PermDitValues<4>, u32, M, i64, C>(ham, b)
            }
            QuatBasisLargeInt::Sym1024(b) => {
                build_from_symmetric::<H, B1024, PermDitValues<4>, u32, M, i64, C>(ham, b)
            }
            QuatBasisLargeInt::Sym2048(b) => {
                build_from_symmetric::<H, B2048, PermDitValues<4>, u32, M, i64, C>(ham, b)
            }
            QuatBasisLargeInt::Sym4096(b) => {
                build_from_symmetric::<H, B4096, PermDitValues<4>, u32, M, i64, C>(ham, b)
            }
            QuatBasisLargeInt::Sym8192(b) => {
                build_from_symmetric::<H, B8192, PermDitValues<4>, u32, M, i64, C>(ham, b)
            }
            other => plain_largeint_arms!(QuatBasisLargeInt, other, ham, H, M, C),
        },
        DitBasis::Dyn(DynDitBasis::Default(d)) => match d {
            DynDitBasisDefault::Sym32(b) => {
                build_from_symmetric::<H, u32, DynamicPermDitValues, u8, M, i64, C>(ham, b)
            }
            DynDitBasisDefault::Sym64(b) => {
                build_from_symmetric::<H, u64, DynamicPermDitValues, u16, M, i64, C>(ham, b)
            }
            DynDitBasisDefault::Sym128(b) => {
                build_from_symmetric::<H, B128, DynamicPermDitValues, u32, M, i64, C>(ham, b)
            }
            DynDitBasisDefault::Sym256(b) => {
                build_from_symmetric::<H, B256, DynamicPermDitValues, u32, M, i64, C>(ham, b)
            }
            other => plain_default_arms!(DynDitBasisDefault, other, ham, H, M, C),
        },
        #[cfg(feature = "large-int")]
        DitBasis::Dyn(DynDitBasis::LargeInt(d)) => match d {
            DynDitBasisLargeInt::Sym512(b) => {
                build_from_symmetric::<H, B512, DynamicPermDitValues, u32, M, i64, C>(ham, b)
            }
            DynDitBasisLargeInt::Sym1024(b) => {
                build_from_symmetric::<H, B1024, DynamicPermDitValues, u32, M, i64, C>(ham, b)
            }
            DynDitBasisLargeInt::Sym2048(b) => {
                build_from_symmetric::<H, B2048, DynamicPermDitValues, u32, M, i64, C>(ham, b)
            }
            DynDitBasisLargeInt::Sym4096(b) => {
                build_from_symmetric::<H, B4096, DynamicPermDitValues, u32, M, i64, C>(ham, b)
            }
            DynDitBasisLargeInt::Sym8192(b) => {
                build_from_symmetric::<H, B8192, DynamicPermDitValues, u32, M, i64, C>(ham, b)
            }
            other => plain_largeint_arms!(DynDitBasisLargeInt, other, ham, H, M, C),
        },
    }
}

/// Build a `QMatrix<M, i64, C>` from a `GenericBasis`. Branches into
/// [`build_from_bit`] / [`build_from_dit`] depending on the LHSS family.
pub fn build_from_space<H, M, C>(ham: &H, space: &GenericBasis) -> QMatrix<M, i64, C>
where
    H: Operator<C> + Sync,
    M: Primitive,
    C: CIndex + Copy + Ord,
{
    match space {
        GenericBasis::Bit(b) => build_from_bit(ham, b),
        GenericBasis::Dit(b) => build_from_dit(ham, b),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use quspin_basis::space::{FullSpace, Subspace};
    use quspin_operator::pauli::HardcoreOperator;
    use smallvec::smallvec;

    fn xx_ham() -> HardcoreOperator<u8> {
        use quspin_operator::pauli::{HardcoreOp, OpEntry};
        let ops0 = smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops0)];
        HardcoreOperator::new(terms)
    }

    #[test]
    fn build_fullspace_2site_xx() {
        let ham = xx_ham();
        let basis = FullSpace::<u32>::new(2, 2, false);
        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);

        assert_eq!(mat.dim(), 4);
        assert_eq!(mat.nnz(), 4);

        for r in 0..4 {
            assert_eq!(mat.row(r).len(), 1);
        }

        assert_eq!(mat.row(0)[0].col, 3i64);
        assert_eq!(mat.row(3)[0].col, 0i64);
    }

    #[test]
    fn build_subspace_1particle_2site() {
        use quspin_operator::pauli::{HardcoreOp, OpEntry};
        let ops = smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops)];
        let ham = HardcoreOperator::new(terms);

        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0b01u32, &ham);
        assert_eq!(sub.size(), 2);

        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &sub);
        assert_eq!(mat.dim(), 2);
        assert_eq!(mat.nnz(), 2);

        assert!((mat.row(0)[0].value - 1.0).abs() < 1e-12);
        assert_eq!(mat.row(0)[0].col, 1i64);
        assert!((mat.row(1)[0].value - 1.0).abs() < 1e-12);
        assert_eq!(mat.row(1)[0].col, 0i64);
    }

    #[test]
    fn dot_after_build_fullspace() {
        let ham = xx_ham();
        let basis = FullSpace::<u32>::new(2, 2, false);
        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
        let coeff = vec![1.0_f64];
        let psi = vec![1.0_f64, 0.0, 0.0, 0.0];
        let mut hpsi = vec![0.0_f64; 4];
        mat.dot(true, &coeff, &psi, &mut hpsi).unwrap();
        let mut h2psi = vec![0.0_f64; 4];
        mat.dot(true, &coeff, &hpsi, &mut h2psi).unwrap();
        for i in 0..4 {
            assert!((h2psi[i] - psi[i]).abs() < 1e-12, "h2psi[{i}]={}", h2psi[i]);
        }
    }

    fn xx_chain(n_sites: usize) -> HardcoreOperator<u8> {
        use quspin_operator::pauli::{HardcoreOp, OpEntry};
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
        let ham = xx_chain(9);
        let basis = FullSpace::<u32>::new(2, 9, false);
        assert_eq!(basis.size(), 512);
        let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
        assert_eq!(mat.dim(), 512);
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
