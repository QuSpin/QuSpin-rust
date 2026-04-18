use crate::Operator;
use ndarray::Array2;
use num_complex::Complex;
use quspin_bitbasis::BitInt;
use quspin_bitbasis::manip::{DitManip, DynamicDitManip};
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// BondTerm
// ---------------------------------------------------------------------------

/// One term in a `BondOperator`: a dense two-site matrix applied to a list
/// of site pairs.
///
/// `matrix` is the (lhss² × lhss²) interaction matrix.
/// `matrix[[output_row, input_col]]` is ⟨out|M|in⟩, where the row/column
/// index encodes the two-site state as `a * lhss + b` (site `si` occupies the
/// more-significant digit).
#[derive(Clone, Debug)]
pub struct BondTerm<C> {
    pub cindex: C,
    /// (lhss²)×(lhss²) interaction matrix; shape `[lhss², lhss²]`.
    pub matrix: Array2<Complex<f64>>,
    /// Site pairs (left, right) to apply the matrix to.
    pub bonds: Vec<(u32, u32)>,
}

// ---------------------------------------------------------------------------
// BondOperator
// ---------------------------------------------------------------------------

/// A Hamiltonian built from dense two-site interaction matrices.
///
/// Each `BondTerm<C>` specifies a single dense (lhss² × lhss²) matrix that is
/// applied to every listed pair of sites.  All terms share the same `lhss`
/// (local Hilbert-space size).
#[derive(Clone, Debug)]
pub struct BondOperator<C> {
    terms: Vec<BondTerm<C>>,
    lhss: usize,
    max_site: usize,
    num_cindices: usize,
}

impl<C: Copy + Ord> BondOperator<C> {
    /// Construct from a list of `BondTerm` entries.
    ///
    /// `lhss` is inferred from the shape of the first term's matrix:
    /// `lhss = sqrt(matrix.nrows())`.
    /// `max_site` is inferred as the largest site index appearing in any bond.
    ///
    /// # Errors
    /// Returns `QuSpinError::ValueError` if:
    /// - `terms` is empty
    /// - the first term's matrix is not square or its dimension is not a perfect square
    /// - `lhss` (inferred) is not in `2..=255`
    /// - any term's `matrix` shape is not `[lhss², lhss²]`
    pub fn new(terms: Vec<BondTerm<C>>) -> Result<Self, QuSpinError> {
        let first = terms
            .first()
            .ok_or_else(|| QuSpinError::ValueError("terms must not be empty".to_string()))?;
        let shape = first.matrix.shape();
        if shape[0] != shape[1] {
            return Err(QuSpinError::ValueError(format!(
                "term 0: matrix must be square, got shape {:?}",
                shape
            )));
        }
        let dim = shape[0];
        let lhss = (dim as f64).sqrt() as usize;
        if lhss * lhss != dim {
            return Err(QuSpinError::ValueError(format!(
                "term 0: matrix dimension {dim} is not a perfect square"
            )));
        }
        if !(2..=255).contains(&lhss) {
            return Err(QuSpinError::ValueError(format!(
                "inferred lhss={lhss} is not in 2..=255"
            )));
        }
        let expected_shape = [dim, dim];
        for (i, term) in terms.iter().enumerate() {
            if term.matrix.shape() != expected_shape {
                return Err(QuSpinError::ValueError(format!(
                    "term {i}: matrix shape {:?} but expected [{dim}, {dim}] (lhss={lhss})",
                    term.matrix.shape(),
                )));
            }
        }
        let max_site = terms
            .iter()
            .flat_map(|t| t.bonds.iter())
            .flat_map(|&(si, sj)| [si as usize, sj as usize])
            .max()
            .unwrap_or(0);
        let num_cindices = {
            let mut count = 0;
            let mut last: Option<C> = None;
            let mut sorted_cindices: Vec<C> = terms.iter().map(|t| t.cindex).collect();
            sorted_cindices.sort();
            for c in sorted_cindices {
                if Some(c) != last {
                    count += 1;
                    last = Some(c);
                }
            }
            count
        };
        Ok(BondOperator {
            terms,
            lhss,
            max_site,
            num_cindices,
        })
    }

    pub fn lhss(&self) -> usize {
        self.lhss
    }

    pub fn terms(&self) -> &[BondTerm<C>] {
        &self.terms
    }
}

// ---------------------------------------------------------------------------
// Operator<C> impl
// ---------------------------------------------------------------------------

impl<C: Copy + Ord> Operator<C> for BondOperator<C> {
    fn max_site(&self) -> usize {
        self.max_site
    }

    fn num_cindices(&self) -> usize {
        self.num_cindices
    }

    #[inline]
    fn apply<B: BitInt, F>(&self, state: B, mut emit: F)
    where
        F: FnMut(C, Complex<f64>, B),
    {
        match self.lhss {
            2 => apply_static::<B, C, 2, _>(&self.terms, state, &mut emit),
            3 => apply_static::<B, C, 3, _>(&self.terms, state, &mut emit),
            4 => apply_static::<B, C, 4, _>(&self.terms, state, &mut emit),
            l => apply_dynamic::<B, C, _>(&self.terms, l, state, &mut emit),
        }
    }
}

// ---------------------------------------------------------------------------
// apply_static — compile-time LHSS, loop bounds are constants
// ---------------------------------------------------------------------------

#[inline]
fn apply_static<B: BitInt, C: Copy, const LHSS: usize, F>(
    terms: &[BondTerm<C>],
    state: B,
    emit: &mut F,
) where
    F: FnMut(C, Complex<f64>, B),
{
    let dim = LHSS * LHSS;
    for term in terms {
        for &(si, sj) in &term.bonds {
            let a = DitManip::<LHSS>::get_dit(state, si as usize);
            let b = DitManip::<LHSS>::get_dit(state, sj as usize);
            let input_col = a * LHSS + b;
            for output_row in 0..dim {
                let amp = term.matrix[[output_row, input_col]];
                if amp.norm_sqr() == 0.0 {
                    continue;
                }
                let c = output_row / LHSS;
                let d = output_row % LHSS;
                let ns = DitManip::<LHSS>::set_dit(
                    DitManip::<LHSS>::set_dit(state, c, si as usize),
                    d,
                    sj as usize,
                );
                emit(term.cindex, amp, ns);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// apply_dynamic — runtime lhss, used for lhss > 4
// ---------------------------------------------------------------------------

#[inline]
fn apply_dynamic<B: BitInt, C: Copy, F>(terms: &[BondTerm<C>], lhss: usize, state: B, emit: &mut F)
where
    F: FnMut(C, Complex<f64>, B),
{
    let manip = DynamicDitManip::new(lhss);
    let dim = lhss * lhss;
    for term in terms {
        for &(si, sj) in &term.bonds {
            let a = manip.get_dit(state, si as usize);
            let b = manip.get_dit(state, sj as usize);
            let input_col = a * lhss + b;
            for output_row in 0..dim {
                let amp = term.matrix[[output_row, input_col]];
                if amp.norm_sqr() == 0.0 {
                    continue;
                }
                let c = output_row / lhss;
                let d = output_row % lhss;
                let ns = manip.set_dit(manip.set_dit(state, c, si as usize), d, sj as usize);
                emit(term.cindex, amp, ns);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
    use ndarray::Array2;
    use num_complex::Complex;
    use smallvec::smallvec;

    fn xx_matrix() -> Array2<Complex<f64>> {
        // XX acts on two qubits: |ab⟩ → |ā b̄⟩ with amplitude 1.
        // States ordered as: 00→0, 01→1, 10→2, 11→3
        // XX|00⟩=|11⟩, XX|01⟩=|10⟩, XX|10⟩=|01⟩, XX|11⟩=|00⟩
        let mut m = Array2::zeros((4, 4));
        m[[3, 0]] = Complex::new(1.0, 0.0); // XX|00⟩ → |11⟩
        m[[2, 1]] = Complex::new(1.0, 0.0); // XX|01⟩ → |10⟩
        m[[1, 2]] = Complex::new(1.0, 0.0); // XX|10⟩ → |01⟩
        m[[0, 3]] = Complex::new(1.0, 0.0); // XX|11⟩ → |00⟩
        m
    }

    fn zz_matrix() -> Array2<Complex<f64>> {
        // ZZ|ab⟩ = (2a-1)(2b-1)|ab⟩  (diagonal, QuSpin pauli=1 convention)
        let mut m = Array2::zeros((4, 4));
        m[[0, 0]] = Complex::new(1.0, 0.0); // |00⟩: (-1)(-1)=+1
        m[[1, 1]] = Complex::new(-1.0, 0.0); // |01⟩: (-1)(+1)=-1
        m[[2, 2]] = Complex::new(-1.0, 0.0); // |10⟩: (+1)(-1)=-1
        m[[3, 3]] = Complex::new(1.0, 0.0); // |11⟩: (+1)(+1)=+1
        m
    }

    fn heisenberg_matrix() -> Array2<Complex<f64>> {
        // H_bond = XX + YY + ZZ on two sites (lhss=2, QuSpin pauli=1 convention).
        // |00⟩ → ZZ=+1|00⟩,  XX+YY cancel (XX→+|11⟩, YY→-|11⟩)
        // |01⟩ → ZZ=-1|01⟩,  XX+YY: XX|01⟩=+|10⟩, YY|01⟩=+|10⟩ → 2|10⟩
        // |10⟩ → ZZ=-1|10⟩,  XX+YY: 2|01⟩
        // |11⟩ → ZZ=+1|11⟩,  XX+YY cancel
        let mut m = Array2::zeros((4, 4));
        m[[0, 0]] = Complex::new(1.0, 0.0); // ⟨00|H|00⟩ = +1
        m[[1, 1]] = Complex::new(-1.0, 0.0); // ⟨01|H|01⟩ = -1
        m[[2, 1]] = Complex::new(2.0, 0.0); // ⟨10|H|01⟩ = +2
        m[[1, 2]] = Complex::new(2.0, 0.0); // ⟨01|H|10⟩ = +2
        m[[2, 2]] = Complex::new(-1.0, 0.0); // ⟨10|H|10⟩ = -1
        m[[3, 3]] = Complex::new(1.0, 0.0); // ⟨11|H|11⟩ = +1
        m
    }

    /// Collect all `(cindex, amp, new_state)` contributions for `state`.
    fn collect_apply<C: Copy + Ord>(
        ham: &BondOperator<C>,
        state: u32,
    ) -> Vec<(C, Complex<f64>, u32)> {
        let mut out = vec![];
        ham.apply(state, |c, amp, ns| out.push((c, amp, ns)));
        out
    }

    fn collect_apply_hardcore<C: Copy + Ord>(
        ham: &HardcoreOperator<C>,
        state: u32,
    ) -> Vec<(C, Complex<f64>, u32)> {
        let mut out = vec![];
        <HardcoreOperator<C> as Operator<C>>::apply(ham, state, |c, amp, ns| {
            out.push((c, amp, ns))
        });
        out
    }

    // ------------------------------------------------------------------
    // Construction / validation
    // ------------------------------------------------------------------

    #[test]
    fn new_valid() {
        let term = BondTerm {
            cindex: 0u8,
            matrix: xx_matrix(),
            bonds: vec![(0, 1)],
        };
        assert!(BondOperator::new(vec![term]).is_ok());
    }

    #[test]
    fn new_wrong_matrix_shape() {
        let term = BondTerm {
            cindex: 0u8,
            matrix: Array2::zeros((3, 3)), // wrong: should be [4, 4] for lhss=2
            bonds: vec![(0, 1)],
        };
        assert!(BondOperator::new(vec![term]).is_err());
    }

    #[test]
    fn new_lhss_out_of_range() {
        let term = BondTerm {
            cindex: 0u8,
            matrix: Array2::zeros((1, 1)), // lhss=1 rejected before shape check
            bonds: vec![],
        };
        assert!(BondOperator::new(vec![term]).is_err());
    }

    // ------------------------------------------------------------------
    // XX bond: compare BondOperator with HardcoreOperator
    // ------------------------------------------------------------------

    #[test]
    fn xx_bond_matches_hardcore_xx() {
        let term = BondTerm {
            cindex: 0u8,
            matrix: xx_matrix(),
            bonds: vec![(0, 1)],
        };
        let bond_ham = BondOperator::new(vec![term]).unwrap();

        let ops: smallvec::SmallVec<[(HardcoreOp, u32); 4]> =
            smallvec![(HardcoreOp::X, 0), (HardcoreOp::X, 1)];
        let hardcore_ham =
            HardcoreOperator::new(vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops)]);

        for state in 0u32..4 {
            let mut bond_out = collect_apply(&bond_ham, state);
            let mut hardcore_out = collect_apply_hardcore(&hardcore_ham, state);
            bond_out.sort_by_key(|e| e.2);
            hardcore_out.sort_by_key(|e| e.2);
            assert_eq!(
                bond_out.len(),
                hardcore_out.len(),
                "state={state:#06b}: output count mismatch"
            );
            for (b, h) in bond_out.iter().zip(hardcore_out.iter()) {
                assert_eq!(b.0, h.0, "state={state:#06b}: cindex mismatch");
                assert_eq!(b.2, h.2, "state={state:#06b}: new_state mismatch");
                assert!(
                    (b.1 - h.1).norm() < 1e-12,
                    "state={state:#06b}: amplitude mismatch: bond={}, hardcore={}",
                    b.1,
                    h.1
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // ZZ bond: compare BondOperator with HardcoreOperator
    // ------------------------------------------------------------------

    #[test]
    fn zz_bond_matches_hardcore_zz() {
        let term = BondTerm {
            cindex: 0u8,
            matrix: zz_matrix(),
            bonds: vec![(0, 1)],
        };
        let bond_ham = BondOperator::new(vec![term]).unwrap();

        let ops: smallvec::SmallVec<[(HardcoreOp, u32); 4]> =
            smallvec![(HardcoreOp::Z, 0), (HardcoreOp::Z, 1)];
        let hardcore_ham =
            HardcoreOperator::new(vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops)]);

        for state in 0u32..4 {
            let mut bond_out = collect_apply(&bond_ham, state);
            let mut hardcore_out = collect_apply_hardcore(&hardcore_ham, state);
            bond_out.sort_by_key(|e| e.2);
            hardcore_out.sort_by_key(|e| e.2);
            assert_eq!(bond_out.len(), hardcore_out.len(), "state={state}");
            for (b, h) in bond_out.iter().zip(hardcore_out.iter()) {
                assert_eq!(b.2, h.2, "state={state}: new_state");
                assert!(
                    (b.1 - h.1).norm() < 1e-12,
                    "state={state}: amp bond={} hardcore={}",
                    b.1,
                    h.1
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // num_cindices
    // ------------------------------------------------------------------

    #[test]
    fn num_cindices_two_terms() {
        let t0 = BondTerm {
            cindex: 0u8,
            matrix: xx_matrix(),
            bonds: vec![(0, 1)],
        };
        let t1 = BondTerm {
            cindex: 1u8,
            matrix: zz_matrix(),
            bonds: vec![(0, 1)],
        };
        let ham = BondOperator::new(vec![t0, t1]).unwrap();
        assert_eq!(ham.num_cindices(), 2);
    }

    // ------------------------------------------------------------------
    // Heisenberg chain: XX + YY + ZZ via BondOperator vs HardcoreOperator
    // ------------------------------------------------------------------

    #[test]
    fn heisenberg_bond_matches_hardcore_4site() {
        let n_sites = 4usize;
        let bonds: Vec<(u32, u32)> = (0..n_sites as u32 - 1).map(|i| (i, i + 1)).collect();

        let term = BondTerm {
            cindex: 0u8,
            matrix: heisenberg_matrix(),
            bonds: bonds.clone(),
        };
        let bond_ham = BondOperator::new(vec![term]).unwrap();

        let mut terms = vec![];
        for &(i, j) in &bonds {
            for ops_vec in [
                smallvec![(HardcoreOp::X, i), (HardcoreOp::X, j)],
                smallvec![(HardcoreOp::Y, i), (HardcoreOp::Y, j)],
                smallvec![(HardcoreOp::Z, i), (HardcoreOp::Z, j)],
            ] {
                terms.push(OpEntry::new(0u8, Complex::new(1.0, 0.0), ops_vec));
            }
        }
        let hardcore_ham = HardcoreOperator::new(terms);

        for state in 0u32..(1 << n_sites) {
            let mut bond_sums: std::collections::HashMap<u32, Complex<f64>> =
                std::collections::HashMap::new();
            bond_ham.apply(state, |_c, amp, ns| {
                *bond_sums.entry(ns).or_insert(Complex::new(0.0, 0.0)) += amp;
            });

            let mut hardcore_sums: std::collections::HashMap<u32, Complex<f64>> =
                std::collections::HashMap::new();
            <HardcoreOperator<u8> as Operator<u8>>::apply(&hardcore_ham, state, |_c, amp, ns| {
                *hardcore_sums.entry(ns).or_insert(Complex::new(0.0, 0.0)) += amp;
            });

            // Filter near-zero entries (XX+YY cancellations leave zero-amp keys).
            bond_sums.retain(|_, v| v.norm() > 1e-12);
            hardcore_sums.retain(|_, v| v.norm() > 1e-12);

            let mut keys: Vec<u32> = bond_sums.keys().cloned().collect();
            keys.sort();
            let mut hc_keys: Vec<u32> = hardcore_sums.keys().cloned().collect();
            hc_keys.sort();
            assert_eq!(keys, hc_keys, "state={state:#06b}: output states differ");
            for &ns in &keys {
                let ba = bond_sums[&ns];
                let ha = hardcore_sums[&ns];
                assert!(
                    (ba - ha).norm() < 1e-12,
                    "state={state:#06b} → ns={ns:#06b}: bond={ba} hardcore={ha}"
                );
            }
        }
    }
}
