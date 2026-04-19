use crate::Operator;
use num_complex::Complex;
use quspin_bitbasis::BitInt;
use quspin_bitbasis::manip::{DitManip, DynamicDitManip};
use quspin_types::QuSpinError;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// MonomialTerm
// ---------------------------------------------------------------------------

/// One term in a `MonomialOperator`: a monomial matrix applied to every bond
/// in `bonds`.
///
/// The matrix is stored as parallel arrays:
/// - `perm[i]` — output joint-state index for input joint-state `i`
/// - `amp[i]`  — complex amplitude for input joint-state `i`
///
/// Length of both arrays is `lhss^k` where `k = bond.len()`.
/// Joint-state indexing is row-major over the k dits:
/// `idx = d_0 * lhss^(k-1) + d_1 * lhss^(k-2) + ... + d_{k-1}`.
///
/// No separate `coeff` field — `amp` carries the full amplitude.
pub struct MonomialTerm<C> {
    pub cindex: C,
    /// Output joint-state index per input joint-state.  Length = `lhss^k`.
    pub perm: Vec<usize>,
    /// Complex amplitude per input joint-state.  Length = `lhss^k`.
    pub amp: Vec<Complex<f64>>,
    /// Each element is a k-tuple of site indices.  All bonds must have the
    /// same `k`.
    pub bonds: Vec<SmallVec<[u32; 4]>>,
}

// ---------------------------------------------------------------------------
// MonomialOperator
// ---------------------------------------------------------------------------

/// A Hamiltonian built from monomial matrices (exactly one non-zero per row).
///
/// Each `MonomialTerm<C>` specifies a monomial matrix over the joint local
/// space of a k-site bond, applied to every bond in the term's bond list.
/// All terms share the same `lhss`.
///
/// Compared to `BondOperator` (dense matrix, 2-site only), `MonomialOperator`:
/// - is **sparse**: `lhss^k` non-zeros vs `lhss^{2k}` for a dense matrix
/// - supports **k-site bonds** for arbitrary k
/// - emits **exactly one output per bond** (no branching)
#[derive(Clone)]
pub struct MonomialOperator<C> {
    terms: Vec<MonomialTerm<C>>,
    lhss: usize,
    max_site: usize,
    num_cindices: usize,
    // No cached DynamicDitManip — constructed inline in apply_dynamic,
    // matching the BondOperator pattern.
}

impl<C: Clone> Clone for MonomialTerm<C> {
    fn clone(&self) -> Self {
        MonomialTerm {
            cindex: self.cindex.clone(),
            perm: self.perm.clone(),
            amp: self.amp.clone(),
            bonds: self.bonds.clone(),
        }
    }
}

impl<C: Copy + Ord> MonomialOperator<C> {
    /// Construct from a list of `MonomialTerm` entries and an explicit `lhss`.
    ///
    /// # Errors
    /// Returns `QuSpinError::ValueError` if:
    /// - `terms` is empty
    /// - `lhss` is not in `2..=255`
    /// - for any term: `len(perm) != len(amp)` or `len(perm) != lhss^k`
    /// - for any term: bonds within the term have inconsistent `k`
    /// - for any term: any `perm[i]` is not in `0..lhss^k`
    pub fn new(terms: Vec<MonomialTerm<C>>, lhss: usize) -> Result<Self, QuSpinError> {
        if terms.is_empty() {
            return Err(QuSpinError::ValueError(
                "MonomialOperator: terms must not be empty".to_string(),
            ));
        }
        if !(2..=255).contains(&lhss) {
            return Err(QuSpinError::ValueError(format!(
                "MonomialOperator: lhss={lhss} is not in 2..=255"
            )));
        }

        for (i, term) in terms.iter().enumerate() {
            if term.perm.len() != term.amp.len() {
                return Err(QuSpinError::ValueError(format!(
                    "term {i}: perm.len()={} != amp.len()={}",
                    term.perm.len(),
                    term.amp.len()
                )));
            }
            if term.bonds.is_empty() {
                continue;
            }
            let k = term.bonds[0].len();
            let expected_dim = lhss.pow(k as u32);
            if term.perm.len() != expected_dim {
                return Err(QuSpinError::ValueError(format!(
                    "term {i}: expected perm/amp length lhss^k = {lhss}^{k} = {expected_dim}, \
                     got {}",
                    term.perm.len()
                )));
            }
            for (bi, bond) in term.bonds.iter().enumerate() {
                if bond.len() != k {
                    return Err(QuSpinError::ValueError(format!(
                        "term {i}, bond {bi}: expected k={k} sites, got {}",
                        bond.len()
                    )));
                }
            }
            for (pi, &out_idx) in term.perm.iter().enumerate() {
                if out_idx >= expected_dim {
                    return Err(QuSpinError::ValueError(format!(
                        "term {i}: perm[{pi}]={out_idx} is out of range 0..{expected_dim}"
                    )));
                }
            }
        }

        let max_site = terms
            .iter()
            .flat_map(|t| t.bonds.iter())
            .flat_map(|bond| bond.iter())
            .map(|&s| s as usize)
            .max()
            .unwrap_or(0);

        let num_cindices = {
            let mut sorted: Vec<C> = terms.iter().map(|t| t.cindex).collect();
            sorted.sort();
            sorted.dedup();
            sorted.len()
        };

        Ok(MonomialOperator {
            terms,
            lhss,
            max_site,
            num_cindices,
        })
    }

    pub fn lhss(&self) -> usize {
        self.lhss
    }

    pub fn max_site(&self) -> usize {
        self.max_site
    }

    pub fn num_cindices(&self) -> usize {
        self.num_cindices
    }

    pub fn terms(&self) -> &[MonomialTerm<C>] {
        &self.terms
    }
}

// ---------------------------------------------------------------------------
// Operator<C> impl
// ---------------------------------------------------------------------------

impl<C: Copy + Ord> Operator<C> for MonomialOperator<C> {
    fn max_site(&self) -> usize {
        self.max_site
    }

    fn num_cindices(&self) -> usize {
        self.num_cindices
    }

    fn lhss(&self) -> usize {
        self.lhss
    }

    #[inline]
    fn apply<B: BitInt, F>(&self, state: B, mut emit: F)
    where
        F: FnMut(C, Complex<f64>, B),
    {
        match self.lhss {
            2 => apply_static::<B, C, 2, F>(&self.terms, state, &mut emit),
            3 => apply_static::<B, C, 3, F>(&self.terms, state, &mut emit),
            4 => apply_static::<B, C, 4, F>(&self.terms, state, &mut emit),
            l => apply_dynamic::<B, C, F>(&self.terms, l, state, &mut emit),
        }
    }
}

// ---------------------------------------------------------------------------
// apply_static — compile-time LHSS
// ---------------------------------------------------------------------------

#[inline]
fn apply_static<B: BitInt, C: Copy, const LHSS: usize, F>(
    terms: &[MonomialTerm<C>],
    state: B,
    emit: &mut F,
) where
    F: FnMut(C, Complex<f64>, B),
{
    for term in terms {
        for bond in &term.bonds {
            // Encode joint input state as row-major index.
            let mut idx = 0usize;
            for &site in bond.iter() {
                let d = DitManip::<LHSS>::get_dit(state, site as usize);
                idx = idx * LHSS + d;
            }

            let amp = term.amp[idx];
            if amp.norm_sqr() == 0.0 {
                continue;
            }

            // Decode output joint state and write dits back.
            let mut out_idx = term.perm[idx];
            let mut new_state = state;
            for &site in bond.iter().rev() {
                let d = out_idx % LHSS;
                out_idx /= LHSS;
                new_state = DitManip::<LHSS>::set_dit(new_state, d, site as usize);
            }

            emit(term.cindex, amp, new_state);
        }
    }
}

// ---------------------------------------------------------------------------
// apply_dynamic — runtime lhss (lhss > 4)
// ---------------------------------------------------------------------------

#[inline]
fn apply_dynamic<B: BitInt, C: Copy, F>(
    terms: &[MonomialTerm<C>],
    lhss: usize,
    state: B,
    emit: &mut F,
) where
    F: FnMut(C, Complex<f64>, B),
{
    let manip = DynamicDitManip::new(lhss);
    for term in terms {
        for bond in &term.bonds {
            let mut idx = 0usize;
            for &site in bond.iter() {
                let d = manip.get_dit(state, site as usize);
                idx = idx * lhss + d;
            }

            let amp = term.amp[idx];
            if amp.norm_sqr() == 0.0 {
                continue;
            }

            let mut out_idx = term.perm[idx];
            let mut new_state = state;
            for &site in bond.iter().rev() {
                let d = out_idx % lhss;
                out_idx /= lhss;
                new_state = manip.set_dit(new_state, d, site as usize);
            }

            emit(term.cindex, amp, new_state);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn make_swap_term(lhss: usize) -> MonomialTerm<u8> {
        // SWAP on 2 sites: (a,b) -> (b,a)
        let dim = lhss * lhss;
        let mut perm = vec![0usize; dim];
        let mut amp = vec![Complex::new(0.0, 0.0); dim];
        for a in 0..lhss {
            for b in 0..lhss {
                let in_idx = a * lhss + b;
                let out_idx = b * lhss + a;
                perm[in_idx] = out_idx;
                amp[in_idx] = Complex::new(1.0, 0.0);
            }
        }
        MonomialTerm {
            cindex: 0u8,
            perm,
            amp,
            bonds: vec![smallvec![0, 1]],
        }
    }

    #[test]
    fn construction_valid() {
        let term = make_swap_term(2);
        assert!(MonomialOperator::new(vec![term], 2).is_ok());
    }

    #[test]
    fn construction_empty_terms() {
        let result = MonomialOperator::<u8>::new(vec![], 2);
        assert!(result.is_err());
    }

    #[test]
    fn construction_bad_lhss() {
        let term = make_swap_term(2);
        assert!(MonomialOperator::new(vec![term], 1).is_err());
    }

    #[test]
    fn construction_wrong_perm_len() {
        let mut term = make_swap_term(2);
        term.perm.push(0); // wrong length
        assert!(MonomialOperator::new(vec![term], 2).is_err());
    }

    #[test]
    fn construction_inconsistent_bond_k() {
        let mut term = make_swap_term(2);
        term.bonds.push(smallvec![0, 1, 2]); // k=3 mixed with k=2
        assert!(MonomialOperator::new(vec![term], 2).is_err());
    }

    #[test]
    fn swap_lhss2_single_emit() {
        // SWAP|01> = |10>, SWAP|10> = |01>, SWAP|00> = |00>, SWAP|11> = |11>
        let term = make_swap_term(2);
        let op = MonomialOperator::new(vec![term], 2).unwrap();

        let manip = quspin_bitbasis::DynamicDitManip::new(2);

        // state = 0b01 (site0=0, site1=1) -> expected 0b10
        let mut s: u32 = 0;
        s = manip.set_dit(s, 0, 0);
        s = manip.set_dit(s, 1, 1);

        let mut results: Vec<(u8, Complex<f64>, u32)> = vec![];
        op.apply(s, |c, amp, ns| results.push((c, amp, ns)));

        assert_eq!(results.len(), 1, "exactly one emit per bond");
        let mut expected: u32 = 0;
        expected = manip.set_dit(expected, 1, 0);
        expected = manip.set_dit(expected, 0, 1);
        assert_eq!(results[0].2, expected);
        assert!((results[0].1 - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn swap_lhss3_static_path() {
        let term = make_swap_term(3);
        let op = MonomialOperator::new(vec![term], 3).unwrap();
        let manip = quspin_bitbasis::DynamicDitManip::new(3);

        // state: site0=1, site1=2 -> expected site0=2, site1=1
        let mut s: u64 = 0;
        s = manip.set_dit(s, 1, 0);
        s = manip.set_dit(s, 2, 1);

        let mut results: Vec<(u8, Complex<f64>, u64)> = vec![];
        op.apply(s, |c, amp, ns| results.push((c, amp, ns)));

        assert_eq!(results.len(), 1);
        let mut expected: u64 = 0;
        expected = manip.set_dit(expected, 2, 0);
        expected = manip.set_dit(expected, 1, 1);
        assert_eq!(results[0].2, expected);
    }

    #[test]
    fn swap_lhss5_dynamic_path() {
        let term = make_swap_term(5);
        let op = MonomialOperator::new(vec![term], 5).unwrap();
        let manip = quspin_bitbasis::DynamicDitManip::new(5);

        let mut s: u64 = 0;
        s = manip.set_dit(s, 1, 0);
        s = manip.set_dit(s, 3, 1);

        let mut results: Vec<(u8, Complex<f64>, u64)> = vec![];
        op.apply(s, |c, amp, ns| results.push((c, amp, ns)));

        assert_eq!(results.len(), 1);
        let mut expected: u64 = 0;
        expected = manip.set_dit(expected, 3, 0);
        expected = manip.set_dit(expected, 1, 1);
        assert_eq!(results[0].2, expected);
    }

    #[test]
    fn zero_amplitude_skipped() {
        // amp[0] = 0 should produce no emit
        let mut term = make_swap_term(2);
        term.amp[0] = Complex::new(0.0, 0.0);
        let op = MonomialOperator::new(vec![term], 2).unwrap();

        let s: u32 = 0; // state 00: idx=0, amp=0 -> skip
        let mut results: Vec<(u8, Complex<f64>, u32)> = vec![];
        op.apply(s, |c, amp, ns| results.push((c, amp, ns)));
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn complex_amplitude() {
        let lhss = 2;
        let dim = lhss * lhss;
        let mut perm = (0..dim).collect::<Vec<_>>();
        let phase = Complex::new(0.0, 1.0); // i
        let amp = vec![phase; dim];
        perm[1] = 2;
        perm[2] = 1;
        let term = MonomialTerm {
            cindex: 0u8,
            perm,
            amp,
            bonds: vec![smallvec![0, 1]],
        };
        let op = MonomialOperator::new(vec![term], lhss).unwrap();

        let manip = quspin_bitbasis::DynamicDitManip::new(2);
        let mut s: u32 = 0;
        s = manip.set_dit(s, 0, 0);
        s = manip.set_dit(s, 1, 1); // idx=1

        let mut results: Vec<(u8, Complex<f64>, u32)> = vec![];
        op.apply(s, |c, a, ns| results.push((c, a, ns)));
        assert_eq!(results.len(), 1);
        assert!((results[0].1 - phase).norm() < 1e-12);
    }
}
