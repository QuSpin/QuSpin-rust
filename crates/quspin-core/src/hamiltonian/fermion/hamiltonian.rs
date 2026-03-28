use crate::bitbasis::BitInt;
use crate::hamiltonian::Hamiltonian;
use num_complex::Complex;
use smallvec::SmallVec;

use super::op::FermionOpEntry;

// ---------------------------------------------------------------------------
// FermionHamiltonian
// ---------------------------------------------------------------------------

/// A collection of fermionic operator strings forming a Hamiltonian.
///
/// Mirrors the structure of `HardcoreHamiltonian<C>`, but each term carries
/// Jordan-Wigner sign accumulation (handled inside `FermionOpEntry::apply`).
///
/// The basis is a `HardcoreBasis` (LHSS=2); orbital labelling: site `2*i` =
/// spin-down orbital `i`, site `2*i+1` = spin-up orbital `i`.
#[derive(Clone, Debug)]
pub struct FermionHamiltonian<C> {
    terms: Vec<FermionOpEntry<C>>,
    /// Maximum site index across all operator strings (inferred from terms).
    max_site: usize,
    /// Number of distinct cindex values.
    num_cindices: usize,
}

impl<C: Copy + Ord> FermionHamiltonian<C> {
    /// Construct from a list of `FermionOpEntry` terms.  Terms are sorted by
    /// `cindex`.  `max_site` is inferred from the largest site index appearing
    /// in any operator.
    pub fn new(mut terms: Vec<FermionOpEntry<C>>) -> Self {
        terms.sort_by_key(|e| e.cindex);
        let num_cindices = {
            let mut count = 0;
            let mut last: Option<C> = None;
            for t in &terms {
                if Some(t.cindex) != last {
                    count += 1;
                    last = Some(t.cindex);
                }
            }
            count
        };
        let max_site = terms
            .iter()
            .flat_map(|t| t.ops.iter())
            .map(|&(_, site)| site as usize)
            .max()
            .unwrap_or(0);
        FermionHamiltonian {
            terms,
            max_site,
            num_cindices,
        }
    }

    pub fn max_site(&self) -> usize {
        self.max_site
    }

    pub fn num_cindices(&self) -> usize {
        self.num_cindices
    }

    pub fn terms(&self) -> &[FermionOpEntry<C>] {
        &self.terms
    }

    /// Apply the Hamiltonian to `state`, returning results as a `SmallVec`.
    ///
    /// Zero-amplitude contributions are filtered out.
    #[inline]
    pub fn apply_smallvec<B: BitInt>(&self, state: B) -> SmallVec<[(Complex<f64>, B, C); 8]> {
        let mut result = SmallVec::new();
        for entry in &self.terms {
            let (amp, new_state) = entry.apply(state);
            if amp != Complex::new(0.0, 0.0) {
                result.push((amp, new_state, entry.cindex));
            }
        }
        result
    }
}

impl<C: Copy + Ord> Hamiltonian<C> for FermionHamiltonian<C> {
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
        for entry in &self.terms {
            let (amp, new_state) = entry.apply(state);
            if amp != Complex::new(0.0, 0.0) {
                emit(entry.cindex, amp, new_state);
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
    use crate::hamiltonian::fermion::op::{FermionOp, FermionOpEntry};
    use num_complex::Complex;
    use smallvec::smallvec;

    fn hopping_ham() -> FermionHamiltonian<u8> {
        // H = c†_0 c_1 + c†_1 c_0  (nearest-neighbour hopping, 2 sites)
        let ops01: SmallVec<[(FermionOp, u32); 4]> =
            smallvec![(FermionOp::Plus, 0), (FermionOp::Minus, 1)];
        let ops10: SmallVec<[(FermionOp, u32); 4]> =
            smallvec![(FermionOp::Plus, 1), (FermionOp::Minus, 0)];
        let terms = vec![
            FermionOpEntry::new(0u8, Complex::new(1.0, 0.0), ops01),
            FermionOpEntry::new(0u8, Complex::new(1.0, 0.0), ops10),
        ];
        FermionHamiltonian::new(terms)
    }

    #[test]
    fn hopping_ham_connects_single_particle_states() {
        let ham = hopping_ham();

        // H |01⟩ = |10⟩  (particle hops from site 0 to site 1)
        let result = ham.apply_smallvec(0b01u32);
        assert_eq!(result.len(), 1);
        let (amp, ns, _) = result[0];
        assert_eq!(ns, 0b10u32);
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);

        // H |10⟩ = |01⟩
        let result2 = ham.apply_smallvec(0b10u32);
        assert_eq!(result2.len(), 1);
        let (amp2, ns2, _) = result2[0];
        assert_eq!(ns2, 0b01u32);
        assert!((amp2 - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn hopping_ham_zero_on_vacuum() {
        // H |00⟩ = 0  (no particle to hop)
        let ham = hopping_ham();
        let result = ham.apply_smallvec(0b00u32);
        assert!(result.is_empty());
    }

    #[test]
    fn hopping_ham_zero_on_full() {
        // H |11⟩ = 0  (both sites occupied, Pauli exclusion prevents hopping)
        let ham = hopping_ham();
        let result = ham.apply_smallvec(0b11u32);
        assert!(result.is_empty());
    }

    #[test]
    fn number_term() {
        // H = n̂_0, cindex=0
        let ops: SmallVec<[(FermionOp, u32); 4]> = smallvec![(FermionOp::N, 0)];
        let terms = vec![FermionOpEntry::new(0u8, Complex::new(1.0, 0.0), ops)];
        let ham = FermionHamiltonian::new(terms);

        // n̂_0 |00⟩ = 0
        let r0 = ham.apply_smallvec(0b00u32);
        assert!(r0.is_empty());

        // n̂_0 |01⟩ = |01⟩
        let r1 = ham.apply_smallvec(0b01u32);
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].1, 0b01u32);
        assert!((r1[0].0 - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }
}
