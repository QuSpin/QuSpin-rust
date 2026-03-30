use crate::bitbasis::{BitInt, manip::DynamicDitManip};
use crate::hamiltonian::Operator;
use num_complex::Complex;
use smallvec::SmallVec;

use super::op::BosonOpEntry;

// ---------------------------------------------------------------------------
// BosonOperator
// ---------------------------------------------------------------------------

/// A collection of operator strings forming a bosonic Hamiltonian.
///
/// Operators are `+` (a†), `-` (a), and `n` (n̂) acting on sites with
/// `lhss` levels each (truncated harmonic oscillator, `lhss ≥ 2`).
///
/// The `manip` field encodes the LHSS and per-site bit-width used to
/// extract/insert dit values in basis states.
///
/// Terms are stored sorted by `cindex` and a `DynamicDitManip` is stored
/// to avoid re-construction on every `apply` call.
#[derive(Clone, Debug)]
pub struct BosonOperator<C> {
    terms: Vec<BosonOpEntry<C>>,
    manip: DynamicDitManip,
    /// Maximum site index across all operator strings (inferred from terms).
    max_site: usize,
    /// Number of distinct cindex values.
    num_cindices: usize,
}

impl<C: Copy + Ord> BosonOperator<C> {
    /// Construct from a list of `BosonOpEntry` terms, the LHSS, and the number
    /// of distinct cindices.  Terms are sorted by `cindex`.
    pub fn new(mut terms: Vec<BosonOpEntry<C>>, lhss: usize) -> Self {
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
        BosonOperator {
            terms,
            manip: DynamicDitManip::new(lhss),
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

    pub fn lhss(&self) -> usize {
        self.manip.lhss
    }

    pub fn manip(&self) -> &DynamicDitManip {
        &self.manip
    }

    pub fn terms(&self) -> &[BosonOpEntry<C>] {
        &self.terms
    }

    /// Apply the Hamiltonian to `state`.
    ///
    /// Returns a `SmallVec` of `(amplitude, new_state, cindex)` tuples,
    /// skipping zero-amplitude results.  Inline storage for up to 8 results.
    #[inline]
    pub fn apply_smallvec<B: BitInt>(&self, state: B) -> SmallVec<[(Complex<f64>, B, C); 8]> {
        let mut result = SmallVec::new();
        for entry in &self.terms {
            let (amp, new_state) = entry.apply(state, &self.manip);
            if amp != Complex::new(0.0, 0.0) {
                result.push((amp, new_state, entry.cindex));
            }
        }
        result
    }
}

impl<C: Copy + Ord> Operator<C> for BosonOperator<C> {
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
            let (amp, new_state) = entry.apply(state, &self.manip);
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
    use crate::hamiltonian::boson::op::{BosonOp, BosonOpEntry};
    use num_complex::Complex;
    use smallvec::smallvec;

    // --- BosonOp::apply ---

    #[test]
    fn plus_lhss3_raises_occupation() {
        // lhss=3: manip encodes 2 bits per site.
        // state with n=0 at site 0 → n=1, amp = √1 = 1
        let manip = DynamicDitManip::new(3);
        let state: u32 = 0;
        let (ns, amp) = BosonOp::Plus.apply(state, 0, &manip);
        assert_eq!(manip.get_dit(ns, 0), 1);
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn plus_at_lhss_boundary_gives_zero() {
        // lhss=3, n=2 (max) → Plus gives amplitude 0
        let manip = DynamicDitManip::new(3);
        let mut state: u32 = 0;
        state = manip.set_dit(state, 2, 0);
        let (_, amp) = BosonOp::Plus.apply(state, 0, &manip);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn minus_lhss3_lowers_occupation() {
        // n=2 at site 0 → n=1, amp = √2
        let manip = DynamicDitManip::new(3);
        let mut state: u32 = 0;
        state = manip.set_dit(state, 2, 0);
        let (ns, amp) = BosonOp::Minus.apply(state, 0, &manip);
        assert_eq!(manip.get_dit(ns, 0), 1);
        assert!((amp - Complex::new(2.0f64.sqrt(), 0.0)).norm() < 1e-12);
    }

    #[test]
    fn minus_at_zero_gives_zero() {
        // n=0 → Minus gives amplitude 0
        let manip = DynamicDitManip::new(3);
        let state: u32 = 0;
        let (_, amp) = BosonOp::Minus.apply(state, 0, &manip);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn number_op_gives_occupation() {
        let manip = DynamicDitManip::new(4);
        let mut state: u32 = 0;
        state = manip.set_dit(state, 3, 1); // site 1 has n=3
        let (ns, amp) = BosonOp::N.apply(state, 1, &manip);
        assert_eq!(ns, state);
        assert!((amp - Complex::new(3.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn number_op_zero_occupation() {
        let manip = DynamicDitManip::new(4);
        let state: u32 = 0;
        let (ns, amp) = BosonOp::N.apply(state, 0, &manip);
        assert_eq!(ns, state);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    // --- BosonOperator::apply ---

    #[test]
    fn hamiltonian_hop_connects_states() {
        // H = a†_0 a_1 + a_0 a†_1  (hopping, lhss=3, 2 sites)
        // State |1,0⟩: site 0 = 1, site 1 = 0
        // a†_0 a_1 |1,0⟩ → a_1|1,0⟩ = 0 (n_1=0)
        // a_0 a†_1 |1,0⟩ → a†_1 first: |1,1⟩, then a_0: |0,1⟩ with amp √1 * √1 = 1
        let manip = DynamicDitManip::new(3);
        let mut state: u32 = 0;
        state = manip.set_dit(state, 1, 0); // site 0 = 1

        let terms = vec![
            BosonOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(BosonOp::Plus, 0), (BosonOp::Minus, 1)],
            ),
            BosonOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(BosonOp::Minus, 0), (BosonOp::Plus, 1)],
            ),
        ];
        let ham = BosonOperator::new(terms, 3);
        let result = ham.apply_smallvec(state);

        // Only a_0 a†_1 |1,0⟩ is non-zero: gives |0,1⟩ with amp 1
        assert_eq!(result.len(), 1);
        let (amp, new_state, _cindex) = result[0];
        assert_eq!(manip.get_dit(new_state, 0), 0);
        assert_eq!(manip.get_dit(new_state, 1), 1);
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn hamiltonian_number_diagonal() {
        // H = n̂_0 (diagonal), lhss=4, site 0 has n=2
        let manip = DynamicDitManip::new(4);
        let mut state: u32 = 0;
        state = manip.set_dit(state, 2, 0);

        let terms = vec![BosonOpEntry::new(
            0u8,
            Complex::new(1.0, 0.0),
            smallvec![(BosonOp::N, 0)],
        )];
        let ham = BosonOperator::new(terms, 4);
        let result = ham.apply_smallvec(state);

        assert_eq!(result.len(), 1);
        let (amp, ns, _) = result[0];
        assert_eq!(ns, state);
        assert!((amp - Complex::new(2.0, 0.0)).norm() < 1e-12);
    }
}
