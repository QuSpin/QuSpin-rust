use crate::Operator;
use num_complex::Complex;
use quspin_bitbasis::{BitInt, manip::DynamicDitManip};
use smallvec::SmallVec;

use crate::ParseOp;
use quspin_types::QuSpinError;

// ---------------------------------------------------------------------------
// BosonOp
// ---------------------------------------------------------------------------

/// A single-site bosonic operator for LHSS > 2 (truncated harmonic oscillator).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BosonOp {
    /// Creation operator a†: a†|n⟩ = √(n+1)|n+1⟩  (zero if n = LHSS-1)
    Plus,
    /// Annihilation operator a: a|n⟩ = √n|n-1⟩  (zero if n = 0)
    Minus,
    /// Bare number operator n̂: n̂|n⟩ = n|n⟩
    N,
}

impl BosonOp {
    /// Parse a single ASCII character into a `BosonOp`.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '+' => Some(BosonOp::Plus),
            '-' => Some(BosonOp::Minus),
            'n' | 'N' => Some(BosonOp::N),
            _ => None,
        }
    }

    /// Apply this operator to `state` at site `loc`.
    ///
    /// Returns `(new_state, amplitude)`.  Returns `(state, 0)` when the
    /// action is zero (upper truncation boundary for `Plus`, lower for `Minus`).
    ///
    /// `manip` encodes the LHSS and bit-width for dit extraction/insertion.
    #[inline]
    pub fn apply<B: BitInt>(
        self,
        state: B,
        loc: usize,
        manip: &DynamicDitManip,
    ) -> (B, Complex<f64>) {
        let n = manip.get_dit(state, loc);
        let lhss = manip.lhss;
        match self {
            BosonOp::Plus => {
                if n + 1 >= lhss {
                    return (state, Complex::new(0.0, 0.0));
                }
                let new_state = manip.set_dit(state, n + 1, loc);
                (new_state, Complex::new((n as f64 + 1.0).sqrt(), 0.0))
            }
            BosonOp::Minus => {
                if n == 0 {
                    return (state, Complex::new(0.0, 0.0));
                }
                let new_state = manip.set_dit(state, n - 1, loc);
                (new_state, Complex::new((n as f64).sqrt(), 0.0))
            }
            BosonOp::N => (state, Complex::new(n as f64, 0.0)),
        }
    }
}

impl ParseOp for BosonOp {
    fn from_char(ch: char) -> Result<Self, QuSpinError> {
        BosonOp::from_char(ch).ok_or_else(|| {
            QuSpinError::ValueError(format!(
                "unknown boson operator character '{ch}'; expected one of +, -, n"
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// BosonOpEntry
// ---------------------------------------------------------------------------

/// A single term in a boson Hamiltonian: coefficient, cindex, and the ordered
/// list of `(BosonOp, site)` pairs.
///
/// `SmallVec<[_; 4]>` keeps 1–4-body operators heap-free.
#[derive(Clone, Debug)]
pub struct BosonOpEntry<C> {
    pub cindex: C,
    pub coeff: Complex<f64>,
    /// Ordered right-to-left: element 0 is applied last.
    pub ops: SmallVec<[(BosonOp, u32); 4]>,
}

impl<C: Copy> BosonOpEntry<C> {
    pub fn new(cindex: C, coeff: Complex<f64>, ops: SmallVec<[(BosonOp, u32); 4]>) -> Self {
        BosonOpEntry { cindex, coeff, ops }
    }

    /// Apply this operator string to `state`, returning `(amplitude, new_state)`.
    ///
    /// Ops are applied right-to-left.  A zero amplitude from any single op
    /// short-circuits to `(0, state)`.
    #[inline]
    pub fn apply<B: BitInt>(&self, state: B, manip: &DynamicDitManip) -> (Complex<f64>, B) {
        let mut amplitude = self.coeff;
        let mut s = state;
        for &(op, loc) in self.ops.iter().rev() {
            let (ns, amp) = op.apply(s, loc as usize, manip);
            s = ns;
            amplitude *= amp;
        }
        (amplitude, s)
    }
}

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
    use crate::boson::{BosonOp, BosonOpEntry};
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
