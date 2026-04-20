use crate::Operator;
use crate::ParseOp;
use num_complex::Complex;
use quspin_bitbasis::{BitInt, manip::DynamicDitManip};
use quspin_types::QuSpinError;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// SpinOp
// ---------------------------------------------------------------------------

/// A single-site spin-S operator.
///
/// State encoding: dit value `n ∈ [0, lhss-1]` represents spin projection
/// `m = S - n`, where `S = (lhss - 1) / 2`.  So `n = 0` is the highest
/// spin state `m = +S` and `n = lhss - 1` is the lowest `m = -S`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinOp {
    /// Raising operator S+: S+|n⟩ = √(S(S+1) - m(m+1)) |n-1⟩  (zero if n = 0)
    Plus,
    /// Lowering operator S-: S-|n⟩ = √(S(S+1) - m(m-1)) |n+1⟩  (zero if n = lhss-1)
    Minus,
    /// z-component Sz: Sz|n⟩ = m|n⟩  where m = S - n
    Z,
}

impl SpinOp {
    /// Parse a single ASCII character into a `SpinOp`.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '+' => Some(SpinOp::Plus),
            '-' => Some(SpinOp::Minus),
            _ => None,
        }
    }

    /// Apply this operator to `state` at site `loc`.
    ///
    /// Returns `(new_state, amplitude)`.  Returns `(state, 0)` when the
    /// action is zero (upper boundary for `Plus`, lower for `Minus`).
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
        let s = (lhss - 1) as f64 / 2.0;
        let m = s - n as f64;
        match self {
            SpinOp::Plus => {
                if n == 0 {
                    return (state, Complex::new(0.0, 0.0));
                }
                let amp = (s * (s + 1.0) - m * (m + 1.0)).sqrt();
                (manip.set_dit(state, n - 1, loc), Complex::new(amp, 0.0))
            }
            SpinOp::Minus => {
                if n + 1 >= lhss {
                    return (state, Complex::new(0.0, 0.0));
                }
                let amp = (s * (s + 1.0) - m * (m - 1.0)).sqrt();
                (manip.set_dit(state, n + 1, loc), Complex::new(amp, 0.0))
            }
            SpinOp::Z => (state, Complex::new(m, 0.0)),
        }
    }
}

impl ParseOp for SpinOp {
    fn from_char(ch: char) -> Result<Self, QuSpinError> {
        SpinOp::from_char(ch).ok_or_else(|| {
            QuSpinError::ValueError(format!(
                "unknown spin operator character '{ch}'; expected one of +, -, z"
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// SpinOpEntry
// ---------------------------------------------------------------------------

/// A single term in a spin-S Hamiltonian: coefficient, cindex, and the ordered
/// list of `(SpinOp, site)` pairs.
///
/// `SmallVec<[_; 4]>` keeps 1–4-body operators heap-free.
#[derive(Clone, Debug)]
pub struct SpinOpEntry<C> {
    pub cindex: C,
    pub coeff: Complex<f64>,
    /// Ordered right-to-left: element 0 is applied last.
    pub ops: SmallVec<[(SpinOp, u32); 4]>,
}

impl<C: Copy> SpinOpEntry<C> {
    pub fn new(cindex: C, coeff: Complex<f64>, ops: SmallVec<[(SpinOp, u32); 4]>) -> Self {
        SpinOpEntry { cindex, coeff, ops }
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
            if amplitude == Complex::new(0.0, 0.0) {
                return (amplitude, state);
            }
        }
        (amplitude, s)
    }
}

// ---------------------------------------------------------------------------
// SpinOperator
// ---------------------------------------------------------------------------

/// A collection of operator strings forming a spin-S Hamiltonian.
///
/// Operators are `+` (S+), `-` (S-), and `z` (Sz) acting on sites with
/// `lhss = 2S+1` levels each.
///
/// State encoding: dit value `n` represents spin projection `m = S - n`,
/// so `n = 0` is the highest spin state `m = +S`.
///
/// Terms are stored sorted by `cindex` and a `DynamicDitManip` is stored
/// to avoid re-construction on every `apply` call.
#[derive(Clone, Debug)]
pub struct SpinOperator<C> {
    terms: Vec<SpinOpEntry<C>>,
    manip: DynamicDitManip,
    /// Maximum site index across all operator strings (inferred from terms).
    max_site: usize,
    /// Number of distinct cindex values.
    num_cindices: usize,
}

impl<C: Copy + Ord> SpinOperator<C> {
    /// Construct from a list of `SpinOpEntry` terms and the LHSS.
    /// Terms are sorted by `cindex`.
    pub fn new(mut terms: Vec<SpinOpEntry<C>>, lhss: usize) -> Self {
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
        SpinOperator {
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

    pub fn terms(&self) -> &[SpinOpEntry<C>] {
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

impl<C: Copy + Ord> Operator<C> for SpinOperator<C> {
    fn max_site(&self) -> usize {
        self.max_site
    }

    fn num_cindices(&self) -> usize {
        self.num_cindices
    }

    fn lhss(&self) -> usize {
        self.manip.lhss
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
    use crate::spin::{SpinOp, SpinOpEntry};
    use num_complex::Complex;
    use smallvec::smallvec;

    // Helpers
    fn manip(lhss: usize) -> DynamicDitManip {
        DynamicDitManip::new(lhss)
    }

    // --- SpinOp::apply: spin-1/2 (lhss=2, S=1/2) ---

    #[test]
    fn spin_half_plus_on_down_gives_one() {
        // n=1 (m=-1/2, |↓⟩) → S+|↓⟩ = |↑⟩ with amp = √(3/4 - (-1/2)(1/2)) = √1 = 1
        let m = manip(2);
        let state: u32 = m.set_dit(0u32, 1, 0); // n=1 at site 0
        let (ns, amp) = SpinOp::Plus.apply(state, 0, &m);
        assert_eq!(m.get_dit(ns, 0), 0); // raised to n=0 (m=+1/2)
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn spin_half_plus_on_up_gives_zero() {
        // n=0 (m=+1/2, |↑⟩) → S+|↑⟩ = 0
        let m = manip(2);
        let state: u32 = 0; // n=0 at site 0
        let (_, amp) = SpinOp::Plus.apply(state, 0, &m);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn spin_half_minus_on_up_gives_one() {
        // n=0 (m=+1/2) → S-|↑⟩ = |↓⟩ with amp = √(3/4 - (1/2)(-1/2)) = √1 = 1
        let m = manip(2);
        let state: u32 = 0; // n=0
        let (ns, amp) = SpinOp::Minus.apply(state, 0, &m);
        assert_eq!(m.get_dit(ns, 0), 1); // lowered to n=1
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn spin_half_minus_on_down_gives_zero() {
        // n=1 (m=-1/2) → S-|↓⟩ = 0
        let m = manip(2);
        let state: u32 = m.set_dit(0u32, 1, 0);
        let (_, amp) = SpinOp::Minus.apply(state, 0, &m);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn spin_half_z_eigenvalues() {
        let m = manip(2);
        // |↑⟩: n=0, m=+1/2
        let (_, amp_up) = SpinOp::Z.apply(0u32, 0, &m);
        assert!((amp_up - Complex::new(0.5, 0.0)).norm() < 1e-12);
        // |↓⟩: n=1, m=-1/2
        let state_down = m.set_dit(0u32, 1, 0);
        let (_, amp_down) = SpinOp::Z.apply(state_down, 0, &m);
        assert!((amp_down - Complex::new(-0.5, 0.0)).norm() < 1e-12);
    }

    // --- SpinOp::apply: spin-1 (lhss=3, S=1) ---

    #[test]
    fn spin_one_plus_at_interior() {
        // n=2 (m=-1) → S+|m=-1⟩: amp = √(S(S+1) - (-1)(0)) = √(2+1) = √2? wait:
        // S=1, m=-1: amp² = 1*2 - (-1)*0 = 2, amp = √2
        let m = manip(3);
        let state: u32 = m.set_dit(0u32, 2, 0); // n=2 → m=-1
        let (ns, amp) = SpinOp::Plus.apply(state, 0, &m);
        assert_eq!(m.get_dit(ns, 0), 1); // raised to n=1 (m=0)
        assert!((amp - Complex::new(2.0f64.sqrt(), 0.0)).norm() < 1e-12);
    }

    #[test]
    fn spin_one_plus_at_max_gives_zero() {
        // n=0 (m=+1) → S+|m=+1⟩ = 0
        let m = manip(3);
        let (_, amp) = SpinOp::Plus.apply(0u32, 0, &m);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn spin_one_minus_at_interior() {
        // n=0 (m=+1) → S-|m=+1⟩: amp² = 1*2 - 1*0 = 2, amp = √2
        let m = manip(3);
        let (ns, amp) = SpinOp::Minus.apply(0u32, 0, &m);
        assert_eq!(m.get_dit(ns, 0), 1); // lowered to n=1 (m=0)
        assert!((amp - Complex::new(2.0f64.sqrt(), 0.0)).norm() < 1e-12);
    }

    #[test]
    fn spin_one_minus_at_min_gives_zero() {
        // n=2 (m=-1) → S-|m=-1⟩ = 0
        let m = manip(3);
        let state: u32 = m.set_dit(0u32, 2, 0);
        let (_, amp) = SpinOp::Minus.apply(state, 0, &m);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn spin_one_z_eigenvalues() {
        let m = manip(3);
        for (n, expected_m) in [(0, 1.0), (1, 0.0), (2, -1.0)] {
            let state: u32 = m.set_dit(0u32, n, 0);
            let (ns, amp) = SpinOp::Z.apply(state, 0, &m);
            assert_eq!(ns, state);
            assert!((amp - Complex::new(expected_m, 0.0)).norm() < 1e-12);
        }
    }

    // --- SpinOperator integration: S+_0 S-_1 + S-_0 S+_1 hopping for S=1 ---

    #[test]
    fn spin_one_hopping_connects_states() {
        // H = S+_0 S-_1 + S-_0 S+_1, lhss=3
        // State |m=0, m=+1⟩: site 0 = n=1 (m=0), site 1 = n=0 (m=+1)
        // S+_0 S-_1 |0,+1⟩: S-_1 on m=+1 gives amp √2, n=1; S+_0 on m=0 gives amp √2, n=0
        //   → |+1, 0⟩ with amp 2
        // S-_0 S+_1 |0,+1⟩: S+_1 on m=+1 → 0 (already max)
        let m_manip = manip(3);
        let mut state: u32 = 0;
        state = m_manip.set_dit(state, 1, 0); // site 0 = n=1 (m=0)
        // site 1 = n=0 (m=+1) already

        let terms = vec![
            SpinOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(SpinOp::Plus, 0), (SpinOp::Minus, 1)],
            ),
            SpinOpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(SpinOp::Minus, 0), (SpinOp::Plus, 1)],
            ),
        ];
        let ham = SpinOperator::new(terms, 3);
        let result = ham.apply_smallvec(state);

        assert_eq!(result.len(), 1);
        let (amp, new_state, _) = result[0];
        assert_eq!(m_manip.get_dit(new_state, 0), 0); // site 0 raised to m=+1
        assert_eq!(m_manip.get_dit(new_state, 1), 1); // site 1 lowered to m=0
        assert!((amp - Complex::new(2.0, 0.0)).norm() < 1e-12);
    }
}
