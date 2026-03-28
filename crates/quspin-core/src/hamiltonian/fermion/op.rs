use crate::bitbasis::BitInt;
use crate::error::QuSpinError;
use crate::hamiltonian::ParseOp;
use num_complex::Complex;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// FermionOp
// ---------------------------------------------------------------------------

/// A single-site fermionic operator.
///
/// Orbital labelling convention: site `2*i` = spin-down orbital `i`,
/// site `2*i+1` = spin-up orbital `i`.  The basis is a `HardcoreBasis`
/// (LHSS=2) — each bit represents orbital occupancy.
///
/// Jordan-Wigner string: the sign for operator at site `i` is
/// `(-1)^popcount(state & ((1<<i)-1))`.  For multi-operator terms the
/// sign is accumulated right-to-left, each step using the intermediate
/// state after applying the previous operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FermionOp {
    /// c†_i  (creation): flip 0→1 at site i, amplitude = JW_sign if empty, else 0
    Plus,
    /// c_i   (annihilation): flip 1→0 at site i, amplitude = JW_sign if occupied, else 0
    Minus,
    /// n̂_i  (number): no flip, amplitude = occupancy (no JW sign needed)
    N,
}

impl FermionOp {
    /// Parse a single ASCII character into a `FermionOp`.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '+' => Some(FermionOp::Plus),
            '-' => Some(FermionOp::Minus),
            'n' | 'N' => Some(FermionOp::N),
            _ => None,
        }
    }

    /// Compute the Jordan-Wigner sign for site `loc` and `state`.
    ///
    /// Sign = (-1)^popcount(state & ((1<<loc)-1)).
    ///
    /// Computed by summing the occupancy of each site below `loc`.
    /// `BitInt` has no `count_ones` method, so we iterate bit-by-bit.
    #[inline]
    fn jw_sign<B: BitInt>(state: B, loc: u32) -> f64 {
        let mut parity = 0u32;
        for i in 0..loc {
            parity += ((state >> i as usize) & B::from_u64(1)).to_usize() as u32;
        }
        if parity & 1 == 0 { 1.0 } else { -1.0 }
    }

    /// Apply this operator to `state` at site `loc`.
    ///
    /// Returns `(new_state, amplitude)`.  Amplitude is zero when the
    /// operator cannot act (e.g., creation on an occupied site).
    ///
    /// The JW sign is included for `Plus` and `Minus`.
    #[inline]
    pub fn apply<B: BitInt>(self, state: B, loc: u32) -> (B, Complex<f64>) {
        let n = ((state >> loc as usize) & B::from_u64(1)).to_usize() & 1;

        match self {
            FermionOp::Plus => {
                if n == 1 {
                    // site occupied — c† annihilates
                    (state, Complex::new(0.0, 0.0))
                } else {
                    let sign = Self::jw_sign(state, loc);
                    let new_state = state ^ (B::from_u64(1) << loc as usize);
                    (new_state, Complex::new(sign, 0.0))
                }
            }
            FermionOp::Minus => {
                if n == 0 {
                    // site empty — c annihilates
                    (state, Complex::new(0.0, 0.0))
                } else {
                    let sign = Self::jw_sign(state, loc);
                    let new_state = state ^ (B::from_u64(1) << loc as usize);
                    (new_state, Complex::new(sign, 0.0))
                }
            }
            FermionOp::N => {
                // number operator: no flip, no JW sign
                (state, Complex::new(n as f64, 0.0))
            }
        }
    }
}

impl ParseOp for FermionOp {
    fn from_char(ch: char) -> Result<Self, QuSpinError> {
        FermionOp::from_char(ch).ok_or_else(|| {
            QuSpinError::ValueError(format!(
                "unknown fermion operator character '{ch}'; expected one of +, -, n"
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// FermionOpEntry
// ---------------------------------------------------------------------------

/// A single term in a fermionic Hamiltonian: a coefficient, a cindex, and
/// the ordered list of (FermionOp, site) pairs.
///
/// `SmallVec<[_; 4]>` keeps 1–4-body operators heap-free.
///
/// The JW sign is accumulated right-to-left as each operator is applied;
/// each step uses the *intermediate* state (after previous operators).
#[derive(Clone, Debug)]
pub struct FermionOpEntry<C> {
    pub cindex: C,
    pub coeff: Complex<f64>,
    /// Ordered right-to-left: element 0 is applied last.
    pub ops: SmallVec<[(FermionOp, u32); 4]>,
}

impl<C: Copy> FermionOpEntry<C> {
    pub fn new(cindex: C, coeff: Complex<f64>, ops: SmallVec<[(FermionOp, u32); 4]>) -> Self {
        FermionOpEntry { cindex, coeff, ops }
    }

    /// Apply this operator string to `state`, returning `(amplitude, new_state)`.
    ///
    /// Ops are applied right-to-left (last element in `ops` first).  For
    /// each Plus/Minus op the Jordan-Wigner sign is computed against the
    /// *current intermediate state* and multiplied into the running amplitude.
    #[inline]
    pub fn apply<B: BitInt>(&self, state: B) -> (Complex<f64>, B) {
        let mut amplitude = self.coeff;
        let mut s = state;
        for &(op, loc) in self.ops.iter().rev() {
            let (ns, amp) = op.apply(s, loc);
            s = ns;
            amplitude *= amp;
            // Short-circuit: if amplitude is already zero further ops won't
            // change that, but we still need to return consistently.
            if amplitude == Complex::new(0.0, 0.0) {
                return (amplitude, s);
            }
        }
        (amplitude, s)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    // --- FermionOp::apply ---

    #[test]
    fn creation_on_empty_site() {
        // c†_0 |00⟩ = +|01⟩  (no bits below site 0 → JW sign = +1)
        let (ns, amp) = FermionOp::Plus.apply(0b00u32, 0);
        assert_eq!(ns, 0b01u32);
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn creation_on_occupied_site() {
        // c†_0 |01⟩ = 0  (site 0 already occupied)
        let (_, amp) = FermionOp::Plus.apply(0b01u32, 0);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn annihilation_on_occupied_site() {
        // c_0 |01⟩ = +|00⟩  (no bits below site 0 → JW sign = +1)
        let (ns, amp) = FermionOp::Minus.apply(0b01u32, 0);
        assert_eq!(ns, 0b00u32);
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn annihilation_on_empty_site() {
        // c_0 |00⟩ = 0
        let (_, amp) = FermionOp::Minus.apply(0b00u32, 0);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn number_operator() {
        let (ns0, amp0) = FermionOp::N.apply(0b00u32, 0);
        assert_eq!(ns0, 0b00u32);
        assert!((amp0 - Complex::new(0.0, 0.0)).norm() < 1e-12);

        let (ns1, amp1) = FermionOp::N.apply(0b01u32, 0);
        assert_eq!(ns1, 0b01u32);
        assert!((amp1 - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn jw_sign_site1_with_site0_occupied() {
        // c†_1 |01⟩: bits below site 1 = bit 0 = 1 → parity 1 → sign = -1
        // Result: |11⟩ with amplitude -1
        let (ns, amp) = FermionOp::Plus.apply(0b01u32, 1);
        assert_eq!(ns, 0b11u32);
        assert!((amp - Complex::new(-1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn jw_sign_site1_with_site0_empty() {
        // c†_1 |00⟩: bits below site 1 = bit 0 = 0 → parity 0 → sign = +1
        let (ns, amp) = FermionOp::Plus.apply(0b00u32, 1);
        assert_eq!(ns, 0b10u32);
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    // --- FermionOpEntry::apply ---

    #[test]
    fn two_body_hopping_term() {
        // c†_0 c_1 |10⟩ = |01⟩ with overall coefficient 1
        // Applied right-to-left:
        //   1. c_1 |10⟩: site 1 occupied, bits below = bit 0 = 0 → sign +1 → |00⟩, amp=+1
        //   2. c†_0 |00⟩: site 0 empty, bits below = 0 → sign +1 → |01⟩, amp=+1
        // Total: +1 * 1 = +1
        let ops: SmallVec<[(FermionOp, u32); 4]> =
            smallvec![(FermionOp::Plus, 0), (FermionOp::Minus, 1)];
        let entry = FermionOpEntry::<u8>::new(0, Complex::new(1.0, 0.0), ops);

        let state: u32 = 0b10; // site 1 occupied
        let (amp, ns) = entry.apply(state);
        assert_eq!(ns, 0b01u32);
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn two_body_hopping_jw_sign() {
        // c†_0 c_1 |11⟩: site 1 occupied, site 0 occupied
        // Applied right-to-left:
        //   1. c_1 |11⟩: site 1 occupied, bits below site 1 = bit 0 = 1 → sign -1 → |01⟩, amp=-1
        //   2. c†_0 |01⟩: site 0 occupied → amplitude 0
        // Total: 0
        let ops: SmallVec<[(FermionOp, u32); 4]> =
            smallvec![(FermionOp::Plus, 0), (FermionOp::Minus, 1)];
        let entry = FermionOpEntry::<u8>::new(0, Complex::new(1.0, 0.0), ops);

        let state: u32 = 0b11;
        let (amp, _) = entry.apply(state);
        assert_eq!(amp, Complex::new(0.0, 0.0));
    }

    #[test]
    fn hopping_from_site0_to_site1() {
        // c†_1 c_0 |01⟩ = |10⟩
        // Applied right-to-left:
        //   1. c_0 |01⟩: site 0 occupied, bits below = 0 → sign +1 → |00⟩, amp=+1
        //   2. c†_1 |00⟩: site 1 empty, bits below = bit 0 = 0 → sign +1 → |10⟩, amp=+1
        // Total: +1
        let ops: SmallVec<[(FermionOp, u32); 4]> =
            smallvec![(FermionOp::Plus, 1), (FermionOp::Minus, 0)];
        let entry = FermionOpEntry::<u8>::new(0, Complex::new(1.0, 0.0), ops);

        let state: u32 = 0b01;
        let (amp, ns) = entry.apply(state);
        assert_eq!(ns, 0b10u32);
        assert!((amp - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }
}
