use crate::bitbasis::{BitInt, manip::DynamicDitManip};
use crate::error::QuSpinError;
use crate::hamiltonian::ParseOp;
use num_complex::Complex;
use smallvec::SmallVec;

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
