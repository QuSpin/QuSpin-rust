use crate::bitbasis::{BitInt, manip::DynamicDitManip};
use crate::error::QuSpinError;
use crate::hamiltonian::ParseOp;
use num_complex::Complex;
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
            'z' | 'Z' => Some(SpinOp::Z),
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
