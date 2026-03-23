use crate::bitbasis::BitInt;
use crate::error::QuSpinError;
use crate::hamiltonian::ParseOp;
use num_complex::Complex;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// HardcoreOp
// ---------------------------------------------------------------------------

/// A single-site Pauli operator.
///
/// Mirrors `pauli::OperatorType` from `operator.hpp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardcoreOp {
    X,
    Y,
    Z,
    /// Creation (σ⁺)
    P,
    /// Annihilation (σ⁻)
    M,
    /// Number (n = σ⁺σ⁻)
    N,
}

impl HardcoreOp {
    /// Parse a single ASCII character into a `HardcoreOp`.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'x' | 'X' => Some(HardcoreOp::X),
            'y' | 'Y' => Some(HardcoreOp::Y),
            'z' | 'Z' => Some(HardcoreOp::Z),
            '+' => Some(HardcoreOp::P),
            '-' => Some(HardcoreOp::M),
            'n' | 'N' => Some(HardcoreOp::N),
            _ => None,
        }
    }

    /// Apply this operator to `state` at site `loc`.
    ///
    /// Returns `(new_state, amplitude)`.  The implementation mirrors the
    /// branchless arithmetic in `pauli::apply_op` from `operator.hpp`.
    ///
    /// Conventions (matching QuSpin `pauli=1`):
    /// - Site occupancy `n = (state >> loc) & 1`.
    /// - `s = 2n - 1` (+1 if occupied, -1 if empty).
    /// - X: flips bit, amplitude = 1.
    /// - Y: flips bit, amplitude = i*s.
    /// - Z: no flip, amplitude = s.
    /// - P (σ⁺): flips bit (0→1), amplitude = 1 if n=0, else 0.
    /// - M (σ⁻): flips bit (1→0), amplitude = 1 if n=1, else 0.
    /// - N (number): no flip, amplitude = n.
    #[inline]
    pub fn apply<B: BitInt>(self, state: B, loc: u32) -> (B, Complex<f64>) {
        let n = ((state >> loc as usize) & B::from_u64(1)).to_usize() & 1;
        let s = 2.0 * n as f64 - 1.0; // +1 if occupied, -1 if empty

        let is_x = self == HardcoreOp::X;
        let is_y = self == HardcoreOp::Y;
        let is_z = self == HardcoreOp::Z;
        let is_p = self == HardcoreOp::P;
        let is_m = self == HardcoreOp::M;
        let is_n = self == HardcoreOp::N;

        // Flip the bit for X, Y, P, M.
        let flips = is_x || is_y || is_p || is_m;
        let new_state = state ^ (B::from_u64(flips as u64) << loc as usize);

        // Amplitude: real part = Z*s + X + (M|N)*n + P*(1-n)
        //            imag part = Y*s
        let real = (is_z as i32 as f64) * s
            + (is_x as i32 as f64)
            + ((is_m || is_n) as i32 as f64) * n as f64
            + (is_p as i32 as f64) * (1 - n) as f64;
        let imag = (is_y as i32 as f64) * s;

        (new_state, Complex::new(real, imag))
    }
}

impl ParseOp for HardcoreOp {
    fn from_char(ch: char) -> Result<Self, QuSpinError> {
        HardcoreOp::from_char(ch).ok_or_else(|| {
            QuSpinError::ValueError(format!(
                "unknown operator character '{ch}'; expected one of x, y, z, +, -, n"
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// OpEntry
// ---------------------------------------------------------------------------

/// A single term in a Pauli Hamiltonian: a coefficient, a cindex (operator
/// string index used to look up the corresponding matrix coefficient), and
/// the ordered list of (HardcoreOp, site) pairs.
///
/// `SmallVec<[_; 4]>` keeps 1–4-body operators heap-free; longer strings
/// fall back to heap allocation gracefully.
///
/// Mirrors the collapsed `fixed_pauli_operator_string` / `pauli_operator_string`
/// design from `operator.hpp`.
#[derive(Clone, Debug)]
pub struct OpEntry<C> {
    pub cindex: C,
    pub coeff: Complex<f64>,
    /// Ordered right-to-left: element 0 is applied last.
    pub ops: SmallVec<[(HardcoreOp, u32); 4]>,
}

impl<C: Copy> OpEntry<C> {
    pub fn new(cindex: C, coeff: Complex<f64>, ops: SmallVec<[(HardcoreOp, u32); 4]>) -> Self {
        OpEntry { cindex, coeff, ops }
    }

    /// Apply this operator string to `state`, returning `(amplitude, new_state)`.
    ///
    /// Ops are applied right-to-left (last element in `ops` first), matching
    /// the C++ convention.
    #[inline]
    pub fn apply<B: BitInt>(&self, state: B) -> (Complex<f64>, B) {
        let mut amplitude = self.coeff;
        let mut s = state;
        for &(op, loc) in self.ops.iter().rev() {
            let (ns, amp) = op.apply(s, loc);
            s = ns;
            amplitude *= amp;
        }
        (amplitude, s)
    }
}
