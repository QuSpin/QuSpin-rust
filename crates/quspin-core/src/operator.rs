use bitbasis::BitInt;
use num_complex::Complex;
use smallvec::SmallVec;

// ---------------------------------------------------------------------------
// PauliOp
// ---------------------------------------------------------------------------

/// A single-site Pauli operator.
///
/// Mirrors `pauli::OperatorType` from `operator.hpp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliOp {
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

impl PauliOp {
    /// Parse a single ASCII character into a `PauliOp`.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'x' | 'X' => Some(PauliOp::X),
            'y' | 'Y' => Some(PauliOp::Y),
            'z' | 'Z' => Some(PauliOp::Z),
            '+' => Some(PauliOp::P),
            '-' => Some(PauliOp::M),
            'n' | 'N' => Some(PauliOp::N),
            _ => None,
        }
    }

    /// Apply this operator to `state` at site `loc`.
    ///
    /// Returns `(new_state, amplitude)`.  The implementation mirrors the
    /// branchless arithmetic in `pauli::apply_op` from `operator.hpp`.
    ///
    /// Conventions:
    /// - Site occupancy `n = (state >> loc) & 1`.
    /// - `s = 1 - 2n` (±1 for empty/occupied).
    /// - X: flips bit, amplitude = 1.
    /// - Y: flips bit, amplitude = i*s.
    /// - Z: no flip, amplitude = s.
    /// - P (σ⁺): flips bit (0→1), amplitude = 1 if n=0, else 0.
    /// - M (σ⁻): flips bit (1→0), amplitude = 1 if n=1, else 0.
    /// - N (number): no flip, amplitude = n.
    #[inline]
    pub fn apply<B: BitInt>(self, state: B, loc: u32) -> (B, Complex<f64>) {
        let n = ((state >> loc as usize) & B::from_u64(1)).to_usize() & 1;
        let s = 1.0 - 2.0 * n as f64; // +1 if empty, -1 if occupied

        let is_x = self == PauliOp::X;
        let is_y = self == PauliOp::Y;
        let is_z = self == PauliOp::Z;
        let is_p = self == PauliOp::P;
        let is_m = self == PauliOp::M;
        let is_n = self == PauliOp::N;

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

// ---------------------------------------------------------------------------
// OpEntry
// ---------------------------------------------------------------------------

/// A single term in a Pauli Hamiltonian: a coefficient, a cindex (operator
/// string index used to look up the corresponding matrix coefficient), and
/// the ordered list of (PauliOp, site) pairs.
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
    pub ops: SmallVec<[(PauliOp, u32); 4]>,
}

impl<C: Copy> OpEntry<C> {
    pub fn new(cindex: C, coeff: Complex<f64>, ops: SmallVec<[(PauliOp, u32); 4]>) -> Self {
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

// ---------------------------------------------------------------------------
// HardcoreHamiltonian
// ---------------------------------------------------------------------------

/// A collection of operator strings forming a Pauli Hamiltonian.
///
/// Terms are stored in a single `Vec<OpEntry<C>>` sorted by `cindex`.
/// The cindex type `C` is either `u8` (≤255 operator strings / site indices)
/// or `u16` (≤65535), chosen at construction time based on the input.
///
/// Mirrors `pauli_hamiltonian<cindex_t>` from `operator.hpp`.
#[derive(Clone, Debug)]
pub struct HardcoreHamiltonian<C> {
    terms: Vec<OpEntry<C>>,
    /// Number of sites (max site index + 1).
    n_sites: usize,
}

impl<C: Copy + Ord> HardcoreHamiltonian<C> {
    /// Construct from a list of `OpEntry` terms.  Terms are sorted by `cindex`.
    pub fn new(mut terms: Vec<OpEntry<C>>, n_sites: usize) -> Self {
        terms.sort_by_key(|e| e.cindex);
        HardcoreHamiltonian { terms, n_sites }
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    pub fn terms(&self) -> &[OpEntry<C>] {
        &self.terms
    }

    /// Apply the Hamiltonian to `state`.
    ///
    /// Returns a `SmallVec` of `(amplitude, new_state, cindex)` tuples,
    /// skipping zero-amplitude results.  Inline storage for up to 8 results
    /// covers typical Hamiltonian densities without heap allocation.
    ///
    /// Mirrors `pauli_hamiltonian::operator()` from `operator.hpp`.
    #[inline]
    pub fn apply<B: BitInt>(&self, state: B) -> SmallVec<[(Complex<f64>, B, C); 8]> {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- PauliOp::apply ---

    #[test]
    fn pauli_x_flips_bit() {
        // X at site 0: flips bit, amplitude = 1
        let (ns, amp) = PauliOp::X.apply(0u32, 0);
        assert_eq!(ns, 1u32);
        assert_eq!(amp, Complex::new(1.0, 0.0));

        let (ns2, amp2) = PauliOp::X.apply(1u32, 0);
        assert_eq!(ns2, 0u32);
        assert_eq!(amp2, Complex::new(1.0, 0.0));
    }

    #[test]
    fn pauli_z_no_flip() {
        // Z at site 0: no flip, amplitude = +1 (empty) or -1 (occupied)
        let (ns0, amp0) = PauliOp::Z.apply(0u32, 0);
        assert_eq!(ns0, 0u32);
        assert_eq!(amp0, Complex::new(1.0, 0.0)); // n=0 → s=+1

        let (ns1, amp1) = PauliOp::Z.apply(1u32, 0);
        assert_eq!(ns1, 1u32);
        assert_eq!(amp1, Complex::new(-1.0, 0.0)); // n=1 → s=-1
    }

    #[test]
    fn pauli_p_creation() {
        // P (σ⁺) at site 0: 0→1 with amp=1, 1→? with amp=0
        let (ns, amp) = PauliOp::P.apply(0u32, 0);
        assert_eq!(ns, 1u32);
        assert_eq!(amp, Complex::new(1.0, 0.0));

        let (_, amp_zero) = PauliOp::P.apply(1u32, 0);
        assert_eq!(amp_zero, Complex::new(0.0, 0.0));
    }

    #[test]
    fn pauli_m_annihilation() {
        // M (σ⁻) at site 0: 1→0 with amp=1, 0→? with amp=0
        let (ns, amp) = PauliOp::M.apply(1u32, 0);
        assert_eq!(ns, 0u32);
        assert_eq!(amp, Complex::new(1.0, 0.0));

        let (_, amp_zero) = PauliOp::M.apply(0u32, 0);
        assert_eq!(amp_zero, Complex::new(0.0, 0.0));
    }

    #[test]
    fn pauli_n_number() {
        // N at site 0: no flip, amplitude = n
        let (ns0, amp0) = PauliOp::N.apply(0u32, 0);
        assert_eq!(ns0, 0u32);
        assert_eq!(amp0, Complex::new(0.0, 0.0));

        let (ns1, amp1) = PauliOp::N.apply(1u32, 0);
        assert_eq!(ns1, 1u32);
        assert_eq!(amp1, Complex::new(1.0, 0.0));
    }

    #[test]
    fn pauli_y_imaginary() {
        // Y at site 0: flips bit, amplitude = i*s
        // n=0 (empty): s=+1, amplitude = i
        let (ns0, amp0) = PauliOp::Y.apply(0u32, 0);
        assert_eq!(ns0, 1u32);
        assert!((amp0 - Complex::new(0.0, 1.0)).norm() < 1e-12);

        // n=1 (occupied): s=-1, amplitude = -i
        let (ns1, amp1) = PauliOp::Y.apply(1u32, 0);
        assert_eq!(ns1, 0u32);
        assert!((amp1 - Complex::new(0.0, -1.0)).norm() < 1e-12);
    }

    // --- OpEntry::apply ---

    #[test]
    fn op_entry_two_body_xx() {
        // XX at sites (0, 1): flip both bits, amplitude = 1*1 = 1
        let ops: SmallVec<[(PauliOp, u32); 4]> =
            smallvec::smallvec![(PauliOp::X, 0), (PauliOp::X, 1),];
        let entry = OpEntry::<u8>::new(0, Complex::new(0.5, 0.0), ops);

        let state: u32 = 0b00;
        let (amp, ns) = entry.apply(state);
        assert_eq!(ns, 0b11);
        assert!((amp - Complex::new(0.5, 0.0)).norm() < 1e-12);
    }

    // --- HardcoreHamiltonian::apply ---

    #[test]
    fn hamiltonian_single_x_term() {
        // H = 0.5 * X_0, cindex=0
        let ops: SmallVec<[(PauliOp, u32); 4]> = smallvec::smallvec![(PauliOp::X, 0)];
        let terms = vec![OpEntry::<u8>::new(0, Complex::new(0.5, 0.0), ops)];
        let ham = HardcoreHamiltonian::new(terms, 1);

        let state: u32 = 0;
        let result = ham.apply(state);
        assert_eq!(result.len(), 1);
        let (amp, ns, cindex) = result[0];
        assert_eq!(ns, 1u32);
        assert!((amp - Complex::new(0.5, 0.0)).norm() < 1e-12);
        assert_eq!(cindex, 0u8);
    }

    #[test]
    fn hamiltonian_zero_amplitude_filtered() {
        // H = P_0 applied to state |1⟩ (occupied): P gives amplitude 0, should be filtered
        let ops: SmallVec<[(PauliOp, u32); 4]> = smallvec::smallvec![(PauliOp::P, 0)];
        let terms = vec![OpEntry::<u8>::new(0, Complex::new(1.0, 0.0), ops)];
        let ham = HardcoreHamiltonian::new(terms, 1);
        let result = ham.apply(1u32);
        assert!(result.is_empty());
    }
}
