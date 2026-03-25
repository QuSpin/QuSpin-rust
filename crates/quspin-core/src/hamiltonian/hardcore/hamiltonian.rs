use crate::bitbasis::BitInt;
use crate::hamiltonian::Hamiltonian;
use num_complex::Complex;
use smallvec::SmallVec;

use super::op::OpEntry;

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
    /// Maximum site index across all operator strings (inferred from terms).
    max_site: usize,
    /// Number of distinct cindex values.
    num_cindices: usize,
}

impl<C: Copy + Ord> HardcoreHamiltonian<C> {
    /// Construct from a list of `OpEntry` terms.  Terms are sorted by `cindex`.
    /// `max_site` is inferred as the largest site index appearing in any op.
    pub fn new(mut terms: Vec<OpEntry<C>>) -> Self {
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
        HardcoreHamiltonian {
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

impl<C: Copy + Ord> Hamiltonian<C> for HardcoreHamiltonian<C> {
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
    use crate::hamiltonian::hardcore::op::{HardcoreOp, OpEntry};
    use num_complex::Complex;
    use smallvec::smallvec;

    // --- HardcoreOp::apply ---

    #[test]
    fn pauli_x_flips_bit() {
        // X at site 0: flips bit, amplitude = 1
        let (ns, amp) = HardcoreOp::X.apply(0u32, 0);
        assert_eq!(ns, 1u32);
        assert_eq!(amp, Complex::new(1.0, 0.0));

        let (ns2, amp2) = HardcoreOp::X.apply(1u32, 0);
        assert_eq!(ns2, 0u32);
        assert_eq!(amp2, Complex::new(1.0, 0.0));
    }

    #[test]
    fn pauli_z_no_flip() {
        // Z at site 0: no flip, amplitude = s = 2n-1 (QuSpin pauli=1 convention)
        let (ns0, amp0) = HardcoreOp::Z.apply(0u32, 0);
        assert_eq!(ns0, 0u32);
        assert_eq!(amp0, Complex::new(-1.0, 0.0)); // n=0 → s=-1

        let (ns1, amp1) = HardcoreOp::Z.apply(1u32, 0);
        assert_eq!(ns1, 1u32);
        assert_eq!(amp1, Complex::new(1.0, 0.0)); // n=1 → s=+1
    }

    #[test]
    fn pauli_p_creation() {
        // P (σ⁺) at site 0: 0→1 with amp=1, 1→? with amp=0
        let (ns, amp) = HardcoreOp::P.apply(0u32, 0);
        assert_eq!(ns, 1u32);
        assert_eq!(amp, Complex::new(1.0, 0.0));

        let (_, amp_zero) = HardcoreOp::P.apply(1u32, 0);
        assert_eq!(amp_zero, Complex::new(0.0, 0.0));
    }

    #[test]
    fn pauli_m_annihilation() {
        // M (σ⁻) at site 0: 1→0 with amp=1, 0→? with amp=0
        let (ns, amp) = HardcoreOp::M.apply(1u32, 0);
        assert_eq!(ns, 0u32);
        assert_eq!(amp, Complex::new(1.0, 0.0));

        let (_, amp_zero) = HardcoreOp::M.apply(0u32, 0);
        assert_eq!(amp_zero, Complex::new(0.0, 0.0));
    }

    #[test]
    fn pauli_n_number() {
        // N at site 0: no flip, amplitude = n
        let (ns0, amp0) = HardcoreOp::N.apply(0u32, 0);
        assert_eq!(ns0, 0u32);
        assert_eq!(amp0, Complex::new(0.0, 0.0));

        let (ns1, amp1) = HardcoreOp::N.apply(1u32, 0);
        assert_eq!(ns1, 1u32);
        assert_eq!(amp1, Complex::new(1.0, 0.0));
    }

    #[test]
    fn pauli_y_imaginary() {
        // Y at site 0: flips bit, amplitude = i*s  (s = 2n-1, QuSpin convention)
        // n=0 (empty): s=-1, amplitude = -i
        let (ns0, amp0) = HardcoreOp::Y.apply(0u32, 0);
        assert_eq!(ns0, 1u32);
        assert!((amp0 - Complex::new(0.0, -1.0)).norm() < 1e-12);

        // n=1 (occupied): s=+1, amplitude = +i
        let (ns1, amp1) = HardcoreOp::Y.apply(1u32, 0);
        assert_eq!(ns1, 0u32);
        assert!((amp1 - Complex::new(0.0, 1.0)).norm() < 1e-12);
    }

    // --- OpEntry::apply ---

    #[test]
    fn op_entry_two_body_xx() {
        // XX at sites (0, 1): flip both bits, amplitude = 1*1 = 1
        let ops: SmallVec<[(HardcoreOp, u32); 4]> =
            smallvec![(HardcoreOp::X, 0), (HardcoreOp::X, 1),];
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
        let ops: SmallVec<[(HardcoreOp, u32); 4]> = smallvec![(HardcoreOp::X, 0)];
        let terms = vec![OpEntry::<u8>::new(0, Complex::new(0.5, 0.0), ops)];
        let ham = HardcoreHamiltonian::new(terms);

        let state: u32 = 0;
        let result = ham.apply_smallvec(state);
        assert_eq!(result.len(), 1);
        let (amp, ns, cindex) = result[0];
        assert_eq!(ns, 1u32);
        assert!((amp - Complex::new(0.5, 0.0)).norm() < 1e-12);
        assert_eq!(cindex, 0u8);
    }

    #[test]
    fn hamiltonian_zero_amplitude_filtered() {
        // H = P_0 applied to state |1⟩ (occupied): P gives amplitude 0, should be filtered
        let ops: SmallVec<[(HardcoreOp, u32); 4]> = smallvec![(HardcoreOp::P, 0)];
        let terms = vec![OpEntry::<u8>::new(0, Complex::new(1.0, 0.0), ops)];
        let ham = HardcoreHamiltonian::new(terms);
        let result = ham.apply_smallvec(1u32);
        assert!(result.is_empty());
    }
}
