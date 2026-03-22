use bitbasis::{BitInt, DynamicHigherSpinInv, DynamicPermDitValues, PermDitLocations, PermDitMask};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// GrpOpKind — closed enum of all supported bit operations
// ---------------------------------------------------------------------------

/// All supported symmetry bit-operations.
///
/// Using a closed enum avoids heap allocation for each group element while
/// keeping the `Grp` type uniform.
///
/// Mirrors the heterogeneous `grp_element` instantiations in `grp_element.hpp`.
pub enum GrpOpKind<B: BitInt> {
    /// Site permutation (lattice translation, reflection, …).
    Lattice(PermDitLocations),
    /// XOR with a fixed mask (Z₂ bit-flip symmetry).
    Bitflip(PermDitMask<B>),
    /// Permutation of dit *values* at given sites (local on-site symmetry).
    LocalValue(DynamicPermDitValues),
    /// Spin inversion: v → lhss − v − 1 at given sites.
    SpinInversion(DynamicHigherSpinInv),
}

impl<B: BitInt> GrpOpKind<B> {
    /// Apply the operation to `state`.
    #[inline]
    pub fn app(&self, state: B) -> B {
        match self {
            GrpOpKind::Lattice(op) => op.app(state),
            GrpOpKind::Bitflip(op) => op.app(state),
            GrpOpKind::LocalValue(op) => op.app(state),
            GrpOpKind::SpinInversion(op) => op.app(state),
        }
    }
}

// ---------------------------------------------------------------------------
// GrpElement — a symmetry operation with an associated character
// ---------------------------------------------------------------------------

/// A single symmetry group element: a bit-operation and its character
/// (eigenvalue / phase factor).
///
/// `apply` maps `(state, accumulated_coeff)` → `(new_state, new_coeff)`.
///
/// Mirrors `grp_element<bit_op, bitset_t>` from `single_grp_element.hpp`.
pub struct GrpElement<B: BitInt> {
    pub grp_char: Complex<f64>,
    pub op: GrpOpKind<B>,
}

impl<B: BitInt> GrpElement<B> {
    pub fn new(grp_char: Complex<f64>, op: GrpOpKind<B>) -> Self {
        GrpElement { grp_char, op }
    }

    /// Apply this element: returns `(op.app(state), coeff * grp_char)`.
    #[inline]
    pub fn apply(&self, state: B, coeff: Complex<f64>) -> (B, Complex<f64>) {
        (self.op.app(state), coeff * self.grp_char)
    }
}

// ---------------------------------------------------------------------------
// SymmetryGrp
// ---------------------------------------------------------------------------

/// A symmetry group: a product of lattice and local group elements.
///
/// Given an input state, `get_refstate` finds the lexicographically largest
/// state in the orbit (the representative state), together with the
/// accumulated coefficient.  `check_refstate` computes the norm of the orbit
/// (number of distinct group images).
///
/// The combined action follows the C++ `grp::check_refstate` / `get_refstate`
/// convention: first apply lattice ops only, then apply each local op
/// followed by all lattice ops.
///
/// Mirrors `grp<grp_result_t, lattice_t, local_t>` from `grp.hpp`.
pub struct SymmetryGrp<B: BitInt> {
    lattice: Vec<GrpElement<B>>,
    local: Vec<GrpElement<B>>,
}

impl<B: BitInt> SymmetryGrp<B> {
    pub fn new(lattice: Vec<GrpElement<B>>, local: Vec<GrpElement<B>>) -> Self {
        SymmetryGrp { lattice, local }
    }

    /// Iterate over all group images of `state` (lattice-only, then local+lattice).
    fn iter_images(&self, state: B) -> impl Iterator<Item = (B, Complex<f64>)> + '_ {
        let input_coeff = Complex::new(1.0, 0.0);

        // lattice ops applied to the input directly
        let lattice_images = self
            .lattice
            .iter()
            .map(move |el| el.apply(state, input_coeff));

        // local op first, then all lattice ops
        let local_then_lattice = self.local.iter().flat_map(move |local_el| {
            let (local_state, local_coeff) = local_el.apply(state, input_coeff);
            self.lattice
                .iter()
                .map(move |lat_el| lat_el.apply(local_state, local_coeff))
        });

        lattice_images.chain(local_then_lattice)
    }

    /// Find the representative (largest state in the orbit) and the
    /// accumulated coefficient for that image.
    ///
    /// Returns `(ref_state, coeff)`.
    pub fn get_refstate(&self, state: B) -> (B, Complex<f64>) {
        let mut best_state = state;
        let mut best_coeff = Complex::new(1.0, 0.0);
        for (s, c) in self.iter_images(state) {
            if s > best_state {
                best_state = s;
                best_coeff = c;
            }
        }
        (best_state, best_coeff)
    }

    /// Count the number of distinct group images of `state` (the orbit norm).
    ///
    /// Returns `(ref_state, norm)` where `norm` is the count.
    pub fn check_refstate(&self, state: B) -> (B, f64) {
        let mut ref_state = state;
        let mut norm = 0.0_f64;
        for (s, _) in self.iter_images(state) {
            if s > ref_state {
                ref_state = s;
            }
            if s == state {
                norm += 1.0;
            }
        }
        (ref_state, norm)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Build a Z₂ bitflip group on a 2-site chain (sites 0 and 1).
    fn bitflip_grp_2site() -> SymmetryGrp<u32> {
        let mask: u32 = 0b11; // flip both sites
        let op = GrpOpKind::Bitflip(PermDitMask::new(mask));
        let el = GrpElement::new(Complex::new(1.0, 0.0), op);
        SymmetryGrp::new(vec![el], vec![])
    }

    #[test]
    fn bitflip_get_refstate() {
        let grp = bitflip_grp_2site();
        // state |01⟩ = 1, its image under full bitflip = |10⟩ = 2 (larger)
        let (ref_state, _coeff) = grp.get_refstate(0b01u32);
        assert_eq!(ref_state, 0b10u32);
    }

    #[test]
    fn bitflip_check_refstate_norm() {
        let grp = bitflip_grp_2site();
        // |00⟩ maps to |11⟩: distinct states, norm = 0 (state ≠ image for all images)
        let (_ref, norm_00) = grp.check_refstate(0b00u32);
        assert_eq!(norm_00, 0.0);

        // |01⟩ maps to |10⟩: also never equals itself in images, norm = 0
        let (_ref, norm_01) = grp.check_refstate(0b01u32);
        assert_eq!(norm_01, 0.0);
    }

    #[test]
    fn lattice_permutation_translation() {
        // 3-site translation: site 0→1, 1→2, 2→0 (lhss=2, spin-1/2)
        let op = GrpOpKind::Lattice(PermDitLocations::new(2, &[1, 2, 0]));
        let el = GrpElement::new(Complex::new(1.0, 0.0), op);
        let grp = SymmetryGrp::new(vec![el], vec![]);

        // state |100⟩ = 1: site 0 set, sites 1,2 clear
        // after T: site 1 set = |010⟩ = 2
        let (ref_state, _) = grp.get_refstate(0b001u32);
        // |001⟩=1, T(|001⟩)=|010⟩=2, so ref = 2
        assert_eq!(ref_state, 0b010u32);
    }
}
