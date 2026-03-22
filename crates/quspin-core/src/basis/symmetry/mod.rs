pub mod dispatch;

pub use dispatch::SymmetryGrpInner;

use crate::error::QuSpinError;
use bitbasis::{BitInt, DynamicHigherSpinInv, DynamicPermDitValues, PermDitLocations, PermDitMask};
use num_complex::Complex;

// ---------------------------------------------------------------------------
// LatticeElement — a site-permutation with an associated character
// ---------------------------------------------------------------------------

/// A lattice symmetry element (translation, reflection, …).
///
/// Always wraps a `PermDitLocations`; stored separately from local elements
/// because the lattice field of `SymmetryGrp` is always this type.
#[derive(Clone)]
pub struct LatticeElement {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub op: PermDitLocations,
}

impl LatticeElement {
    pub fn new(grp_char: Complex<f64>, op: PermDitLocations, n_sites: usize) -> Self {
        LatticeElement {
            grp_char,
            n_sites,
            op,
        }
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Apply this element: returns `(op.app(state), coeff * grp_char)`.
    #[inline]
    pub fn apply<B: BitInt>(&self, state: B, coeff: Complex<f64>) -> (B, Complex<f64>) {
        (self.op.app(state), coeff * self.grp_char)
    }
}

// ---------------------------------------------------------------------------
// GrpOpKind — closed enum of local symmetry operations
// ---------------------------------------------------------------------------

/// Local symmetry bit-operations (not lattice permutations).
///
/// Using a closed enum avoids heap allocation for each group element while
/// keeping the `GrpElement` type uniform.
///
/// Mirrors the non-lattice `grp_element` instantiations in `grp_element.hpp`.
#[derive(Clone)]
pub enum GrpOpKind<B: BitInt> {
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
            GrpOpKind::Bitflip(op) => op.app(state),
            GrpOpKind::LocalValue(op) => op.app(state),
            GrpOpKind::SpinInversion(op) => op.app(state),
        }
    }
}

// ---------------------------------------------------------------------------
// GrpElement — a local symmetry operation with an associated character
// ---------------------------------------------------------------------------

/// A single local symmetry group element: a bit-operation and its character.
///
/// Mirrors `grp_element<bit_op, bitset_t>` from `single_grp_element.hpp`
/// for the non-lattice operations.
#[derive(Clone)]
pub struct GrpElement<B: BitInt> {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub op: GrpOpKind<B>,
}

impl<B: BitInt> GrpElement<B> {
    pub fn new(grp_char: Complex<f64>, op: GrpOpKind<B>, n_sites: usize) -> Self {
        GrpElement {
            grp_char,
            n_sites,
            op,
        }
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
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
#[derive(Clone)]
pub struct SymmetryGrp<B: BitInt> {
    n_sites: usize,
    lattice: Vec<LatticeElement>,
    local: Vec<GrpElement<B>>,
}

impl<B: BitInt> SymmetryGrp<B> {
    /// Construct a symmetry group, validating that all elements agree on `n_sites`.
    ///
    /// Returns `Err` if any lattice or local element has a different `n_sites`
    /// than the first element encountered.  Returns `Ok` with `n_sites = 0` if
    /// both lists are empty.
    pub fn new(
        lattice: Vec<LatticeElement>,
        local: Vec<GrpElement<B>>,
    ) -> Result<Self, QuSpinError> {
        let mut n_sites_opt: Option<usize> = None;

        for el in &lattice {
            let n = el.n_sites;
            match n_sites_opt {
                None => n_sites_opt = Some(n),
                Some(existing) if existing != n => {
                    return Err(QuSpinError::ValueError(format!(
                        "n_sites mismatch in symmetry group: lattice element has {n} but expected {existing}"
                    )));
                }
                _ => {}
            }
        }

        for el in &local {
            let n = el.n_sites;
            match n_sites_opt {
                None => n_sites_opt = Some(n),
                Some(existing) if existing != n => {
                    return Err(QuSpinError::ValueError(format!(
                        "n_sites mismatch in symmetry group: local element has {n} but expected {existing}"
                    )));
                }
                _ => {}
            }
        }

        let n_sites = n_sites_opt.unwrap_or(0);
        Ok(SymmetryGrp {
            n_sites,
            lattice,
            local,
        })
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
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

    // Build a Z₂ bitflip group on a 2-site chain.
    // Identity permutation goes in lattice; bitflip goes in local.
    fn bitflip_grp_2site() -> SymmetryGrp<u32> {
        let id_lat = LatticeElement::new(
            Complex::new(1.0, 0.0),
            PermDitLocations::new(2, &[0, 1]), // identity on 2 sites
            2,
        );
        let mask: u32 = 0b11;
        let op = GrpOpKind::Bitflip(PermDitMask::new(mask));
        let el = GrpElement::new(Complex::new(1.0, 0.0), op, 2);
        SymmetryGrp::new(vec![id_lat], vec![el]).unwrap()
    }

    #[test]
    fn bitflip_get_refstate() {
        let grp = bitflip_grp_2site();
        // Images of |01⟩: {id(|01⟩), id(flip(|01⟩))} = {1, 2}. Representative = 2.
        let (ref_state, _coeff) = grp.get_refstate(0b01u32);
        assert_eq!(ref_state, 0b10u32);
    }

    #[test]
    fn bitflip_check_refstate_norm() {
        let grp = bitflip_grp_2site();
        // Images of |00⟩: {0, 3}. count(== |00⟩) = 1 (identity maps it to itself).
        let (_ref, norm_00) = grp.check_refstate(0b00u32);
        assert_eq!(norm_00, 1.0);

        // Images of |01⟩: {1, 2}. count(== |01⟩) = 1.
        let (_ref, norm_01) = grp.check_refstate(0b01u32);
        assert_eq!(norm_01, 1.0);
    }

    #[test]
    fn lattice_permutation_translation() {
        // 3-site translation: site 0→1, 1→2, 2→0 (lhss=2, spin-1/2)
        let lat_el = LatticeElement::new(
            Complex::new(1.0, 0.0),
            PermDitLocations::new(2, &[1, 2, 0]),
            3,
        );
        let grp = SymmetryGrp::<u32>::new(vec![lat_el], vec![]).unwrap();

        // state |001⟩=1, T(|001⟩)=|010⟩=2, so ref = 2
        let (ref_state, _) = grp.get_refstate(0b001u32);
        assert_eq!(ref_state, 0b010u32);
    }
}
