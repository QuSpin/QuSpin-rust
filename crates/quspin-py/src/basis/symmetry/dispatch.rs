/// Type-erased symmetry operation description.
///
/// `GrpOpDesc` stores the data for a local symmetry operation before the basis
/// integer type `B` is known.  `into_grp_element::<B>` instantiates the
/// concrete `GrpElement<B>` at basis-construction time.
use bitbasis::{BitInt, DynamicHigherSpinInv, DynamicPermDitValues, PermDitMask};
use num_complex::Complex;
use quspin_core::basis::group::{GrpElement, GrpOpKind};

// ---------------------------------------------------------------------------
// GrpOpDesc
// ---------------------------------------------------------------------------

/// Stores the data for a local symmetry operation independently of the basis
/// integer type `B`.
///
/// The concrete `GrpElement<B>` is constructed lazily via
/// [`GrpOpDesc::into_grp_element`] at basis-construction time.
#[derive(Clone, Debug)]
pub enum GrpOpDesc {
    Bitflip {
        /// Site indices whose bits are flipped.
        locs: Vec<usize>,
    },
    LocalValue {
        lhss: usize,
        /// Value permutation: maps dit value v → perm[v].  Length == lhss.
        perm: Vec<u8>,
        /// Sites to which the value permutation is applied.
        locs: Vec<usize>,
    },
    SpinInversion {
        lhss: usize,
        locs: Vec<usize>,
    },
}

impl GrpOpDesc {
    /// Convert into a `GrpElement<B>` for a concrete basis integer type.
    pub fn into_grp_element<B: BitInt>(self, grp_char: Complex<f64>) -> GrpElement<B> {
        let op = match self {
            GrpOpDesc::Bitflip { locs } => {
                let mask = locs.iter().fold(B::from_u64(0), |acc, &site| {
                    if site < B::BITS as usize {
                        acc | (B::from_u64(1) << site)
                    } else {
                        acc
                    }
                });
                GrpOpKind::Bitflip(PermDitMask::new(mask))
            }
            GrpOpDesc::LocalValue { lhss, perm, locs } => {
                GrpOpKind::LocalValue(DynamicPermDitValues::new(lhss, perm, locs))
            }
            GrpOpDesc::SpinInversion { lhss, locs } => {
                GrpOpKind::SpinInversion(DynamicHigherSpinInv::new(lhss, locs))
            }
        };
        GrpElement::new(grp_char, op)
    }
}
