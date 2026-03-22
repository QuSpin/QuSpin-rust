/// Type-erased symmetry operation description and Python integer helpers.
///
/// `GrpOpDesc` stores the data for a local symmetry operation before the basis
/// integer type `B` is known.  `into_grp_element::<B>` instantiates the
/// concrete `GrpElement<B>` at basis-construction time.
use bitbasis::{BitInt, DynamicHigherSpinInv, DynamicPermDitValues, PermDitMask};
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
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
        /// Mask bits packed as little-endian `u64` limbs.
        mask_limbs: Vec<u64>,
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
            GrpOpDesc::Bitflip { mask_limbs } => {
                let mask = mask_from_limbs::<B>(&mask_limbs);
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reconstruct a `B` value from little-endian `u64` limbs.
pub fn mask_from_limbs<B: BitInt>(limbs: &[u64]) -> B {
    let bits = B::BITS as usize;
    let full_limbs = bits / 64;
    let mut result = B::from_u64(0);

    for (i, &limb) in limbs.iter().take(full_limbs).enumerate() {
        result = result | (B::from_u64(limb) << (i * 64));
    }

    let partial_bits = bits % 64;
    if partial_bits != 0 {
        if let Some(&limb) = limbs.get(full_limbs) {
            let mask = (1u64 << partial_bits) - 1;
            result = result | (B::from_u64(limb & mask) << (full_limbs * 64));
        }
    } else if full_limbs == 0
        && let Some(&limb) = limbs.first()
    {
        result = B::from_u64(limb);
    }

    result
}

/// Extract a Python `int` as a vector of little-endian `u64` limbs.
pub fn python_int_to_limbs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let bit_len: usize = obj.call_method0("bit_length")?.extract()?;
    if bit_len == 0 {
        return Ok(vec![0]);
    }
    let n_bytes = bit_len.div_ceil(8);
    let n_bytes_padded = n_bytes.div_ceil(8) * 8;
    let py_bytes = obj.call_method1("to_bytes", (n_bytes_padded, "little"))?;
    let bytes: Vec<u8> = py_bytes.extract()?;
    let limbs = bytes
        .chunks(8)
        .map(|chunk| {
            let mut arr = [0u8; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(arr)
        })
        .collect();
    Ok(limbs)
}
