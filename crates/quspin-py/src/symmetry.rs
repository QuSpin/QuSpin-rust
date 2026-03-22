/// Python-facing symmetry group pyclasses.
///
/// `PyLatticeElement`, `PyGrpElement`, and `PySymmetryGrp` mirror the core
/// types `LatticeElement`, `GrpElement<B>`, and `SymmetryGrp<B>` but store
/// their data in a type-erased form (before the basis integer type `B` is
/// known).  The concrete `B` is selected at basis-construction time, at which
/// point `PySymmetryGrp::into_symmetry_grp::<B>()` instantiates everything.
///
/// ## Python API
///
/// ```python
/// T   = PyLatticeElement(grp_char=1.0+0j, perm=[1, 2, 3, 0], lhss=2)
/// P   = PyGrpElement.bitflip(grp_char=1.0+0j, mask=0b1111)
/// grp = PySymmetryGrp(lattice=[T], local=[P])
/// ```
use bitbasis::{BitInt, DynamicHigherSpinInv, DynamicPermDitValues, PermDitLocations, PermDitMask};
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use quspin_core::basis::group::{GrpElement, GrpOpKind, LatticeElement, SymmetryGrp};

// ---------------------------------------------------------------------------
// GrpOpDesc — type-erased local symmetry op
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

/// Reconstruct a `B` value from little-endian `u64` limbs.
///
/// Uses repeated `from_u64` + left-shift + `BitOr` to build up the value
/// without requiring additional trait bounds on `B`.  Limbs that would shift
/// completely outside `B::BITS` are skipped so this is safe for all sizes.
fn mask_from_limbs<B: BitInt>(limbs: &[u64]) -> B {
    let bits = B::BITS as usize;
    // Number of full 64-bit limbs that fit in B.
    let full_limbs = bits / 64;
    let mut result = B::from_u64(0);

    // Pack full limbs.
    for (i, &limb) in limbs.iter().take(full_limbs).enumerate() {
        result = result | (B::from_u64(limb) << (i * 64));
    }

    // Pack a partial limb if bits is not a multiple of 64 and there are
    // more limbs available.
    let partial_bits = bits % 64;
    if partial_bits != 0 {
        if let Some(&limb) = limbs.get(full_limbs) {
            let mask = (1u64 << partial_bits) - 1;
            result = result | (B::from_u64(limb & mask) << (full_limbs * 64));
        }
    } else if full_limbs == 0 {
        // B::BITS < 64 (e.g., u32): take the low bits of the first limb.
        if let Some(&limb) = limbs.first() {
            result = B::from_u64(limb); // from_u64 truncates to B::BITS
        }
    }

    result
}

/// Extract a Python `int` as a vector of little-endian `u64` limbs.
///
/// Uses Python's `int.bit_length()` and `int.to_bytes()` to handle integers
/// of any size.
fn python_int_to_limbs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let bit_len: usize = obj.call_method0("bit_length")?.extract()?;
    if bit_len == 0 {
        return Ok(vec![0]);
    }
    // Round up to the next multiple of 8 bytes (64 bits) for clean limb packing.
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

// ---------------------------------------------------------------------------
// PyLatticeElement
// ---------------------------------------------------------------------------

/// A lattice symmetry element: a site permutation with a group character.
///
/// Wraps `LatticeElement` (`PermDitLocations` + `grp_char`) in a form that
/// can be stored before the basis integer type is known.
#[pyclass(name = "PyLatticeElement")]
pub struct PyLatticeElement {
    pub grp_char: Complex<f64>,
    /// Forward permutation: perm[src] = dst.
    pub perm: Vec<usize>,
    pub lhss: usize,
}

impl PyLatticeElement {
    /// Convert to a core `LatticeElement`.
    pub fn to_lattice_element(&self) -> LatticeElement {
        LatticeElement::new(self.grp_char, PermDitLocations::new(self.lhss, &self.perm))
    }
}

#[pymethods]
impl PyLatticeElement {
    /// Create a lattice element.
    ///
    /// Args:
    ///   grp_char: Group character (complex scalar).
    ///   perm:     Forward site permutation as a list of ints.
    ///   lhss:     Local Hilbert space size (2 for spin-1/2).
    #[new]
    pub fn new(grp_char: Complex<f64>, perm: Vec<usize>, lhss: usize) -> Self {
        PyLatticeElement {
            grp_char,
            perm,
            lhss,
        }
    }
}

// ---------------------------------------------------------------------------
// PyGrpElement
// ---------------------------------------------------------------------------

/// A local symmetry group element: a bit-operation with a group character.
///
/// Type-erased before the basis integer type is known; converted to
/// `GrpElement<B>` at basis-construction time via `GrpOpDesc::into_grp_element`.
#[pyclass(name = "PyGrpElement")]
pub struct PyGrpElement {
    pub grp_char: Complex<f64>,
    pub op: GrpOpDesc,
}

#[pymethods]
impl PyGrpElement {
    /// XOR-with-mask (Z₂ bit-flip) symmetry.
    ///
    /// Args:
    ///   grp_char: Group character.
    ///   mask:     Python int; the bits that are flipped.
    #[staticmethod]
    pub fn bitflip(grp_char: Complex<f64>, mask: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mask_limbs = python_int_to_limbs(mask)?;
        Ok(PyGrpElement {
            grp_char,
            op: GrpOpDesc::Bitflip { mask_limbs },
        })
    }

    /// Local dit-value permutation symmetry.
    ///
    /// Args:
    ///   grp_char: Group character.
    ///   lhss:     Local Hilbert space size.
    ///   perm:     Value permutation: v → perm[v].  Length must equal lhss.
    ///   locs:     Sites to which the permutation is applied.
    #[staticmethod]
    pub fn local_value(
        grp_char: Complex<f64>,
        lhss: usize,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) -> Self {
        PyGrpElement {
            grp_char,
            op: GrpOpDesc::LocalValue { lhss, perm, locs },
        }
    }

    /// Spin-inversion symmetry: v → lhss − v − 1 at the given sites.
    ///
    /// Args:
    ///   grp_char: Group character.
    ///   lhss:     Local Hilbert space size.
    ///   locs:     Sites to which the inversion is applied.
    #[staticmethod]
    pub fn spin_inversion(grp_char: Complex<f64>, lhss: usize, locs: Vec<usize>) -> Self {
        PyGrpElement {
            grp_char,
            op: GrpOpDesc::SpinInversion { lhss, locs },
        }
    }
}

// ---------------------------------------------------------------------------
// PySymmetryGrp
// ---------------------------------------------------------------------------

/// A symmetry group: a product of lattice and local group elements.
///
/// At basis-construction time, call `into_symmetry_grp::<B>()` to produce a
/// `SymmetryGrp<B>` for the concrete basis integer type `B`.
#[pyclass(name = "PySymmetryGrp")]
pub struct PySymmetryGrp {
    /// Data extracted from the Python `PyLatticeElement` objects.
    pub lattice: Vec<(Complex<f64>, Vec<usize>, usize)>,
    /// Data extracted from the Python `PyGrpElement` objects.
    pub local: Vec<(Complex<f64>, GrpOpDesc)>,
}

impl PySymmetryGrp {
    /// Instantiate a `SymmetryGrp<B>` for the concrete basis integer type.
    pub fn into_symmetry_grp<B: BitInt>(&self) -> SymmetryGrp<B> {
        let lattice: Vec<LatticeElement> = self
            .lattice
            .iter()
            .map(|(char_, perm, lhss)| {
                LatticeElement::new(*char_, PermDitLocations::new(*lhss, perm))
            })
            .collect();

        let local: Vec<GrpElement<B>> = self
            .local
            .iter()
            .map(|(char_, op)| op.clone().into_grp_element::<B>(*char_))
            .collect();

        SymmetryGrp::new(lattice, local)
    }
}

#[pymethods]
impl PySymmetryGrp {
    /// Construct from lists of `PyLatticeElement` and `PyGrpElement` objects.
    ///
    /// Data is extracted from the Python objects immediately; the Python
    /// objects do not need to remain alive after this call.
    #[new]
    pub fn new(
        py: Python<'_>,
        lattice: &Bound<'_, PyList>,
        local: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        let mut lattice_data = Vec::with_capacity(lattice.len());
        for item in lattice.iter() {
            let el: PyRef<'_, PyLatticeElement> = item.extract()?;
            lattice_data.push((el.grp_char, el.perm.clone(), el.lhss));
        }

        let mut local_data = Vec::with_capacity(local.len());
        for item in local.iter() {
            let el: PyRef<'_, PyGrpElement> = item.extract()?;
            local_data.push((el.grp_char, el.op.clone()));
        }

        let _ = py; // consumed by extract calls above
        Ok(PySymmetryGrp {
            lattice: lattice_data,
            local: local_data,
        })
    }
}
