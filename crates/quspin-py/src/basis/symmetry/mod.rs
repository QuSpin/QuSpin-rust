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
pub mod dispatch;

use bitbasis::{BitInt, PermDitLocations};
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use quspin_core::basis::group::{GrpElement, LatticeElement, SymmetryGrp};

use dispatch::GrpOpDesc;

// ---------------------------------------------------------------------------
// PyLatticeElement
// ---------------------------------------------------------------------------

/// A lattice symmetry element: a site permutation with a group character.
#[pyclass(name = "PyLatticeElement")]
pub struct PyLatticeElement {
    pub grp_char: Complex<f64>,
    /// Forward permutation: perm[src] = dst.
    pub perm: Vec<usize>,
    pub lhss: usize,
}

impl PyLatticeElement {
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
    ///   locs:     List of site indices whose bits are flipped.
    #[staticmethod]
    pub fn bitflip(grp_char: Complex<f64>, locs: Vec<usize>) -> Self {
        PyGrpElement {
            grp_char,
            op: GrpOpDesc::Bitflip { locs },
        }
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
    pub lattice: Vec<(Complex<f64>, Vec<usize>, usize)>,
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

        let _ = py;
        Ok(PySymmetryGrp {
            lattice: lattice_data,
            local: local_data,
        })
    }
}
