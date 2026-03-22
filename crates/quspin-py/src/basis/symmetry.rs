/// Python-facing symmetry group pyclasses.
///
/// `PyLatticeElement` and `PyGrpElement` accept Python-level arguments.
/// `PySymmetryGrp` resolves the concrete basis integer type `B` at construction
/// time (based on `n_sites`) and stores an eager `SymmetryGrpInner`.
///
/// ## Python API
///
/// ```python
/// T   = PyLatticeElement(grp_char=1.0+0j, perm=[1, 2, 3, 0], lhss=2)
/// P   = PyGrpElement.bitflip(grp_char=1.0+0j, n_sites=4)
/// grp = PySymmetryGrp(lattice=[T], local=[P])
/// ```
use bitbasis::{BitInt, DynamicHigherSpinInv, DynamicPermDitValues, PermDitLocations, PermDitMask};
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use quspin_core::basis::SymmetryGrpInner;
use quspin_core::basis::symmetry::{GrpElement, GrpOpKind, LatticeElement, SymmetryGrp};

use crate::error::Error;

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
        n_sites: usize,
        /// Site indices whose bits are flipped.  `None` = all `n_sites` bits.
        locs: Option<Vec<usize>>,
    },
    LocalValue {
        n_sites: usize,
        lhss: usize,
        /// Value permutation: maps dit value v → perm[v].  Length == lhss.
        perm: Vec<u8>,
        /// Sites to which the value permutation is applied.
        locs: Vec<usize>,
    },
    SpinInversion {
        n_sites: usize,
        lhss: usize,
        locs: Vec<usize>,
    },
}

impl GrpOpDesc {
    pub fn n_sites(&self) -> usize {
        match self {
            GrpOpDesc::Bitflip { n_sites, .. } => *n_sites,
            GrpOpDesc::LocalValue { n_sites, .. } => *n_sites,
            GrpOpDesc::SpinInversion { n_sites, .. } => *n_sites,
        }
    }

    /// Convert into a `GrpElement<B>` for a concrete basis integer type.
    pub fn into_grp_element<B: BitInt>(self, grp_char: Complex<f64>) -> GrpElement<B> {
        let op = match &self {
            GrpOpDesc::Bitflip { n_sites, locs } => {
                let effective: Vec<usize> = match locs {
                    Some(l) => l.clone(),
                    None => (0..*n_sites).collect(),
                };
                let mask = effective.iter().fold(B::from_u64(0), |acc, &site| {
                    if site < B::BITS as usize {
                        acc | (B::from_u64(1) << site)
                    } else {
                        acc
                    }
                });
                GrpOpKind::Bitflip(PermDitMask::new(mask))
            }
            GrpOpDesc::LocalValue {
                lhss, perm, locs, ..
            } => {
                GrpOpKind::LocalValue(DynamicPermDitValues::new(*lhss, perm.clone(), locs.clone()))
            }
            GrpOpDesc::SpinInversion { lhss, locs, .. } => {
                GrpOpKind::SpinInversion(DynamicHigherSpinInv::new(*lhss, locs.clone()))
            }
        };
        let n_sites = self.n_sites();
        GrpElement::new(grp_char, op, n_sites)
    }
}

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
        LatticeElement::new(
            self.grp_char,
            PermDitLocations::new(self.lhss, &self.perm),
            self.perm.len(),
        )
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
/// `GrpElement<B>` inside `PySymmetryGrp::new` via `GrpOpDesc::into_grp_element`.
#[pyclass(name = "PyGrpElement")]
pub struct PyGrpElement {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub op: GrpOpDesc,
}

#[pymethods]
impl PyGrpElement {
    /// XOR-with-mask (Z₂ bit-flip) symmetry.
    ///
    /// Args:
    ///   grp_char: Group character.
    ///   n_sites:  Number of sites in the system.
    ///   locs:     Site indices whose bits are flipped.  If omitted, all sites are flipped.
    #[staticmethod]
    #[pyo3(signature = (grp_char, n_sites, locs=None))]
    pub fn bitflip(grp_char: Complex<f64>, n_sites: usize, locs: Option<Vec<usize>>) -> Self {
        PyGrpElement {
            grp_char,
            n_sites,
            op: GrpOpDesc::Bitflip { n_sites, locs },
        }
    }

    /// Local dit-value permutation symmetry.
    ///
    /// Args:
    ///   grp_char: Group character.
    ///   n_sites:  Number of sites in the system.
    ///   lhss:     Local Hilbert space size.
    ///   perm:     Value permutation: v → perm[v].  Length must equal lhss.
    ///   locs:     Sites to which the permutation is applied.
    #[staticmethod]
    pub fn local_value(
        grp_char: Complex<f64>,
        n_sites: usize,
        lhss: usize,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) -> Self {
        PyGrpElement {
            grp_char,
            n_sites,
            op: GrpOpDesc::LocalValue {
                n_sites,
                lhss,
                perm,
                locs,
            },
        }
    }

    /// Spin-inversion symmetry: v → lhss − v − 1 at the given sites.
    ///
    /// Args:
    ///   grp_char: Group character.
    ///   n_sites:  Number of sites in the system.
    ///   lhss:     Local Hilbert space size.
    ///   locs:     Sites to which the inversion is applied.
    #[staticmethod]
    pub fn spin_inversion(
        grp_char: Complex<f64>,
        n_sites: usize,
        lhss: usize,
        locs: Vec<usize>,
    ) -> Self {
        PyGrpElement {
            grp_char,
            n_sites,
            op: GrpOpDesc::SpinInversion {
                n_sites,
                lhss,
                locs,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// PySymmetryGrp
// ---------------------------------------------------------------------------

/// A symmetry group: a product of lattice and local group elements.
///
/// The concrete basis integer type `B` is selected at construction time
/// based on `n_sites`, and the group is stored as a `SymmetryGrpInner`.
#[pyclass(name = "PySymmetryGrp")]
pub struct PySymmetryGrp {
    pub inner: SymmetryGrpInner,
}

#[pymethods]
impl PySymmetryGrp {
    /// Construct from lists of `PyLatticeElement` and `PyGrpElement` objects.
    ///
    /// Validates that all elements agree on `n_sites`.
    /// Selects the concrete `B` type based on `n_sites` and eagerly builds
    /// the `SymmetryGrp<B>`, stored as a `SymmetryGrpInner`.
    #[new]
    pub fn new(
        py: Python<'_>,
        lattice: &Bound<'_, PyList>,
        local: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        let mut n_sites_opt: Option<usize> = None;

        let mut check = |n: usize| -> PyResult<()> {
            match n_sites_opt {
                None => {
                    n_sites_opt = Some(n);
                    Ok(())
                }
                Some(existing) if existing != n => {
                    Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "n_sites mismatch in symmetry group: element has {n} but expected {existing}"
                    )))
                }
                _ => Ok(()),
            }
        };

        let mut lattice_elements: Vec<LatticeElement> = Vec::with_capacity(lattice.len());
        for item in lattice.iter() {
            let el: PyRef<'_, PyLatticeElement> = item.extract()?;
            check(el.perm.len())?;
            lattice_elements.push(el.to_lattice_element());
        }

        // Collect local elements as (grp_char, GrpOpDesc) for deferred B-dispatch.
        let mut local_descs: Vec<(Complex<f64>, GrpOpDesc)> = Vec::with_capacity(local.len());
        for item in local.iter() {
            let el: PyRef<'_, PyGrpElement> = item.extract()?;
            check(el.n_sites)?;
            local_descs.push((el.grp_char, el.op.clone()));
        }

        let _ = py;
        let n_sites = n_sites_opt.unwrap_or(0);

        // Select B and build SymmetryGrpInner.
        macro_rules! build_inner {
            ($B:ty, $from_grp:ident) => {{
                let local_elements: Vec<GrpElement<$B>> = local_descs
                    .into_iter()
                    .map(|(char_, op)| op.into_grp_element::<$B>(char_))
                    .collect();
                let grp = SymmetryGrp::<$B>::new(lattice_elements, local_elements)
                    .map_err(|e| PyErr::from(Error(e)))?;
                SymmetryGrpInner::$from_grp(grp)
            }};
        }

        let inner = if n_sites <= 32 {
            build_inner!(u32, from_grp_32)
        } else if n_sites <= 64 {
            build_inner!(u64, from_grp_64)
        } else if n_sites <= 128 {
            build_inner!(ruint::Uint<128, 2>, from_grp_128)
        } else if n_sites <= 256 {
            build_inner!(ruint::Uint<256, 4>, from_grp_256)
        } else if n_sites <= 512 {
            build_inner!(ruint::Uint<512, 8>, from_grp_512)
        } else if n_sites <= 1024 {
            build_inner!(ruint::Uint<1024, 16>, from_grp_1024)
        } else if n_sites <= 2048 {
            build_inner!(ruint::Uint<2048, 32>, from_grp_2048)
        } else if n_sites <= 4096 {
            build_inner!(ruint::Uint<4096, 64>, from_grp_4096)
        } else if n_sites <= 8192 {
            build_inner!(ruint::Uint<8192, 128>, from_grp_8192)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites={n_sites} exceeds the maximum supported value of 8192"
            )));
        };

        Ok(PySymmetryGrp { inner })
    }

    /// Number of sites in the system.
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }
}
