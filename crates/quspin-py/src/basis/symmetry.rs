use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use quspin_core::basis::SymmetryGrpInner;
use quspin_core::basis::symmetry::{GrpElement, GrpOpKind, LatticeElement, SymmetryGrp};
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
use quspin_core::bitbasis::{
    BitInt, DynamicHigherSpinInv, DynamicPermDitValues, PermDitLocations, PermDitMask,
};

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
    /// Create a lattice symmetry element.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry
    ///         operator, e.g. ``1+0j`` for even parity, ``-1+0j`` for odd).
    ///     perm (list[int]): Forward site permutation where ``perm[src] = dst``.
    ///         The length of ``perm`` determines the number of sites.
    ///     lhss (int): Local Hilbert space size (e.g. ``2`` for spin-1/2,
    ///         ``3`` for spin-1).
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
    /// Create a Z₂ bit-flip (XOR-with-mask) symmetry element.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry operator).
    ///     n_sites (int): Total number of sites in the system.
    ///     locs (list[int] | None): Site indices whose bits are flipped.
    ///         If ``None``, all ``n_sites`` bits are flipped.
    ///
    /// Returns:
    ///     PyGrpElement: A new bit-flip symmetry element.
    #[staticmethod]
    #[pyo3(signature = (grp_char, n_sites, locs=None))]
    pub fn bitflip(grp_char: Complex<f64>, n_sites: usize, locs: Option<Vec<usize>>) -> Self {
        PyGrpElement {
            grp_char,
            n_sites,
            op: GrpOpDesc::Bitflip { n_sites, locs },
        }
    }

    /// Create a local dit-value permutation symmetry element.
    ///
    /// For each site in ``locs``, maps the local occupation value ``v → perm[v]``.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry operator).
    ///     n_sites (int): Total number of sites in the system.
    ///     lhss (int): Local Hilbert space size (number of distinct dit values).
    ///     perm (list[int]): Value permutation of length ``lhss``.
    ///         Entry ``i`` is the image of dit value ``i``.
    ///     locs (list[int]): Site indices to which the permutation is applied.
    ///
    /// Returns:
    ///     PyGrpElement: A new local-value permutation symmetry element.
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

    /// Create a spin-inversion symmetry element.
    ///
    /// Maps each dit value ``v → lhss - v - 1`` at the specified sites.
    /// For spin-1/2 (``lhss=2``) this swaps ``0 ↔ 1``; for spin-1 (``lhss=3``)
    /// it maps ``0 ↔ 2`` and fixes ``1``.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry operator).
    ///     n_sites (int): Total number of sites in the system.
    ///     lhss (int): Local Hilbert space size.
    ///     locs (list[int]): Site indices to which the inversion is applied.
    ///
    /// Returns:
    ///     PyGrpElement: A new spin-inversion symmetry element.
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
    /// Construct a symmetry group from lattice and local elements.
    ///
    /// Validates that all elements agree on ``n_sites``, then selects the
    /// concrete basis integer type and eagerly builds the typed symmetry group.
    ///
    /// Args:
    ///     lattice (list[PyLatticeElement]): Spatial symmetry elements (site
    ///         permutations). The permutation length determines ``n_sites``.
    ///     local (list[PyGrpElement]): On-site symmetry elements (bit-flip,
    ///         value permutation, or spin inversion).
    ///
    /// Raises:
    ///     ValueError: If any two elements disagree on the number of sites, or
    ///         if ``n_sites`` exceeds 8192.
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

        let inner = crate::select_b_for_n_sites!(n_sites, B, {
            let local_elements: Vec<GrpElement<B>> = local_descs
                .into_iter()
                .map(|(char_, op)| op.into_grp_element::<B>(char_))
                .collect();
            let grp = SymmetryGrp::<B>::new(lattice_elements, local_elements)
                .map_err(|e| PyErr::from(Error(e)))?;
            SymmetryGrpInner::from(grp)
        });

        Ok(PySymmetryGrp { inner })
    }

    /// Number of sites in the system.
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }
}
