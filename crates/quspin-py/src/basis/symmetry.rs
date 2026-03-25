/// Python-facing symmetry group pyclasses.
///
/// `PyLatticeElement` and `PyGrpElement` accept Python-level arguments.
/// `PySpinSymGrp` and `PyValuePermSymGrp` resolve both the concrete basis
/// integer type `B` (from `n_sites`) and the LHSS dispatch at construction
/// time.
///
/// ## Python API
///
/// ```python
/// T   = PyLatticeElement(grp_char=1.0+0j, perm=[1, 2, 3, 0], lhss=2)
/// P   = PyGrpElement.bitflip(grp_char=1.0+0j, n_sites=4)
/// grp = PySpinSymGrp(lattice=[T], local=[P])
///
/// T2  = PyLatticeElement(grp_char=1.0+0j, perm=[1, 2, 3, 0], lhss=3)
/// Q   = PyGrpElement.local_value(grp_char=1.0+0j, n_sites=4, lhss=3, perm=[2,1,0], locs=[0,1,2,3])
/// grp2 = PyValuePermSymGrp(lattice=[T2], local=[Q])
/// ```
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use quspin_core::basis::{SpinSymGrp, ValuePermSymGrp};

use crate::error::Error;

fn fmt_complex(c: Complex<f64>) -> String {
    if c.im >= 0.0 {
        format!("({}+{}j)", c.re, c.im)
    } else {
        format!("({}{}j)", c.re, c.im)
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

    pub fn __repr__(&self) -> String {
        let perm: Vec<String> = self.perm.iter().map(|x| x.to_string()).collect();
        format!(
            "PyLatticeElement(grp_char={}, perm=[{}], lhss={})",
            fmt_complex(self.grp_char),
            perm.join(", "),
            self.lhss,
        )
    }
}

// ---------------------------------------------------------------------------
// PyGrpElement
// ---------------------------------------------------------------------------

/// The kind of a local symmetry operation.
pub(crate) enum GrpElemKind {
    /// XOR with a fixed mask (Z₂ bit-flip, LHSS=2 only).
    Bitflip { locs: Option<Vec<usize>> },
    /// Uniform value permutation on a subset of sites (LHSS > 2).
    LocalValue {
        lhss: usize,
        perm: Vec<u8>,
        locs: Vec<usize>,
    },
    /// Spin inversion `v → lhss − v − 1` on a subset of sites.
    SpinInversion { lhss: usize, locs: Vec<usize> },
}

/// A local symmetry group element: a bit-operation with a group character.
#[pyclass(name = "PyGrpElement")]
pub struct PyGrpElement {
    pub grp_char: Complex<f64>,
    pub n_sites: usize,
    pub(crate) kind: GrpElemKind,
}

impl PyGrpElement {
    pub(crate) fn lhss(&self) -> usize {
        match &self.kind {
            GrpElemKind::Bitflip { .. } => 2,
            GrpElemKind::LocalValue { lhss, .. } => *lhss,
            GrpElemKind::SpinInversion { lhss, .. } => *lhss,
        }
    }
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
            kind: GrpElemKind::Bitflip { locs },
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
            kind: GrpElemKind::LocalValue { lhss, perm, locs },
        }
    }

    /// Create a spin-inversion symmetry element.
    ///
    /// Maps each dit value ``v → lhss - v - 1`` at the specified sites.
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
            kind: GrpElemKind::SpinInversion { lhss, locs },
        }
    }

    pub fn __repr__(&self) -> String {
        let c = fmt_complex(self.grp_char);
        match &self.kind {
            GrpElemKind::Bitflip { locs } => {
                let locs_str = match locs {
                    None => "None".to_string(),
                    Some(v) => {
                        let s: Vec<String> = v.iter().map(|x| x.to_string()).collect();
                        format!("[{}]", s.join(", "))
                    }
                };
                format!(
                    "PyGrpElement.bitflip(grp_char={c}, n_sites={}, locs={locs_str})",
                    self.n_sites,
                )
            }
            GrpElemKind::LocalValue { lhss, perm, locs } => {
                let perm_s: Vec<String> = perm.iter().map(|x| x.to_string()).collect();
                let locs_s: Vec<String> = locs.iter().map(|x| x.to_string()).collect();
                format!(
                    "PyGrpElement.local_value(grp_char={c}, n_sites={}, lhss={lhss}, perm=[{}], locs=[{}])",
                    self.n_sites,
                    perm_s.join(", "),
                    locs_s.join(", "),
                )
            }
            GrpElemKind::SpinInversion { lhss, locs } => {
                let locs_s: Vec<String> = locs.iter().map(|x| x.to_string()).collect();
                format!(
                    "PyGrpElement.spin_inversion(grp_char={c}, n_sites={}, lhss={lhss}, locs=[{}])",
                    self.n_sites,
                    locs_s.join(", "),
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PySpinSymGrp
// ---------------------------------------------------------------------------

/// A spin-symmetry group: lattice permutations + spin-inversion / bit-flip ops.
///
/// Accepts only `PyGrpElement.bitflip` and `PyGrpElement.spin_inversion`
/// local elements. For local value-permutation symmetries use
/// :class:`PyValuePermSymGrp`.
#[pyclass(name = "PySpinSymGrp")]
pub struct PySpinSymGrp {
    pub inner: SpinSymGrp,
}

#[pymethods]
impl PySpinSymGrp {
    /// Construct a spin-symmetry group from lattice and local elements.
    ///
    /// Infers ``n_sites`` and ``lhss`` from the supplied elements and validates
    /// that all elements agree.
    ///
    /// Args:
    ///     lattice (list[PyLatticeElement]): Spatial symmetry elements (site
    ///         permutations). The permutation length determines ``n_sites``.
    ///     local (list[PyGrpElement]): On-site symmetry elements. Only
    ///         ``bitflip`` and ``spin_inversion`` elements are accepted.
    ///
    /// Raises:
    ///     ValueError: If any two elements disagree on ``n_sites`` or ``lhss``,
    ///         if a ``local_value`` element is supplied (use
    ///         :class:`PyValuePermSymGrp` instead), or if ``n_sites`` exceeds
    ///         8192.
    #[new]
    pub fn new(
        py: Python<'_>,
        lattice: &Bound<'_, PyList>,
        local: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        let _ = py;

        let mut n_sites_opt: Option<usize> = None;
        let mut lhss_opt: Option<usize> = None;

        let mut check_n_sites = |n: usize| -> PyResult<()> {
            match n_sites_opt {
                None => {
                    n_sites_opt = Some(n);
                    Ok(())
                }
                Some(existing) if existing != n => Err(pyo3::exceptions::PyValueError::new_err(
                    format!("n_sites mismatch: element has {n} but expected {existing}"),
                )),
                _ => Ok(()),
            }
        };

        let mut check_lhss = |l: usize| -> PyResult<()> {
            match lhss_opt {
                None => {
                    lhss_opt = Some(l);
                    Ok(())
                }
                Some(existing) if existing != l => Err(pyo3::exceptions::PyValueError::new_err(
                    format!("lhss mismatch: element has {l} but expected {existing}"),
                )),
                _ => Ok(()),
            }
        };

        let mut lattice_els: Vec<(Complex<f64>, Vec<usize>)> = Vec::with_capacity(lattice.len());
        for item in lattice.iter() {
            let el: PyRef<'_, PyLatticeElement> = item.extract()?;
            check_n_sites(el.perm.len())?;
            check_lhss(el.lhss)?;
            lattice_els.push((el.grp_char, el.perm.clone()));
        }

        let mut local_els: Vec<PyRef<'_, PyGrpElement>> = Vec::with_capacity(local.len());
        for item in local.iter() {
            let el: PyRef<'_, PyGrpElement> = item.extract()?;
            // Reject local_value elements — they belong in PyValuePermSymGrp.
            if matches!(el.kind, GrpElemKind::LocalValue { .. }) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "local_value elements are not allowed in PySpinSymGrp; \
                     use PyValuePermSymGrp for value-permutation symmetries",
                ));
            }
            check_n_sites(el.n_sites)?;
            check_lhss(el.lhss())?;
            local_els.push(el);
        }

        let n_sites = n_sites_opt.unwrap_or(0);
        let lhss = lhss_opt.unwrap_or(2);

        let mut grp = SpinSymGrp::new(lhss, n_sites).map_err(|e| PyErr::from(Error(e)))?;

        for (grp_char, perm) in lattice_els {
            grp.add_lattice(grp_char, perm);
        }

        for el in local_els {
            match &el.kind {
                GrpElemKind::Bitflip { locs } => {
                    let effective: Vec<usize> = match locs {
                        Some(l) => l.clone(),
                        None => (0..n_sites).collect(),
                    };
                    grp.add_local_inv(el.grp_char, effective);
                }
                GrpElemKind::SpinInversion { locs, .. } => {
                    grp.add_local_inv(el.grp_char, locs.clone());
                }
                GrpElemKind::LocalValue { .. } => unreachable!(),
            }
        }

        Ok(PySpinSymGrp { inner: grp })
    }

    /// Number of sites in the system.
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }

    /// Local Hilbert-space size.
    #[getter]
    pub fn lhss(&self) -> usize {
        self.inner.lhss()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PySpinSymGrp(n_sites={}, lhss={})",
            self.inner.n_sites(),
            self.inner.lhss(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyValuePermSymGrp
// ---------------------------------------------------------------------------

/// A value-permutation symmetry group: lattice permutations + local value-perm ops.
///
/// Only supported for LHSS > 2. Accepts only `PyGrpElement.local_value`
/// local elements. For spin-inversion or bit-flip symmetries use
/// :class:`PySpinSymGrp`.
#[pyclass(name = "PyValuePermSymGrp")]
pub struct PyValuePermSymGrp {
    pub inner: ValuePermSymGrp,
}

#[pymethods]
impl PyValuePermSymGrp {
    /// Construct a value-permutation symmetry group from lattice and local elements.
    ///
    /// Infers ``n_sites`` and ``lhss`` from the supplied elements and validates
    /// that all elements agree.
    ///
    /// Args:
    ///     lattice (list[PyLatticeElement]): Spatial symmetry elements (site
    ///         permutations). The permutation length determines ``n_sites``.
    ///     local (list[PyGrpElement]): On-site symmetry elements. Only
    ///         ``local_value`` elements are accepted.
    ///
    /// Raises:
    ///     ValueError: If ``lhss < 3`` (use :class:`PySpinSymGrp` for
    ///         ``lhss = 2``), if any two elements disagree on ``n_sites`` or
    ///         ``lhss``, or if a ``bitflip`` / ``spin_inversion`` element is
    ///         supplied.
    #[new]
    pub fn new(
        py: Python<'_>,
        lattice: &Bound<'_, PyList>,
        local: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        let _ = py;

        let mut n_sites_opt: Option<usize> = None;
        let mut lhss_opt: Option<usize> = None;

        let mut check_n_sites = |n: usize| -> PyResult<()> {
            match n_sites_opt {
                None => {
                    n_sites_opt = Some(n);
                    Ok(())
                }
                Some(existing) if existing != n => Err(pyo3::exceptions::PyValueError::new_err(
                    format!("n_sites mismatch: element has {n} but expected {existing}"),
                )),
                _ => Ok(()),
            }
        };

        let mut check_lhss = |l: usize| -> PyResult<()> {
            match lhss_opt {
                None => {
                    lhss_opt = Some(l);
                    Ok(())
                }
                Some(existing) if existing != l => Err(pyo3::exceptions::PyValueError::new_err(
                    format!("lhss mismatch: element has {l} but expected {existing}"),
                )),
                _ => Ok(()),
            }
        };

        let mut lattice_els: Vec<(Complex<f64>, Vec<usize>)> = Vec::with_capacity(lattice.len());
        for item in lattice.iter() {
            let el: PyRef<'_, PyLatticeElement> = item.extract()?;
            check_n_sites(el.perm.len())?;
            check_lhss(el.lhss)?;
            lattice_els.push((el.grp_char, el.perm.clone()));
        }

        let mut local_els: Vec<PyRef<'_, PyGrpElement>> = Vec::with_capacity(local.len());
        for item in local.iter() {
            let el: PyRef<'_, PyGrpElement> = item.extract()?;
            // Reject spin-type elements — they belong in PySpinSymGrp.
            match &el.kind {
                GrpElemKind::Bitflip { .. } | GrpElemKind::SpinInversion { .. } => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "bitflip and spin_inversion elements are not allowed in \
                         PyValuePermSymGrp; use PySpinSymGrp instead",
                    ));
                }
                GrpElemKind::LocalValue { .. } => {}
            }
            check_n_sites(el.n_sites)?;
            check_lhss(el.lhss())?;
            local_els.push(el);
        }

        let n_sites = n_sites_opt.unwrap_or(0);
        let lhss = lhss_opt.unwrap_or(3);

        let mut grp = ValuePermSymGrp::new(lhss, n_sites).map_err(|e| PyErr::from(Error(e)))?;

        for (grp_char, perm) in lattice_els {
            grp.add_lattice(grp_char, perm);
        }

        for el in local_els {
            match &el.kind {
                GrpElemKind::LocalValue { perm, locs, .. } => {
                    grp.add_local_perm(el.grp_char, perm.clone(), locs.clone());
                }
                _ => unreachable!(),
            }
        }

        Ok(PyValuePermSymGrp { inner: grp })
    }

    /// Number of sites in the system.
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }

    /// Local Hilbert-space size.
    #[getter]
    pub fn lhss(&self) -> usize {
        self.inner.lhss()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyValuePermSymGrp(n_sites={}, lhss={})",
            self.inner.n_sites(),
            self.inner.lhss(),
        )
    }
}
