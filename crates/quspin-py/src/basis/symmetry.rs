/// Python-facing symmetry group pyclasses.
///
/// Mirrors the Rust builder API directly:
///
/// ```python
/// grp = PySpinSymGrp(lhss=2, n_sites=4)
/// grp.add_lattice(grp_char=1+0j, perm=[1, 2, 3, 0])
/// grp.add_inverse(grp_char=-1+0j, locs=[0, 1, 2, 3])
///
/// grp2 = PyDitSymGrp(lhss=3, n_sites=4)
/// grp2.add_lattice(grp_char=1+0j, perm=[1, 2, 3, 0])
/// grp2.add_local_perm(grp_char=1+0j, perm=[2, 1, 0], locs=[0, 1, 2, 3])
///
/// grp3 = PyFermionicSymGrp(n_sites=4)
/// grp3.add_lattice(grp_char=1+0j, perm=[1, 2, 3, 0])
/// ```
use num_complex::Complex;
use pyo3::prelude::*;
use quspin_core::basis::{DitSymGrp, FermionicSymGrp, SpinSymGrp};

use crate::error::Error;

// ---------------------------------------------------------------------------
// PySpinSymGrp
// ---------------------------------------------------------------------------

/// A spin-symmetry group: lattice permutations + spin-inversion / bit-flip ops.
///
/// For LHSS = 2: local operations are XOR bit-flips (Z₂ symmetry).
/// For LHSS > 2: local operations map ``v → lhss − v − 1`` (spin inversion).
///
/// Use :class:`PyDitSymGrp` for local value-permutation symmetries.
/// Mixing both op types in the same group is not supported.
#[pyclass(name = "PySpinSymGrp")]
pub struct PySpinSymGrp {
    pub inner: SpinSymGrp,
}

#[pymethods]
impl PySpinSymGrp {
    /// Construct an empty spin-symmetry group.
    ///
    /// Args:
    ///     lhss (int): Local Hilbert-space size (e.g. ``2`` for spin-1/2,
    ///         ``3`` for spin-1).
    ///     n_sites (int): Number of lattice sites.
    ///
    /// Raises:
    ///     ValueError: If ``lhss = 2`` and ``n_sites > 8192``.
    #[new]
    pub fn new(lhss: usize, n_sites: usize) -> PyResult<Self> {
        let inner = SpinSymGrp::new(lhss, n_sites).map_err(|e| PyErr::from(Error(e)))?;
        Ok(PySpinSymGrp { inner })
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry
    ///         operator, e.g. ``1+0j`` for even parity, ``-1+0j`` for odd).
    ///     perm (list[int]): Forward site permutation where ``perm[src] = dst``.
    ///         Must have length ``n_sites``.
    pub fn add_lattice(&mut self, grp_char: Complex<f64>, perm: Vec<usize>) {
        self.inner.add_lattice(grp_char, perm);
    }

    /// Add a spin-inversion / bit-flip symmetry element.
    ///
    /// For LHSS = 2: XOR-flips the bits at the specified site indices.
    /// For LHSS > 2: maps ``v → lhss − v − 1`` at the specified sites.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry
    ///         operator).
    ///     locs (list[int]): Site indices to which the operation is applied.
    pub fn add_inverse(&mut self, grp_char: Complex<f64>, locs: Vec<usize>) {
        self.inner.add_inverse(grp_char, locs);
    }

    /// Number of lattice sites.
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
            "PySpinSymGrp(lhss={}, n_sites={})",
            self.inner.lhss(),
            self.inner.n_sites(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyDitSymGrp
// ---------------------------------------------------------------------------

/// A dit symmetry group: lattice permutations + local value-permutation ops.
///
/// Only supported for LHSS ≥ 3. Use :class:`PySpinSymGrp` for LHSS = 2 or for
/// spin-inversion symmetries (``v → lhss − v − 1``).
///
/// Mixing value-permutation and spin-inversion ops in the same group is not
/// supported because the orbit computation would be incomplete.
#[pyclass(name = "PyDitSymGrp")]
pub struct PyDitSymGrp {
    pub inner: DitSymGrp,
}

#[pymethods]
impl PyDitSymGrp {
    /// Construct an empty dit symmetry group.
    ///
    /// Args:
    ///     lhss (int): Local Hilbert-space size. Must be ≥ 3.
    ///     n_sites (int): Number of lattice sites.
    ///
    /// Raises:
    ///     ValueError: If ``lhss < 3`` (use :class:`PySpinSymGrp` instead).
    #[new]
    pub fn new(lhss: usize, n_sites: usize) -> PyResult<Self> {
        let inner = DitSymGrp::new(lhss, n_sites).map_err(|e| PyErr::from(Error(e)))?;
        Ok(PyDitSymGrp { inner })
    }

    /// Add a lattice (site-permutation) symmetry element.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry
    ///         operator, e.g. ``1+0j`` for even parity, ``-1+0j`` for odd).
    ///     perm (list[int]): Forward site permutation where ``perm[src] = dst``.
    ///         Must have length ``n_sites``.
    pub fn add_lattice(&mut self, grp_char: Complex<f64>, perm: Vec<usize>) {
        self.inner.add_lattice(grp_char, perm);
    }

    /// Add an on-site value-permutation symmetry element.
    ///
    /// Maps local occupation ``v → perm[v]`` at each site in ``locs``.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry
    ///         operator).
    ///     perm (list[int]): Value permutation of length ``lhss``.
    ///         Entry ``i`` is the image of dit value ``i``.
    ///     locs (list[int]): Site indices to which the permutation is applied.
    pub fn add_local_perm(&mut self, grp_char: Complex<f64>, perm: Vec<u8>, locs: Vec<usize>) {
        self.inner.add_local_perm(grp_char, perm, locs);
    }

    /// Number of lattice sites.
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
            "PyDitSymGrp(lhss={}, n_sites={})",
            self.inner.lhss(),
            self.inner.n_sites(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyFermionicSymGrp
// ---------------------------------------------------------------------------

/// A fermionic symmetry group: lattice permutations with Jordan-Wigner sign
/// tracking.
///
/// All lattice elements automatically include the fermionic permutation sign
/// based on the pre-image state, implementing the Jordan-Wigner transformation
/// for site-permutation symmetries.
///
/// Use :class:`PySpinSymGrp` for bosonic systems.
///
/// Example:
///     >>> grp = PyFermionicSymGrp(n_sites=4)
///     >>> grp.add_lattice(grp_char=1.0+0j, perm=[1, 2, 3, 0])
///     >>> grp.n_sites
///     4
#[pyclass(name = "PyFermionicSymGrp")]
pub struct PyFermionicSymGrp {
    pub inner: FermionicSymGrp,
}

#[pymethods]
impl PyFermionicSymGrp {
    /// Construct an empty fermionic symmetry group.
    ///
    /// Args:
    ///     n_sites (int): Number of lattice sites. Maximum value is 8192.
    ///
    /// Raises:
    ///     ValueError: If ``n_sites > 8192``.
    #[new]
    pub fn new(n_sites: usize) -> PyResult<Self> {
        let inner = FermionicSymGrp::new(n_sites).map_err(|e| PyErr::from(Error(e)))?;
        Ok(PyFermionicSymGrp { inner })
    }

    /// Add a lattice (site-permutation) symmetry element with fermionic sign
    /// tracking.
    ///
    /// The Jordan-Wigner sign of the permutation acting on the pre-image state
    /// is automatically included in the group character.
    ///
    /// Args:
    ///     grp_char (complex): Group character (eigenvalue of the symmetry
    ///         operator, e.g. ``1+0j`` for even parity, ``-1+0j`` for odd).
    ///     perm (list[int]): Forward site permutation where ``perm[src] = dst``.
    ///         Must have length ``n_sites``.
    pub fn add_lattice(&mut self, grp_char: Complex<f64>, perm: Vec<usize>) {
        self.inner.add_lattice(grp_char, perm);
    }

    /// Number of lattice sites.
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }

    pub fn __repr__(&self) -> String {
        format!("PyFermionicSymGrp(n_sites={})", self.inner.n_sites())
    }
}
