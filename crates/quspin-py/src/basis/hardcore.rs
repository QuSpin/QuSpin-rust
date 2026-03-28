/// Python-facing `PyHardcoreBasis` pyclass.
use crate::error::Error;
use crate::hamiltonian::{PyFermionHamiltonian, PyHardcoreHamiltonian};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use quspin_core::basis::dispatch::BasisInner;
use quspin_core::basis::{
    seed_from_bytes, seed_from_str,
    space::{FullSpace, Subspace},
};
use quspin_core::hamiltonian::fermion::dispatch::FermionHamiltonianInner;
use quspin_core::hamiltonian::hardcore::dispatch::HardcoreHamiltonianInner;

use super::symmetry::{PyFermionicSymGrp, PySpinSymGrp};

// ---------------------------------------------------------------------------
// PyHardcoreBasis
// ---------------------------------------------------------------------------

#[pyclass(name = "PyHardcoreBasis")]
pub struct PyHardcoreBasis {
    pub inner: BasisInner,
}

#[pymethods]
impl PyHardcoreBasis {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Build the full Hilbert space of ``n_sites`` spin-1/2 sites.
    ///
    /// Contains all 2^n_sites computational basis states.
    /// Only supported for ``n_sites ≤ 64``.
    ///
    /// Args:
    ///     n_sites (int): Number of lattice sites. Maximum value is 64.
    ///
    /// Returns:
    ///     PyHardcoreBasis: Full-space basis with 2^n_sites states.
    ///
    /// Raises:
    ///     ValueError: If ``n_sites > 64``.
    #[staticmethod]
    pub fn full(n_sites: usize) -> PyResult<Self> {
        if n_sites > 64 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites={n_sites} exceeds 64; full Hilbert spaces beyond 2^64 are not supported"
            )));
        }
        let dim = 1usize << n_sites;
        let inner = if n_sites <= 32 {
            BasisInner::Full32(FullSpace::new(n_sites, dim))
        } else {
            BasisInner::Full64(FullSpace::new(n_sites, dim))
        };
        Ok(PyHardcoreBasis { inner })
    }

    /// Build the subspace reachable from seed states under a Hamiltonian.
    ///
    /// Starting from each seed, repeatedly applies the Hamiltonian to discover
    /// all connected basis states (e.g., a fixed-particle-number sector).
    ///
    /// Args:
    ///     seeds (Iterable[str | list[int]]): Initial states. Each element is
    ///         either a ``str`` of ``'0'``/``'1'`` characters or a ``list[int]``
    ///         of ``0``/``1`` values. Position ``i`` gives the occupation of
    ///         site ``i``.
    ///     ham (PyHardcoreHamiltonian): The Hamiltonian whose connectivity
    ///         defines the sector.
    ///
    /// Returns:
    ///     PyHardcoreBasis: Subspace basis containing all states reachable
    ///     from any seed.
    ///
    /// Raises:
    ///     ValueError: If any seed contains invalid characters or values, or
    ///         if ``n_sites`` exceeds 8192.
    #[staticmethod]
    pub fn subspace(seeds: &Bound<'_, PyAny>, ham: &PyHardcoreHamiltonian) -> PyResult<Self> {
        let n_sites = ham.inner.max_site() + 1;
        let seed_list = extract_seed_list(seeds)?;

        let inner = quspin_core::select_b_for_n_sites!(
            n_sites,
            B,
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites={n_sites} exceeds the maximum supported value of 8192"
            ))),
            {
                let mut basis = Subspace::<B>::new(n_sites);
                for s in &seed_list {
                    let seed = seed_from_bytes::<B>(s);
                    match &ham.inner {
                        HardcoreHamiltonianInner::Ham8(h) => {
                            basis.build(seed, |state| h.apply_smallvec(state).into_iter());
                        }
                        HardcoreHamiltonianInner::Ham16(h) => {
                            basis.build(seed, |state| h.apply_smallvec(state).into_iter());
                        }
                    }
                }
                BasisInner::from(basis)
            }
        );
        Ok(PyHardcoreBasis { inner })
    }

    /// Build a symmetry-reduced subspace.
    ///
    /// Like ``subspace``, but projects into a symmetry sector defined by ``grp``,
    /// yielding a smaller basis.
    ///
    /// Args:
    ///     seeds (Iterable[str | list[int]]): Initial states (same format as
    ///         ``subspace``).
    ///     ham (PyHardcoreHamiltonian): The Hamiltonian defining connectivity.
    ///     grp (PySymmetryGrp): The symmetry group defining the sector.
    ///
    /// Returns:
    ///     PyHardcoreBasis: Symmetry-reduced basis.
    ///
    /// Raises:
    ///     ValueError: If ``ham.n_sites != grp.n_sites``, if any seed is
    ///         malformed, or if ``n_sites`` exceeds 8192.
    #[staticmethod]
    pub fn symmetric(
        seeds: &Bound<'_, PyAny>,
        ham: &PyHardcoreHamiltonian,
        grp: &PySpinSymGrp,
    ) -> PyResult<Self> {
        let n_sites = ham.inner.max_site() + 1;
        if grp.n_sites() != n_sites {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites mismatch: symmetry group has {} sites but Hamiltonian has {}",
                grp.n_sites(),
                n_sites
            )));
        }
        let seed_list = extract_seed_list(seeds)?;

        let hc = grp.inner.as_hardcore().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "symmetric basis requires a spin-symmetry group with LHSS=2",
            )
        })?;
        let inner = quspin_core::with_sym_grp!(hc, B, N, sym_grp, {
            let mut basis =
                quspin_core::basis::sym_basis::SymBasis::<B, _, N>::from_grp(sym_grp.clone());
            for s in &seed_list {
                let seed = seed_from_bytes::<B>(s);
                match &ham.inner {
                    HardcoreHamiltonianInner::Ham8(h) => {
                        basis.build(seed, |state| h.apply_smallvec(state).into_iter());
                    }
                    HardcoreHamiltonianInner::Ham16(h) => {
                        basis.build(seed, |state| h.apply_smallvec(state).into_iter());
                    }
                }
            }
            BasisInner::from(basis)
        });

        Ok(PyHardcoreBasis { inner })
    }

    /// Build a symmetry-reduced subspace for a fermionic system.
    ///
    /// Like :meth:`symmetric`, but accepts a :class:`PyFermionicSymGrp` whose
    /// lattice elements include Jordan-Wigner permutation signs, and a
    /// :class:`PyFermionHamiltonian` that carries Jordan-Wigner sign
    /// accumulation in each operator string.
    ///
    /// Args:
    ///     seeds (Iterable[str | list[int]]): Initial states (same format as
    ///         ``subspace``).
    ///     ham (PyFermionHamiltonian): The fermionic Hamiltonian defining
    ///         connectivity (Jordan-Wigner signs included).
    ///     grp (PyFermionicSymGrp): The fermionic symmetry group.
    ///
    /// Returns:
    ///     PyHardcoreBasis: Symmetry-reduced fermionic basis.
    ///
    /// Raises:
    ///     ValueError: If ``ham.n_sites != grp.n_sites``, if any seed is
    ///         malformed, or if ``n_sites`` exceeds 8192.
    #[staticmethod]
    pub fn symmetric_fermionic(
        seeds: &Bound<'_, PyAny>,
        ham: &PyFermionHamiltonian,
        grp: &PyFermionicSymGrp,
    ) -> PyResult<Self> {
        let n_sites = ham.inner.max_site() + 1;
        if grp.n_sites() != n_sites {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites mismatch: fermionic symmetry group has {} sites but Hamiltonian has {}",
                grp.n_sites(),
                n_sites
            )));
        }
        let seed_list = extract_seed_list(seeds)?;

        let hc = grp.inner.as_hardcore().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "symmetric fermionic basis requires a fermionic symmetry group",
            )
        })?;
        let inner = quspin_core::with_sym_grp!(hc, B, N, sym_grp, {
            let mut basis =
                quspin_core::basis::sym_basis::SymBasis::<B, _, N>::from_grp(sym_grp.clone());
            for s in &seed_list {
                let seed = seed_from_bytes::<B>(s);
                match &ham.inner {
                    FermionHamiltonianInner::Ham8(h) => {
                        basis.build(seed, |state| h.apply_smallvec(state).into_iter());
                    }
                    FermionHamiltonianInner::Ham16(h) => {
                        basis.build(seed, |state| h.apply_smallvec(state).into_iter());
                    }
                }
            }
            BasisInner::from(basis)
        });

        Ok(PyHardcoreBasis { inner })
    }

    // ------------------------------------------------------------------
    // State access
    // ------------------------------------------------------------------

    /// Return the ``i``-th basis state as a bit string.
    ///
    /// Character at position ``j`` is ``'1'`` if site ``j`` is occupied,
    /// ``'0'`` otherwise.  The ordering matches the seed convention used in
    /// :meth:`subspace` and :meth:`symmetric`.
    ///
    /// Args:
    ///     i (int): Row index, ``0 ≤ i < size``.
    ///
    /// Returns:
    ///     str: Bit string of length ``n_sites``.
    ///
    /// Raises:
    ///     IndexError: If ``i`` is out of range.
    pub fn state_at(&self, i: usize) -> PyResult<String> {
        if i >= self.inner.size() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {i} out of range for basis of size {}",
                self.inner.size()
            )));
        }
        Ok(self.inner.state_at_str(i))
    }

    /// Look up the index of a basis state.
    ///
    /// Args:
    ///     state (str | list[int]): Basis state in the same format accepted by
    ///         :meth:`subspace` — a ``'0'``/``'1'`` string or a list of ``0``/``1``
    ///         integers, where position ``j`` gives the occupation of site ``j``.
    ///
    /// Returns:
    ///     int | None: The row index of ``state`` in the basis, or ``None`` if
    ///     the state is not present.
    ///
    /// Raises:
    ///     ValueError: If ``state`` is malformed.
    pub fn index(&self, state: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let bytes = extract_seed(state)?;
        Ok(self.inner.index_of_bytes(&bytes))
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    /// Number of sites.
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.inner.n_sites()
    }

    /// Number of basis states.
    #[getter]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyHardcoreBasis(kind={}, n_sites={}, size={})",
            self.inner.kind(),
            self.inner.n_sites(),
            self.inner.size(),
        )
    }

    pub fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a Python iterable of seed states as a `Vec<Vec<u8>>`.
///
/// Each seed must be either:
/// - a `str` of `'0'`/`'1'` characters — index `i` = site `i`, e.g. `"1100"`
/// - a `list[int]` of `0`/`1` values — same convention, e.g. `[1, 1, 0, 0]`
///
/// Both representations are site-indexed: position 0 = site 0 (LSB).
fn extract_seed_list(seeds: &Bound<'_, PyAny>) -> Result<Vec<Vec<u8>>, Error> {
    seeds
        .try_iter()
        .map_err(|_| {
            Error(quspin_core::error::QuSpinError::ValueError(
                "seeds must be an iterable".to_string(),
            ))
        })?
        .map(|item| {
            item.map_err(|e| Error(quspin_core::error::QuSpinError::ValueError(e.to_string())))
                .and_then(|obj| extract_seed(&obj))
        })
        .collect()
}

/// Convert a single Python seed (str or list[int]) to a site-occupation byte vector.
///
/// String seeds are parsed via `quspin_core::basis::seed_from_str`.
/// List seeds are validated and returned directly.
fn extract_seed(obj: &Bound<'_, PyAny>) -> Result<Vec<u8>, Error> {
    if let Ok(s) = obj.extract::<String>() {
        seed_from_str(&s).map_err(Error)
    } else if let Ok(bits) = obj.extract::<Vec<u8>>() {
        for &v in &bits {
            if v > 1 {
                return Err(Error(quspin_core::error::QuSpinError::ValueError(
                    "each element in a seed list must be 0 or 1".to_string(),
                )));
            }
        }
        Ok(bits)
    } else {
        Err(Error(quspin_core::error::QuSpinError::ValueError(
            "each seed must be a str of '0'/'1' or a list of 0/1 ints".to_string(),
        )))
    }
}
