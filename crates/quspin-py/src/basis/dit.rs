/// Python-facing `PyDitBasis` pyclass.
///
/// Wraps a `BasisInner` and an `lhss: usize` for bosonic (LHSS ≥ 2)
/// basis construction.  State integers are packed using `DynamicDitManip`:
/// each site occupies `BITS_TABLE[lhss]` bits.
///
/// Seed strings are decimal digit sequences, e.g. `"012"` for a 3-site
/// system with LHSS=3.  Seed lists are `list[int]` with values in `0..lhss`.
use crate::error::Error;
use crate::hamiltonian::boson::PyBosonHamiltonian;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use quspin_core::basis::hardcore::dispatch::BasisInner;
use quspin_core::basis::{
    BasisSpace, dit_seed_from_bytes, dit_seed_from_str, dit_state_to_str,
    space::{FullSpace, Subspace},
};
use quspin_core::bitbasis::manip::DynamicDitManip;
use quspin_core::hamiltonian::boson::dispatch::BosonHamiltonianInner;

// ---------------------------------------------------------------------------
// PyDitBasis
// ---------------------------------------------------------------------------

#[pyclass(name = "PyDitBasis")]
pub struct PyDitBasis {
    pub inner: BasisInner,
    pub lhss: usize,
    pub manip: DynamicDitManip,
}

#[pymethods]
impl PyDitBasis {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Build the full Hilbert space of ``n_sites`` bosonic sites with ``lhss``
    /// levels each.
    ///
    /// Contains all ``lhss^n_sites`` computational basis states.
    ///
    /// Args:
    ///     n_sites (int): Number of lattice sites.
    ///     lhss (int): Local Hilbert space size (levels per site). Must be ≥ 2.
    ///
    /// Returns:
    ///     PyDitBasis: Full-space basis with lhss^n_sites states.
    ///
    /// Raises:
    ///     ValueError: If ``lhss < 2``, or the total bit width exceeds 64.
    #[staticmethod]
    pub fn full(n_sites: usize, lhss: usize) -> PyResult<Self> {
        if lhss < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("lhss must be ≥ 2"));
        }
        let manip = DynamicDitManip::new(lhss);
        let total_bits = n_sites * manip.bits;
        if total_bits > 64 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "full dit basis requires n_sites * bits_per_site = {total_bits} ≤ 64; \
                 use subspace() for larger systems"
            )));
        }
        // dim = lhss^n_sites (computed without overflow for small cases)
        let dim = (lhss as u64).checked_pow(n_sites as u32).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "lhss^n_sites overflows u64; use subspace() instead",
            )
        })? as usize;

        let inner = if total_bits <= 32 {
            BasisInner::Full32(FullSpace::new(n_sites, dim))
        } else {
            BasisInner::Full64(FullSpace::new(n_sites, dim))
        };
        Ok(PyDitBasis { inner, lhss, manip })
    }

    /// Build the subspace reachable from seed states under a bosonic Hamiltonian.
    ///
    /// Args:
    ///     seeds (Iterable[str | list[int]]): Initial states. Each element is
    ///         either a decimal digit string (e.g. ``"012"``) or a ``list[int]``
    ///         with values in ``0..lhss``.
    ///     ham (PyBosonHamiltonian): The Hamiltonian whose connectivity defines
    ///         the sector.
    ///
    /// Returns:
    ///     PyDitBasis: Subspace basis.
    ///
    /// Raises:
    ///     ValueError: If any seed is malformed or the total bits exceed 8192.
    #[staticmethod]
    pub fn subspace(seeds: &Bound<'_, PyAny>, ham: &PyBosonHamiltonian) -> PyResult<Self> {
        let lhss = ham.inner.lhss();
        let n_sites = ham.inner.max_site() + 1;
        let manip = DynamicDitManip::new(lhss);
        let total_bits = n_sites * manip.bits;
        let seed_list = extract_dit_seed_list(seeds, &manip)?;

        let inner = quspin_core::select_b_for_n_sites!(
            total_bits,
            B,
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "total_bits={total_bits} exceeds the maximum supported value of 8192"
            ))),
            {
                let mut basis = Subspace::<B>::new(n_sites);
                for s in &seed_list {
                    let seed = dit_seed_from_bytes::<B>(s, &manip);
                    match &ham.inner {
                        BosonHamiltonianInner::Ham8(h) => {
                            basis.build(seed, |state| h.apply_smallvec(state).into_iter());
                        }
                        BosonHamiltonianInner::Ham16(h) => {
                            basis.build(seed, |state| h.apply_smallvec(state).into_iter());
                        }
                    }
                }
                BasisInner::from(basis)
            }
        );
        Ok(PyDitBasis { inner, lhss, manip })
    }

    // ------------------------------------------------------------------
    // State access
    // ------------------------------------------------------------------

    /// Return the ``i``-th basis state as a decimal digit string.
    ///
    /// Character at position ``j`` is the decimal digit for site ``j``'s
    /// occupation, e.g. ``'0'``, ``'1'``, …, ``'9'``.
    ///
    /// Args:
    ///     i (int): Row index, ``0 ≤ i < size``.
    ///
    /// Returns:
    ///     str: Decimal digit string of length ``n_sites``.
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
        Ok(self.state_at_str(i))
    }

    /// Look up the index of a dit basis state.
    ///
    /// Args:
    ///     state (str | list[int]): Basis state — a decimal digit string or a
    ///         ``list[int]`` with values in ``0..lhss``.
    ///
    /// Returns:
    ///     int | None: Row index, or ``None`` if the state is not present.
    ///
    /// Raises:
    ///     ValueError: If ``state`` is malformed.
    pub fn index(&self, state: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let bytes = extract_dit_seed(state, &self.manip)?;
        Ok(self.index_of_bytes(&bytes))
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

    /// Local Hilbert space size.
    #[getter]
    pub fn lhss(&self) -> usize {
        self.lhss
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyDitBasis(lhss={}, n_sites={}, size={})",
            self.lhss,
            self.inner.n_sites(),
            self.inner.size(),
        )
    }
}

impl PyDitBasis {
    /// Return the i-th state as a decimal digit string (internal helper).
    fn state_at_str(&self, i: usize) -> String {
        use quspin_core::with_basis;
        let n_sites = self.inner.n_sites();
        let manip = &self.manip;
        with_basis!(&self.inner, B, basis, {
            dit_state_to_str(basis.state_at(i), n_sites, manip)
        })
    }

    /// Look up the index of a state given as a byte slice (internal helper).
    fn index_of_bytes(&self, bytes: &[u8]) -> Option<usize> {
        use quspin_core::with_basis;
        let manip = &self.manip;
        with_basis!(&self.inner, B, basis, {
            let state = dit_seed_from_bytes::<B>(bytes, manip);
            basis.index(state)
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_dit_seed_list(
    seeds: &Bound<'_, PyAny>,
    manip: &DynamicDitManip,
) -> Result<Vec<Vec<u8>>, Error> {
    seeds
        .try_iter()
        .map_err(|_| {
            Error(quspin_core::error::QuSpinError::ValueError(
                "seeds must be an iterable".to_string(),
            ))
        })?
        .map(|item| {
            item.map_err(|e| Error(quspin_core::error::QuSpinError::ValueError(e.to_string())))
                .and_then(|obj| extract_dit_seed(&obj, manip))
        })
        .collect()
}

fn extract_dit_seed(obj: &Bound<'_, PyAny>, manip: &DynamicDitManip) -> Result<Vec<u8>, Error> {
    let lhss = manip.lhss;
    if let Ok(s) = obj.extract::<String>() {
        dit_seed_from_str(&s, lhss).map_err(Error)
    } else if let Ok(vals) = obj.extract::<Vec<usize>>() {
        for &v in &vals {
            if v >= lhss {
                return Err(Error(quspin_core::error::QuSpinError::ValueError(format!(
                    "dit seed value {v} out of range for lhss={lhss}"
                ))));
            }
        }
        Ok(vals.into_iter().map(|v| v as u8).collect())
    } else {
        Err(Error(quspin_core::error::QuSpinError::ValueError(format!(
            "each dit seed must be a decimal digit string or a list of ints in 0..{lhss}"
        ))))
    }
}
