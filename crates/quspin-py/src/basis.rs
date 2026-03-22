/// Python-facing `PyHardcoreBasis` pyclass.
///
/// Wraps `HardcoreBasisInner` and selects the concrete basis integer type `B`
/// based on `n_sites`:
///
/// | `n_sites` | `B` type                |
/// |-----------|-------------------------|
/// | ≤ 32      | `u32`                   |
/// | ≤ 64      | `u64`                   |
/// | ≤ 128     | `Uint<128, 2>`          |
/// | ≤ 256     | `Uint<256, 4>`          |
/// | ≤ 512     | `Uint<512, 8>`          |
/// | ≤ 1024    | `Uint<1024, 16>`        |
/// | ≤ 2048    | `Uint<2048, 32>`        |
/// | ≤ 4096    | `Uint<4096, 64>`        |
/// | ≤ 8192    | `Uint<8192, 128>`       |
///
/// ## Python API
///
/// ```python
/// # Full Hilbert space
/// basis = PyHardcoreBasis.full(n_sites=4)
///
/// # Particle-number (or energy) subspace
/// basis = PyHardcoreBasis.subspace(seeds=[0b0111, 0b1011], ham=H)
///
/// # Symmetry-reduced subspace
/// basis = PyHardcoreBasis.symmetric(seeds=[0b0111], ham=H, grp=grp)
/// ```
use bitbasis::BitInt;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use quspin_core::basis::{
    space::{FullSpace, Subspace},
    sym::SymmetricSubspace,
};

use crate::dispatch::HardcoreBasisInner;
use crate::error::Error;
use crate::hamiltonian::{PauliHamiltonianInner, PyPauliHamiltonian};
use crate::symmetry::PySymmetryGrp;

// ---------------------------------------------------------------------------
// PyHardcoreBasis
// ---------------------------------------------------------------------------

#[pyclass(name = "PyHardcoreBasis")]
pub struct PyHardcoreBasis {
    pub inner: HardcoreBasisInner,
}

#[pymethods]
impl PyHardcoreBasis {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Full Hilbert space of `n_sites` spin-1/2 sites.
    ///
    /// Uses `u32` for n_sites ≤ 32 and `u64` for n_sites ≤ 64.
    /// Larger full spaces are not supported (2^n_sites states is impractical).
    #[staticmethod]
    pub fn full(n_sites: usize) -> PyResult<Self> {
        if n_sites > 64 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites={n_sites} exceeds 64; full Hilbert spaces beyond 2^64 are not supported"
            )));
        }
        let dim = 1usize << n_sites;
        let inner = if n_sites <= 32 {
            HardcoreBasisInner::Full32(FullSpace::new(dim))
        } else {
            HardcoreBasisInner::Full64(FullSpace::new(dim))
        };
        Ok(PyHardcoreBasis { inner })
    }

    /// Subspace reachable from each seed state under the Hamiltonian.
    ///
    /// Args:
    ///   seeds: Python list of integer seed states.
    ///   ham:   The Hamiltonian defining connectivity.
    #[staticmethod]
    pub fn subspace(seeds: &Bound<'_, PyAny>, ham: &PyPauliHamiltonian) -> PyResult<Self> {
        let n_sites = ham.inner.n_sites();
        let seed_list = extract_seed_list(seeds)?;

        macro_rules! build_subspace {
            ($B:ty, $inner_variant:ident) => {{
                let mut basis = Subspace::<$B>::new();
                for s in &seed_list {
                    let seed = seed_as::<$B>(*s);
                    match &ham.inner {
                        PauliHamiltonianInner::Ham8(h) => {
                            basis.build(seed, |state| h.apply(state).into_iter());
                        }
                        PauliHamiltonianInner::Ham16(h) => {
                            basis.build(seed, |state| h.apply(state).into_iter());
                        }
                    }
                }
                HardcoreBasisInner::$inner_variant(basis)
            }};
        }

        let inner = select_subspace_type!(n_sites, build_subspace);
        Ok(PyHardcoreBasis { inner })
    }

    /// Symmetry-reduced subspace reachable from each seed state.
    ///
    /// Args:
    ///   seeds: Python list of integer seed states.
    ///   ham:   The Hamiltonian defining connectivity.
    ///   grp:   The symmetry group.
    #[staticmethod]
    pub fn symmetric(
        seeds: &Bound<'_, PyAny>,
        ham: &PyPauliHamiltonian,
        grp: &PySymmetryGrp,
    ) -> PyResult<Self> {
        let n_sites = ham.inner.n_sites();
        let seed_list = extract_seed_list(seeds)?;

        macro_rules! build_symmetric {
            ($B:ty, $inner_variant:ident) => {{
                let sym_grp = grp.into_symmetry_grp::<$B>();
                let mut basis = SymmetricSubspace::<$B>::new(sym_grp);
                for s in &seed_list {
                    let seed = seed_as::<$B>(*s);
                    match &ham.inner {
                        PauliHamiltonianInner::Ham8(h) => {
                            basis.build(seed, |state| h.apply(state).into_iter());
                        }
                        PauliHamiltonianInner::Ham16(h) => {
                            basis.build(seed, |state| h.apply(state).into_iter());
                        }
                    }
                }
                HardcoreBasisInner::$inner_variant(basis)
            }};
        }

        let inner = select_symspace_type!(n_sites, build_symmetric);
        Ok(PyHardcoreBasis { inner })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    /// Number of basis states.
    #[getter]
    pub fn size(&self) -> usize {
        self.inner.size()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a Python iterable of seed states as a `Vec<u128>`.
///
/// `u128` is wide enough to hold any seed ≤ 128 bits; larger seeds are
/// rejected with a `ValueError` (use ruint seeds for >128-bit systems via
/// a future API extension).
fn extract_seed_list(seeds: &Bound<'_, PyAny>) -> Result<Vec<u128>, Error> {
    seeds
        .try_iter()
        .map_err(|_| {
            Error(quspin_core::error::QuSpinError::ValueError(
                "seeds must be an iterable of integers".to_string(),
            ))
        })?
        .map(|item| {
            item.map_err(|e| Error(quspin_core::error::QuSpinError::ValueError(e.to_string())))
                .and_then(|obj| {
                    obj.extract::<u128>().map_err(|_| {
                        Error(quspin_core::error::QuSpinError::ValueError(
                            "each seed must be a non-negative integer ≤ 2^128 − 1".to_string(),
                        ))
                    })
                })
        })
        .collect()
}

/// Cast a `u128` seed to the concrete basis integer type `B`.
///
/// Uses two `from_u64` calls (low / high 64-bit halves) ORed together.
fn seed_as<B: BitInt>(seed: u128) -> B {
    let lo = seed as u64;
    let hi = (seed >> 64) as u64;
    B::from_u64(lo) | (B::from_u64(hi) << 64)
}

/// Select the smallest `B` type that fits `n_sites` bits and invoke
/// `$mac!($B, $variant_name)`.
///
/// The `Sub*` / `Sym*` ruint variants are selected here; Full* is handled
/// separately in `PyHardcoreBasis::full`.
/// Select the smallest `B` type that fits `n_sites` bits and invoke
/// `$mac!($B, $variant_name)` where the variant name matches the Sub*/Sym*
/// family (not Full*, which is handled separately in `PyHardcoreBasis::full`).
macro_rules! select_subspace_type {
    ($n_sites:expr, $mac:ident) => {
        if $n_sites <= 32 {
            $mac!(u32, Sub32)
        } else if $n_sites <= 64 {
            $mac!(u64, Sub64)
        } else if $n_sites <= 128 {
            $mac!(ruint::Uint<128, 2>, Sub128)
        } else if $n_sites <= 256 {
            $mac!(ruint::Uint<256, 4>, Sub256)
        } else if $n_sites <= 512 {
            $mac!(ruint::Uint<512, 8>, Sub512)
        } else if $n_sites <= 1024 {
            $mac!(ruint::Uint<1024, 16>, Sub1024)
        } else if $n_sites <= 2048 {
            $mac!(ruint::Uint<2048, 32>, Sub2048)
        } else if $n_sites <= 4096 {
            $mac!(ruint::Uint<4096, 64>, Sub4096)
        } else if $n_sites <= 8192 {
            $mac!(ruint::Uint<8192, 128>, Sub8192)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites={} exceeds the maximum supported value of 8192",
                $n_sites
            )));
        }
    };
}

macro_rules! select_symspace_type {
    ($n_sites:expr, $mac:ident) => {
        if $n_sites <= 32 {
            $mac!(u32, Sym32)
        } else if $n_sites <= 64 {
            $mac!(u64, Sym64)
        } else if $n_sites <= 128 {
            $mac!(ruint::Uint<128, 2>, Sym128)
        } else if $n_sites <= 256 {
            $mac!(ruint::Uint<256, 4>, Sym256)
        } else if $n_sites <= 512 {
            $mac!(ruint::Uint<512, 8>, Sym512)
        } else if $n_sites <= 1024 {
            $mac!(ruint::Uint<1024, 16>, Sym1024)
        } else if $n_sites <= 2048 {
            $mac!(ruint::Uint<2048, 32>, Sym2048)
        } else if $n_sites <= 4096 {
            $mac!(ruint::Uint<4096, 64>, Sym4096)
        } else if $n_sites <= 8192 {
            $mac!(ruint::Uint<8192, 128>, Sym8192)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites={} exceeds the maximum supported value of 8192",
                $n_sites
            )));
        }
    };
}

use select_subspace_type;
use select_symspace_type;
