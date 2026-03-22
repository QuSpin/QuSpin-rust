/// Python-facing `PyHardcoreBasis` pyclass.
pub mod dispatch;

use crate::error::Error;
use crate::hamiltonian::PyHardcoreHamiltonian;
use crate::hamiltonian::dispatch::HardcoreHamiltonianInner;
use bitbasis::BitInt;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use quspin_core::basis::{
    space::{FullSpace, Subspace},
    sym::SymmetricSubspace,
};

use super::symmetry::PySymmetryGrp;
use dispatch::HardcoreBasisInner;

// ---------------------------------------------------------------------------
// Builder-selection macros  (must appear before the impl that uses them)
// ---------------------------------------------------------------------------

/// Select the smallest `B` type that fits `n_sites` bits (Sub* family).
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

// ---------------------------------------------------------------------------
// PyHardcoreBasis
// ---------------------------------------------------------------------------

#[pyclass(name = "PyHardcoreBasis")]
pub struct PyHardcoreBasis {
    pub n_sites: usize,
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
        Ok(PyHardcoreBasis { n_sites, inner })
    }

    /// Subspace reachable from each seed state under the Hamiltonian.
    ///
    /// Args:
    ///   seeds: Python list of integer seed states.
    ///   ham:   The Hamiltonian defining connectivity.
    #[staticmethod]
    pub fn subspace(seeds: &Bound<'_, PyAny>, ham: &PyHardcoreHamiltonian) -> PyResult<Self> {
        let n_sites = ham.inner.n_sites();
        let seed_list = extract_seed_list(seeds)?;

        macro_rules! build_subspace {
            ($B:ty, $inner_variant:ident) => {{
                let mut basis = Subspace::<$B>::new();
                for s in &seed_list {
                    let seed = seed_as::<$B>(s);
                    match &ham.inner {
                        HardcoreHamiltonianInner::Ham8(h) => {
                            basis.build(seed, |state| h.apply(state).into_iter());
                        }
                        HardcoreHamiltonianInner::Ham16(h) => {
                            basis.build(seed, |state| h.apply(state).into_iter());
                        }
                    }
                }
                HardcoreBasisInner::$inner_variant(basis)
            }};
        }

        let inner = select_subspace_type!(n_sites, build_subspace);
        Ok(PyHardcoreBasis { n_sites, inner })
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
        ham: &PyHardcoreHamiltonian,
        grp: &PySymmetryGrp,
    ) -> PyResult<Self> {
        use quspin_core::basis::SymmetryGrpInner;

        let n_sites = ham.inner.n_sites();
        if grp.n_sites() != n_sites {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_sites mismatch: symmetry group has {} sites but Hamiltonian has {}",
                grp.n_sites(),
                n_sites
            )));
        }
        let seed_list = extract_seed_list(seeds)?;

        // Helper macro: build a SymmetricSubspace<$B> from a concrete sym_grp reference,
        // then wrap the result in the matching HardcoreBasisInner variant.
        macro_rules! build_sym {
            ($sym_grp:expr, $B:ty, $basis_variant:ident) => {{
                let mut basis = SymmetricSubspace::<$B>::new($sym_grp.clone());
                for s in &seed_list {
                    let seed = seed_as::<$B>(s);
                    match &ham.inner {
                        HardcoreHamiltonianInner::Ham8(h) => {
                            basis.build(seed, |state| h.apply(state).into_iter());
                        }
                        HardcoreHamiltonianInner::Ham16(h) => {
                            basis.build(seed, |state| h.apply(state).into_iter());
                        }
                    }
                }
                HardcoreBasisInner::$basis_variant(basis)
            }};
        }

        let inner = match &grp.inner {
            SymmetryGrpInner::Sym32(g) => build_sym!(g, u32, Sym32),
            SymmetryGrpInner::Sym64(g) => build_sym!(g, u64, Sym64),
            SymmetryGrpInner::Sym128(g) => build_sym!(g, ruint::Uint<128, 2>, Sym128),
            SymmetryGrpInner::Sym256(g) => build_sym!(g, ruint::Uint<256, 4>, Sym256),
            SymmetryGrpInner::Sym512(g) => build_sym!(g, ruint::Uint<512, 8>, Sym512),
            SymmetryGrpInner::Sym1024(g) => build_sym!(g, ruint::Uint<1024, 16>, Sym1024),
            SymmetryGrpInner::Sym2048(g) => build_sym!(g, ruint::Uint<2048, 32>, Sym2048),
            SymmetryGrpInner::Sym4096(g) => build_sym!(g, ruint::Uint<4096, 64>, Sym4096),
            SymmetryGrpInner::Sym8192(g) => build_sym!(g, ruint::Uint<8192, 128>, Sym8192),
        };

        Ok(PyHardcoreBasis { n_sites, inner })
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    /// Number of sites.
    #[getter]
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Number of basis states.
    #[getter]
    pub fn size(&self) -> usize {
        self.inner.size()
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
                .and_then(|obj| seed_to_bits(&obj))
        })
        .collect()
}

/// Convert a single Python seed (str or list[int]) to a site-occupation bit vector.
fn seed_to_bits(obj: &Bound<'_, PyAny>) -> Result<Vec<u8>, Error> {
    if let Ok(s) = obj.extract::<String>() {
        s.chars()
            .map(|c| match c {
                '0' => Ok(0u8),
                '1' => Ok(1u8),
                _ => Err(Error(quspin_core::error::QuSpinError::ValueError(format!(
                    "invalid character '{c}' in seed string; expected '0' or '1'"
                )))),
            })
            .collect()
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

/// Construct a `B` basis state from a site-occupation bit vector.
///
/// `bits[i]` is the occupation (0 or 1) of site `i`.  Bits beyond `B::BITS`
/// are silently ignored.
fn seed_as<B: BitInt>(bits: &[u8]) -> B {
    let mut result = B::from_u64(0);
    for (i, &v) in bits.iter().enumerate() {
        if v != 0 && i < B::BITS as usize {
            result = result | (B::from_u64(1) << i);
        }
    }
    result
}
