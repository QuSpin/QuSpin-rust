pub mod boson;
pub mod fermion;
pub mod generic;
pub mod spin;

pub use boson::PyBosonBasis;
pub use fermion::PyFermionBasis;
pub use generic::PyGenericBasis;
pub use spin::PySpinBasis;

// ---------------------------------------------------------------------------
// Shared basis helpers
// ---------------------------------------------------------------------------

use crate::error::Error;
use num_complex::Complex;
use pyo3::prelude::*;
use quspin_core::basis::dispatch::SpaceInner;
use quspin_core::basis::seed::{dit_seed_from_str, seed_from_str};
use quspin_core::error::QuSpinError;

/// Single accessor that exposes the underlying [`SpaceInner`] from any
/// of the PyO3 basis wrappers.
///
/// Avoids hard-coding the wrapper-internal nesting depth at every call
/// site (`b.inner.inner.inner` etc.) — the operator/qmatrix layer just
/// asks for `&SpaceInner` via this trait. Future wrapper restructuring
/// (e.g. another nesting level) is contained to one impl per wrapper.
pub(crate) trait AsSpaceInner {
    fn as_space_inner(&self) -> &SpaceInner;
}

impl AsSpaceInner for PySpinBasis {
    #[inline]
    fn as_space_inner(&self) -> &SpaceInner {
        &self.inner.inner.inner
    }
}

impl AsSpaceInner for PyBosonBasis {
    #[inline]
    fn as_space_inner(&self) -> &SpaceInner {
        &self.inner.inner.inner
    }
}

impl AsSpaceInner for PyFermionBasis {
    #[inline]
    fn as_space_inner(&self) -> &SpaceInner {
        &self.inner.inner.inner
    }
}

impl AsSpaceInner for PyGenericBasis {
    #[inline]
    fn as_space_inner(&self) -> &SpaceInner {
        &self.inner.inner
    }
}

/// Parse seed strings into byte vectors.
///
/// For `lhss == 2` uses binary `seed_from_str`; for `lhss > 2` uses
/// `dit_seed_from_str`.
pub(crate) fn parse_seeds(seeds: &[String], lhss: usize) -> PyResult<Vec<Vec<u8>>> {
    seeds
        .iter()
        .map(|s| {
            if lhss == 2 {
                seed_from_str(s).map_err(Error::from).map_err(PyErr::from)
            } else {
                dit_seed_from_str(s, lhss)
                    .map_err(Error::from)
                    .map_err(PyErr::from)
            }
        })
        .collect()
}

/// Apply lattice symmetry generators via an `add_lattice` callback.
pub(crate) fn apply_symmetries<F>(
    symmetries: &[(Vec<usize>, (f64, f64))],
    mut add_lattice: F,
) -> PyResult<()>
where
    F: FnMut(Complex<f64>, Vec<usize>) -> Result<(), QuSpinError>,
{
    for (perm, (re, im)) in symmetries {
        add_lattice(Complex::new(*re, *im), perm.clone()).map_err(Error::from)?;
    }
    Ok(())
}

/// Parse a state string to bytes, handling LHSS=2 (binary) and LHSS>2 (dit).
pub(crate) fn parse_state_str(state_str: &str, lhss: usize) -> PyResult<Vec<u8>> {
    if lhss == 2 {
        seed_from_str(state_str)
            .map_err(Error::from)
            .map_err(PyErr::from)
    } else {
        dit_seed_from_str(state_str, lhss)
            .map_err(Error::from)
            .map_err(PyErr::from)
    }
}
