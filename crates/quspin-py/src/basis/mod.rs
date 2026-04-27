pub mod boson;
pub mod fermion;
pub mod generic;
pub mod spin;
pub mod sym_element;

pub use boson::PyBosonBasis;
pub use fermion::PyFermionBasis;
pub use generic::PyGenericBasis;
pub use spin::PySpinBasis;
pub use sym_element::PySymElement;

// ---------------------------------------------------------------------------
// Shared basis helpers
// ---------------------------------------------------------------------------

use crate::error::Error;
use num_complex::Complex;
use pyo3::prelude::*;
use quspin_core::basis::seed::{dit_seed_from_str, seed_from_str};
use quspin_core::error::QuSpinError;

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
//
// `#[allow(dead_code)]`: scheduled for deletion in Task 14 once the
// last call sites that pass `symmetries=` to `*Basis.symmetric` are
// migrated in Task 15. Kept as a separate task to keep diffs focused.
#[allow(dead_code)]
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

/// Read `(n_sites, lhss)` from a `SymmetryGroup`-like Python object via
/// attribute access. Used by all four `*Basis.symmetric(group, ...)`
/// constructors so they don't have to take `n_sites` / `lhss` as
/// separate arguments.
pub(crate) fn group_n_sites_lhss(group: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    let n_sites: usize = group.getattr("n_sites")?.extract()?;
    let lhss: usize = group.getattr("lhss")?.extract()?;
    Ok((n_sites, lhss))
}

/// Replay each `(element, character)` pair from a `SymmetryGroup`-like
/// Python iterable into a [`GenericBasis`](quspin_core::basis::GenericBasis)
/// via [`PySymElement::add_to_basis`](crate::basis::sym_element::PySymElement::add_to_basis).
/// Used by spin / boson / generic Python wrappers.
pub(crate) fn replay_group_into_generic(
    group: &Bound<'_, PyAny>,
    basis: &mut quspin_core::basis::GenericBasis,
) -> PyResult<()> {
    for item in group.try_iter()? {
        let item = item?;
        let tup = item.downcast::<pyo3::types::PyTuple>()?;
        let elem_obj = tup.get_item(0)?;
        let elem = elem_obj.downcast::<crate::basis::sym_element::PySymElement>()?;
        let chi: Complex<f64> = tup.get_item(1)?.extract()?;
        elem.borrow()
            .add_to_basis(basis, chi)
            .map_err(Error::from)?;
    }
    Ok(())
}

/// Replay each `(element, character)` pair from a `SymmetryGroup`-like
/// Python iterable into a [`BitBasis`](quspin_core::basis::dispatch::BitBasis)
/// via [`PySymElement::add_to_bit_basis`](crate::basis::sym_element::PySymElement::add_to_bit_basis).
/// Used by the fermion Python wrapper.
pub(crate) fn replay_group_into_bit(
    group: &Bound<'_, PyAny>,
    basis: &mut quspin_core::basis::dispatch::BitBasis,
) -> PyResult<()> {
    for item in group.try_iter()? {
        let item = item?;
        let tup = item.downcast::<pyo3::types::PyTuple>()?;
        let elem_obj = tup.get_item(0)?;
        let elem = elem_obj.downcast::<crate::basis::sym_element::PySymElement>()?;
        let chi: Complex<f64> = tup.get_item(1)?.extract()?;
        elem.borrow()
            .add_to_bit_basis(basis, chi)
            .map_err(Error::from)?;
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
