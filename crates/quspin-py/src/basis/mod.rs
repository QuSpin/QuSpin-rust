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

/// Parse seed strings into byte vectors.
///
/// Each seed must have length `n_sites`. For `lhss == 2` uses binary
/// `seed_from_str`; for `lhss > 2` uses `dit_seed_from_str`.
pub(crate) fn parse_seeds(seeds: &[String], n_sites: usize, lhss: usize) -> PyResult<Vec<Vec<u8>>> {
    seeds
        .iter()
        .map(|s| {
            if lhss == 2 {
                seed_from_str(s, n_sites)
                    .map_err(Error::from)
                    .map_err(PyErr::from)
            } else {
                dit_seed_from_str(s, n_sites, lhss)
                    .map_err(Error::from)
                    .map_err(PyErr::from)
            }
        })
        .collect()
}

/// Reject an operator that references a site index `>= n_sites`. Called
/// from the basis builder helpers after the operator has been downcast.
pub(crate) fn validate_op_max_site(op_max_site: usize, n_sites: usize) -> PyResult<()> {
    if op_max_site >= n_sites {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "operator references site {op_max_site} but basis has only \
             {n_sites} sites (max valid index is {})",
            n_sites.saturating_sub(1),
        )));
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

/// Iterate `(element, character)` pairs from a `SymmetryGroup`-like Python
/// iterable, downcast each tuple, and forward the borrowed [`PySymElement`]
/// and complex character to `f`. Shared by [`replay_group_into_generic`] and
/// [`replay_group_into_bit`].
fn replay_group<F>(group: &Bound<'_, PyAny>, mut f: F) -> PyResult<()>
where
    F: FnMut(&crate::basis::sym_element::PySymElement, Complex<f64>) -> PyResult<()>,
{
    for item in group.try_iter()? {
        let item = item?;
        let tup = item.cast::<pyo3::types::PyTuple>()?;
        let elem_obj = tup.get_item(0)?;
        let elem = elem_obj.cast::<crate::basis::sym_element::PySymElement>()?;
        let chi: Complex<f64> = tup.get_item(1)?.extract()?;
        f(&elem.borrow(), chi)?;
    }
    Ok(())
}

/// Replay each `(element, character)` pair from a `SymmetryGroup`-like
/// Python iterable into a [`GenericBasis`](quspin_core::basis::GenericBasis)
/// via [`PySymElement::add_to_basis`](crate::basis::sym_element::PySymElement::add_to_basis).
/// Used by spin / boson / generic Python wrappers.
pub(crate) fn replay_group_into_generic(
    group: &Bound<'_, PyAny>,
    basis: &mut quspin_core::basis::GenericBasis,
) -> PyResult<()> {
    replay_group(group, |elem, chi| {
        elem.add_to_basis(basis, chi).map_err(Error::from)?;
        Ok(())
    })
}

/// Replay each `(element, character)` pair from a `SymmetryGroup`-like
/// Python iterable into a [`BitBasis`](quspin_core::basis::dispatch::BitBasis)
/// via [`PySymElement::add_to_bit_basis`](crate::basis::sym_element::PySymElement::add_to_bit_basis).
/// Used by the fermion Python wrapper.
pub(crate) fn replay_group_into_bit(
    group: &Bound<'_, PyAny>,
    basis: &mut quspin_core::basis::dispatch::BitBasis,
) -> PyResult<()> {
    replay_group(group, |elem, chi| {
        elem.add_to_bit_basis(basis, chi).map_err(Error::from)?;
        Ok(())
    })
}

/// Parse a state string to bytes, handling LHSS=2 (binary) and LHSS>2 (dit).
///
/// `state_str` must have length `n_sites`.
pub(crate) fn parse_state_str(state_str: &str, n_sites: usize, lhss: usize) -> PyResult<Vec<u8>> {
    if lhss == 2 {
        seed_from_str(state_str, n_sites)
            .map_err(Error::from)
            .map_err(PyErr::from)
    } else {
        dit_seed_from_str(state_str, n_sites, lhss)
            .map_err(Error::from)
            .map_err(PyErr::from)
    }
}
