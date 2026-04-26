//! Argument-shape validators shared by the concrete-impl entry points
//! that own validation:
//!
//! - [`validate_perm`] is called by
//!   [`SymBasis::add_symmetry`](crate::sym::SymBasis::add_symmetry)
//!   on the site-permutation component of every group element.
//! - [`validate_perm_vals`] and [`validate_locs`] are called by the
//!   per-family inner enums' `add_local` / `add_inv` methods, where
//!   the typed local op is constructed from the user-supplied
//!   `perm_vals` / `locs`.
//!
//! The umbrella dispatch enums ([`GenericBasis`](super::GenericBasis),
//! [`DitBasis`](super::DitBasis), and the family enums) do no
//! validation themselves — they only emit "method not supported on
//! this variant" errors.

use quspin_types::QuSpinError;

/// Verify `perm` has length `n_sites` and is a permutation of
/// `0..n_sites` (every value in range, no duplicates).
pub(crate) fn validate_perm(perm: &[usize], n_sites: usize) -> Result<(), QuSpinError> {
    if perm.len() != n_sites {
        return Err(QuSpinError::ValueError(format!(
            "perm.len()={} but n_sites={n_sites}",
            perm.len()
        )));
    }
    let mut seen = vec![false; n_sites];
    for (i, &p) in perm.iter().enumerate() {
        if p >= n_sites {
            return Err(QuSpinError::ValueError(format!(
                "perm[{i}]={p} is out of range 0..{n_sites}"
            )));
        }
        if seen[p] {
            return Err(QuSpinError::ValueError(format!(
                "perm has duplicate target site {p}"
            )));
        }
        seen[p] = true;
    }
    Ok(())
}

/// Verify `perm_vals` has length `lhss` and is a permutation of
/// `0..lhss`.
pub(crate) fn validate_perm_vals(perm_vals: &[u8], lhss: usize) -> Result<(), QuSpinError> {
    if perm_vals.len() != lhss {
        return Err(QuSpinError::ValueError(format!(
            "perm_vals.len()={} but lhss={lhss}",
            perm_vals.len()
        )));
    }
    let mut seen = vec![false; lhss];
    for (i, &v) in perm_vals.iter().enumerate() {
        let v = v as usize;
        if v >= lhss {
            return Err(QuSpinError::ValueError(format!(
                "perm_vals[{i}]={v} is out of range 0..{lhss}"
            )));
        }
        if seen[v] {
            return Err(QuSpinError::ValueError(format!(
                "perm_vals has duplicate value {v}"
            )));
        }
        seen[v] = true;
    }
    Ok(())
}

/// Verify every entry of `locs` is in `0..n_sites`.
pub(crate) fn validate_locs(locs: &[usize], n_sites: usize) -> Result<(), QuSpinError> {
    for (i, &loc) in locs.iter().enumerate() {
        if loc >= n_sites {
            return Err(QuSpinError::ValueError(format!(
                "locs[{i}]={loc} is out of range 0..{n_sites}"
            )));
        }
    }
    Ok(())
}
