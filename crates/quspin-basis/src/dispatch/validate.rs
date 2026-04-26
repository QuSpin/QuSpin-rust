//! Argument-shape validators called by the user-facing
//! `GenericBasis::add_lattice` / `add_inv` / `add_local` methods.
//!
//! These check that user-supplied permutations / `perm_vals` / `locs`
//! have the right length, contain valid values, and (for `perm_vals`)
//! describe a real bijection. The per-family inner-enum methods that
//! actually insert the element trust their callers — validation
//! happens once at the [`GenericBasis`](super::GenericBasis) layer.

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
