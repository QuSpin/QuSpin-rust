/// Seed-state conversion utilities for both hardcore (LHSS=2) and dit (LHSS>2)
/// basis states.
///
/// A *seed* is a computational basis state used to initialise basis
/// construction (BFS / symmetry projection).  Internally seeds are values of
/// type `B: BitInt`, but callers typically express them in one of two forms:
///
/// - **byte slice** (`&[u8]`): one byte per site, value 0 or 1.
///   This is the lowest common denominator and is used directly by the C FFI.
/// - **string slice** (`&str`): ASCII characters `'0'` and `'1'`, one per
///   site.  This is the Python / human-readable form.
///
/// `seed_from_str` is a thin wrapper: it validates the string and converts it
/// to a byte slice, then delegates to `seed_from_bytes`.
use crate::bitbasis::{BitInt, manip::DynamicDitManip};
use quspin_types::QuSpinError;

/// Convert a `B` basis state to a `'0'`/`'1'` string.
///
/// `output[i]` is `'1'` if bit `i` of `state` is set, `'0'` otherwise.
/// Exactly `n_sites` characters are produced, matching the convention of
/// `seed_from_str` / `seed_from_bytes`.
pub fn state_to_str<B: BitInt>(state: B, n_sites: usize) -> String {
    let one = B::from_u64(1);
    (0..n_sites)
        .map(|i| {
            if (state >> i) & one != B::from_u64(0) {
                '1'
            } else {
                '0'
            }
        })
        .collect()
}

/// Construct a `B` basis state from a site-occupation byte slice.
///
/// `bytes[i]` is the occupation (0 or 1) of site `i`.
/// Bits beyond `B::BITS` are silently ignored.
pub fn seed_from_bytes<B: BitInt>(bytes: &[u8]) -> B {
    let mut result = B::from_u64(0);
    for (i, &v) in bytes.iter().enumerate() {
        if v != 0 && i < B::BITS as usize {
            result = result | (B::from_u64(1) << i);
        }
    }
    result
}

/// Parse a `'0'`/`'1'` ASCII string into a site-occupation byte vector.
///
/// Returns `QuSpinError::ValueError` if any character is not `'0'` or `'1'`.
/// The resulting `Vec<u8>` can be passed directly to `seed_from_bytes`.
pub fn seed_from_str(s: &str) -> Result<Vec<u8>, QuSpinError> {
    s.chars()
        .map(|c| match c {
            '0' => Ok(0u8),
            '1' => Ok(1u8),
            _ => Err(QuSpinError::ValueError(format!(
                "invalid character '{c}' in seed string; expected '0' or '1'"
            ))),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dit (LHSS > 2) seed utilities
// ---------------------------------------------------------------------------

/// Convert a dit basis state to a decimal digit string.
///
/// `output[i]` is the decimal digit for the dit at site `i`.
/// Exactly `n_sites` characters are produced.
pub fn dit_state_to_str<B: BitInt>(state: B, n_sites: usize, manip: &DynamicDitManip) -> String {
    (0..n_sites)
        .map(|i| {
            let val = manip.get_dit(state, i);
            char::from_digit(val as u32, 10).unwrap_or('?')
        })
        .collect()
}

/// Construct a `B` dit basis state from a site-occupation byte slice.
///
/// `bytes[i]` is the occupation (0 ≤ value < lhss) of site `i`.
/// Uses `DynamicDitManip` to pack each dit into `B`.
pub fn dit_seed_from_bytes<B: BitInt>(bytes: &[u8], manip: &DynamicDitManip) -> B {
    let mut result = B::from_u64(0);
    for (i, &v) in bytes.iter().enumerate() {
        result = manip.set_dit(result, v as usize, i);
    }
    result
}

/// Parse a decimal ASCII string into a site-occupation byte vector.
///
/// Returns `QuSpinError::ValueError` if any character is not a valid decimal
/// digit in range `0..lhss`.
pub fn dit_seed_from_str(s: &str, lhss: usize) -> Result<Vec<u8>, QuSpinError> {
    s.chars()
        .map(|c| {
            c.to_digit(10)
                .and_then(|d| {
                    let d = d as usize;
                    if d < lhss { Some(d as u8) } else { None }
                })
                .ok_or_else(|| {
                    QuSpinError::ValueError(format!(
                        "invalid character '{c}' in dit seed string for lhss={lhss}; \
                         expected a digit 0..{lhss}"
                    ))
                })
        })
        .collect()
}
