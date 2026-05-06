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
/// `seed_from_str` is a thin wrapper: it validates the string length matches
/// `n_sites` and the character set, then delegates to `seed_from_bytes`.
use quspin_bitbasis::{BitInt, manip::DynamicDitManip};
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
/// `n_sites` is the expected length of `s`. Returns
/// `QuSpinError::ValueError` if the length doesn't match, or if any
/// character is not `'0'` or `'1'`. The resulting `Vec<u8>` can be passed
/// directly to `seed_from_bytes`.
pub fn seed_from_str(s: &str, n_sites: usize) -> Result<Vec<u8>, QuSpinError> {
    let len = s.chars().count();
    if len != n_sites {
        return Err(QuSpinError::ValueError(format!(
            "seed string has length {len}, expected {n_sites} (one character per site)"
        )));
    }
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
/// `n_sites` is the expected length of `s`. Returns
/// `QuSpinError::ValueError` if the length doesn't match, or if any
/// character is not a valid decimal digit in range `0..lhss`.
pub fn dit_seed_from_str(s: &str, n_sites: usize, lhss: usize) -> Result<Vec<u8>, QuSpinError> {
    let len = s.chars().count();
    if len != n_sites {
        return Err(QuSpinError::ValueError(format!(
            "seed string has length {len}, expected {n_sites} (one character per site)"
        )));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_from_str_ok() {
        assert_eq!(seed_from_str("0101", 4).unwrap(), vec![0, 1, 0, 1]);
    }

    #[test]
    fn seed_from_str_rejects_short() {
        let err = seed_from_str("01", 4).unwrap_err();
        assert!(matches!(err, QuSpinError::ValueError(ref m) if m.contains("length 2")));
    }

    #[test]
    fn seed_from_str_rejects_long() {
        let err = seed_from_str("01010", 4).unwrap_err();
        assert!(matches!(err, QuSpinError::ValueError(ref m) if m.contains("length 5")));
    }

    #[test]
    fn seed_from_str_rejects_bad_char() {
        let err = seed_from_str("0102", 4).unwrap_err();
        assert!(matches!(err, QuSpinError::ValueError(ref m) if m.contains("'2'")));
    }

    #[test]
    fn dit_seed_from_str_ok() {
        assert_eq!(dit_seed_from_str("0123", 4, 4).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn dit_seed_from_str_rejects_length_mismatch() {
        let err = dit_seed_from_str("012", 4, 4).unwrap_err();
        assert!(matches!(err, QuSpinError::ValueError(ref m) if m.contains("length 3")));
    }

    #[test]
    fn dit_seed_from_str_rejects_oversize_digit() {
        let err = dit_seed_from_str("0124", 4, 4).unwrap_err();
        assert!(matches!(err, QuSpinError::ValueError(ref m) if m.contains("'4'")));
    }
}
