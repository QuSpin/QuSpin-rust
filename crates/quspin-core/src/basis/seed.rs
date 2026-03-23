/// Seed-state conversion utilities.
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
use crate::bitbasis::BitInt;
use crate::error::QuSpinError;

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
