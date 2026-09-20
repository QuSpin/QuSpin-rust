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

// ---------------------------------------------------------------------------
// Human-readable state strings (ket notation, tokenised occupations)
// ---------------------------------------------------------------------------

/// Strip optional ket delimiters from a state string.
///
/// `"|0101>"` → `"0101"`, `"0101"` → `"0101"`. Returns
/// `QuSpinError::ValueError` when only one of the two delimiters is present.
pub fn strip_ket_notation(state_str: &str) -> Result<&str, QuSpinError> {
    const MISMATCH: &str = "state string must either include both '|' and '>' or neither";

    let trimmed = state_str.trim();
    match (trimmed.strip_prefix('|'), trimmed.strip_suffix('>')) {
        (Some(without_prefix), Some(_)) => Ok(without_prefix
            .strip_suffix('>')
            .expect("suffix '>' checked above")
            .trim()),
        (None, None) => Ok(trimmed),
        _ => Err(QuSpinError::ValueError(MISMATCH.to_string())),
    }
}

/// Parse a whitespace- or comma-separated state string into occupation bytes.
///
/// Returns `Ok(None)` when `state_str` contains no separator, signalling that
/// the caller should fall back to the one-character-per-site forms
/// ([`seed_from_str`] / [`dit_seed_from_str`]). The tokenised form is the only
/// way to express per-site occupations of 10 or more.
pub fn tokenized_state_from_str(
    state_str: &str,
    n_sites: usize,
    lhss: usize,
) -> Result<Option<Vec<u8>>, QuSpinError> {
    let is_sep = |c: char| c.is_whitespace() || c == ',';
    if !state_str.contains(is_sep) {
        return Ok(None);
    }

    let tokens: Vec<&str> = state_str.split(is_sep).filter(|t| !t.is_empty()).collect();
    if tokens.len() != n_sites {
        return Err(QuSpinError::ValueError(format!(
            "state string has {} site values, expected {n_sites}",
            tokens.len(),
        )));
    }

    tokens
        .into_iter()
        .enumerate()
        .map(|(site, tok)| {
            let value: usize = tok.parse().map_err(|_| {
                QuSpinError::ValueError(format!(
                    "invalid site value '{tok}' at site {site}; expected a non-negative integer"
                ))
            })?;
            if value >= lhss {
                return Err(QuSpinError::ValueError(format!(
                    "invalid site value {value} at site {site} for lhss={lhss}"
                )));
            }
            Ok(value as u8)
        })
        .collect::<Result<Vec<u8>, _>>()
        .map(Some)
}

/// Parse any supported human-readable state string into occupation bytes.
///
/// Accepts, in order of preference:
///
/// - optional ket delimiters around any of the forms below — `"|0101>"`;
/// - tokenised occupations separated by whitespace or commas —
///   `"0 10 0 1"`, `"0,10,0,1"` (the only form that can express occupations
///   of 10 or more);
/// - one character per site — `"0101"` for `lhss == 2` (via
///   [`seed_from_str`]), one decimal digit per site otherwise (via
///   [`dit_seed_from_str`]).
///
/// `bytes[i]` is the occupation of site `i`, so the result can be handed
/// straight to [`seed_from_bytes`] / [`dit_seed_from_bytes`].
pub fn state_from_str(s: &str, n_sites: usize, lhss: usize) -> Result<Vec<u8>, QuSpinError> {
    let s = strip_ket_notation(s)?;

    if let Some(bytes) = tokenized_state_from_str(s, n_sites, lhss)? {
        return Ok(bytes);
    }

    if lhss == 2 {
        seed_from_str(s, n_sites)
    } else {
        dit_seed_from_str(s, n_sites, lhss)
    }
}

/// Render occupation bytes as a human-readable state string.
///
/// Inverse of [`state_from_str`]. Uses the compact one-digit-per-site form
/// unless some site is occupied by 10 or more, in which case the
/// space-separated tokenised form is emitted so the result round-trips.
pub fn state_to_display_str(bytes: &[u8], bracket_notation: bool) -> String {
    let body = if bytes.iter().any(|&v| v >= 10) {
        bytes
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        bytes
            .iter()
            .map(|&v| char::from_digit(v as u32, 10).unwrap_or('?'))
            .collect()
    };

    if bracket_notation {
        format!("|{body}>")
    } else {
        body
    }
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

    // --- strip_ket_notation --------------------------------------------------

    #[test]
    fn strip_ket_notation_accepts_both_forms() {
        assert_eq!(strip_ket_notation("|0101>").unwrap(), "0101");
        assert_eq!(strip_ket_notation("0101").unwrap(), "0101");
        assert_eq!(strip_ket_notation("  | 0 1 >  ").unwrap(), "0 1");
    }

    #[test]
    fn strip_ket_notation_rejects_half_a_ket() {
        assert!(strip_ket_notation("|0101").is_err());
        assert!(strip_ket_notation("0101>").is_err());
    }

    // --- tokenized_state_from_str --------------------------------------------

    #[test]
    fn tokenized_state_returns_none_without_separator() {
        assert_eq!(tokenized_state_from_str("0101", 4, 2).unwrap(), None);
    }

    #[test]
    fn tokenized_state_parses_multi_digit_occupations() {
        assert_eq!(
            tokenized_state_from_str("0 10 0 1", 4, 11).unwrap(),
            Some(vec![0, 10, 0, 1])
        );
        assert_eq!(
            tokenized_state_from_str("0,10,0,1", 4, 11).unwrap(),
            Some(vec![0, 10, 0, 1])
        );
    }

    #[test]
    fn tokenized_state_rejects_wrong_site_count() {
        let err = tokenized_state_from_str("0 1 0", 4, 2).unwrap_err();
        assert!(matches!(err, QuSpinError::ValueError(ref m) if m.contains("3 site values")));
    }

    #[test]
    fn tokenized_state_rejects_out_of_range_occupation() {
        let err = tokenized_state_from_str("0 3 0 1", 4, 3).unwrap_err();
        assert!(matches!(err, QuSpinError::ValueError(ref m) if m.contains("site value 3")));
    }

    #[test]
    fn tokenized_state_rejects_non_numeric_token() {
        let err = tokenized_state_from_str("0 x 0 1", 4, 2).unwrap_err();
        assert!(matches!(err, QuSpinError::ValueError(ref m) if m.contains("'x'")));
    }

    // --- state_from_str ------------------------------------------------------

    #[test]
    fn state_from_str_accepts_every_supported_form() {
        assert_eq!(state_from_str("0101", 4, 2).unwrap(), vec![0, 1, 0, 1]);
        assert_eq!(state_from_str("|0101>", 4, 2).unwrap(), vec![0, 1, 0, 1]);
        assert_eq!(state_from_str("0 1 0 1", 4, 2).unwrap(), vec![0, 1, 0, 1]);
        assert_eq!(state_from_str("|0,1,0,1>", 4, 2).unwrap(), vec![0, 1, 0, 1]);
        assert_eq!(state_from_str("0123", 4, 4).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(state_from_str("|0 10>", 2, 11).unwrap(), vec![0, 10]);
    }

    #[test]
    fn state_from_str_rejects_bad_input() {
        assert!(state_from_str("|0101", 4, 2).is_err());
        assert!(state_from_str("012", 4, 2).is_err());
        assert!(state_from_str("0102", 4, 2).is_err());
    }

    // --- state_to_display_str ------------------------------------------------

    #[test]
    fn state_to_display_str_round_trips_through_state_from_str() {
        for (bytes, lhss) in [
            (vec![0u8, 1, 0, 1], 2usize),
            (vec![0, 1, 2, 3], 4),
            (vec![0, 10, 3], 11),
        ] {
            for bracket in [true, false] {
                let s = state_to_display_str(&bytes, bracket);
                assert_eq!(state_from_str(&s, bytes.len(), lhss).unwrap(), bytes, "{s}");
            }
        }
    }

    #[test]
    fn state_to_display_str_picks_tokenised_form_for_wide_occupations() {
        assert_eq!(state_to_display_str(&[0, 1, 0], true), "|010>");
        assert_eq!(state_to_display_str(&[0, 1, 0], false), "010");
        assert_eq!(state_to_display_str(&[0, 10, 3], true), "|0 10 3>");
    }
}
