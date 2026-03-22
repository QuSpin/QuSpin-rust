use super::int::BitInt;

// ---------------------------------------------------------------------------
// Lookup tables (mirrors constants::bits / constants::mask from dit_manip.hpp)
// ---------------------------------------------------------------------------

/// Number of bits required to encode `lhss` distinct values.
/// `bits_for_lhss(lhss) == ceil(log2(lhss))`, with lhss=0/1 clamped to 1.
const fn bits_for_lhss(lhss: usize) -> usize {
    match lhss {
        0..=2 => 1,
        _ => {
            let mut b = 1usize;
            let mut pow2 = 2usize;
            while pow2 < lhss {
                b += 1;
                pow2 <<= 1;
            }
            b
        }
    }
}

const fn make_bits_table() -> [usize; 256] {
    let mut table = [0usize; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = bits_for_lhss(i);
        i += 1;
    }
    table
}

const fn make_mask_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        let b = bits_for_lhss(i);
        table[i] = (1u64 << b) - 1;
        i += 1;
    }
    table
}

/// `BITS_TABLE[lhss]` — bits needed to store one dit with `lhss` values.
pub static BITS_TABLE: [usize; 256] = make_bits_table();

/// `MASK_TABLE[lhss]` — bitmask that isolates one dit with `lhss` values.
pub static MASK_TABLE: [u64; 256] = make_mask_table();

// ---------------------------------------------------------------------------
// DynamicDitManip
// ---------------------------------------------------------------------------

/// Runtime dit manipulator. Stores `lhss`, the per-dit bit-width, and the
/// bitmask used to extract a single dit from a basis-state integer.
///
/// Mirrors `dynamic_dit_manip` from `dit_manip.hpp`.
#[derive(Clone, Copy, Debug)]
pub struct DynamicDitManip {
    pub lhss: usize,
    /// Number of bits per dit.
    pub bits: usize,
    /// Bitmask that isolates one dit (always fits in u8, stored as u64 for
    /// convenience when constructing the `BitInt` mask operand).
    pub mask: u64,
}

impl DynamicDitManip {
    /// Create a manipulator for a local Hilbert space of size `lhss`.
    ///
    /// # Panics
    /// Panics if `lhss < 2` or `lhss > 255`.
    #[inline]
    pub fn new(lhss: usize) -> Self {
        assert!((2..=255).contains(&lhss), "lhss must be in 2..=255");
        DynamicDitManip {
            lhss,
            bits: BITS_TABLE[lhss],
            mask: MASK_TABLE[lhss],
        }
    }

    /// Extract the dit at position `i` from basis state `s`.
    ///
    /// Position 0 is the least-significant dit (lowest-order bits).
    #[inline]
    pub fn get_dit<I: BitInt>(&self, s: I, i: usize) -> usize {
        let shift = i * self.bits;
        (s >> shift & I::from_u64(self.mask)).to_usize()
    }

    /// Return a new basis state where the dit at position `i` is set to `val`,
    /// leaving all other dits unchanged.
    #[inline]
    pub fn set_dit<I: BitInt>(&self, s: I, val: usize, i: usize) -> I {
        let shift = i * self.bits;
        let mask_i = I::from_u64(self.mask) << shift;
        let cleared = s & !mask_i;
        cleared | (I::from_u64(val as u64) << shift)
    }

    /// Extract a sub-state value from a slice of dit positions.
    ///
    /// Returns a mixed-radix (base-`lhss`) number where `locs[last]` is the
    /// least-significant "digit":
    ///
    /// ```text
    /// result = dit(locs[last]) * lhss^0 + dit(locs[last-1]) * lhss^1 + ...
    /// ```
    ///
    /// Mirrors the multi-location overload of `get_sub_bitstring`.
    #[inline]
    pub fn get_sub_state<I: BitInt>(&self, s: I, locs: &[usize]) -> usize {
        let mut out = 0usize;
        let mut weight = 1usize;
        for &loc in locs.iter().rev() {
            out += weight * self.get_dit(s, loc);
            weight *= self.lhss;
        }
        out
    }

    /// Insert a mixed-radix sub-state value `val` into basis state `s` at the
    /// given positions.
    ///
    /// `locs[last]` receives the least-significant digit of `val`, consistent
    /// with [`get_sub_state`](Self::get_sub_state).
    #[inline]
    pub fn set_sub_state<I: BitInt>(&self, s: I, mut val: usize, locs: &[usize]) -> I {
        let mut out = s;
        for &loc in locs.iter().rev() {
            out = self.set_dit(out, val % self.lhss, loc);
            val /= self.lhss;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// DitManip — compile-time LHSS variant
// ---------------------------------------------------------------------------

/// Compile-time dit manipulator. A zero-sized type that delegates to a
/// [`DynamicDitManip`] constructed from `LHSS`.
///
/// Mirrors `dit_manip<lhss>` from `dit_manip.hpp`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DitManip<const LHSS: usize>;

impl<const LHSS: usize> DitManip<LHSS> {
    const INNER: DynamicDitManip = DynamicDitManip {
        lhss: LHSS,
        bits: BITS_TABLE[LHSS],
        mask: MASK_TABLE[LHSS],
    };

    #[inline]
    pub fn get_dit<I: BitInt>(s: I, i: usize) -> usize {
        Self::INNER.get_dit(s, i)
    }

    #[inline]
    pub fn set_dit<I: BitInt>(s: I, val: usize, i: usize) -> I {
        Self::INNER.set_dit(s, val, i)
    }

    #[inline]
    pub fn get_sub_state<I: BitInt>(s: I, locs: &[usize]) -> usize {
        Self::INNER.get_sub_state(s, locs)
    }

    #[inline]
    pub fn set_sub_state<I: BitInt>(s: I, val: usize, locs: &[usize]) -> I {
        Self::INNER.set_sub_state(s, val, locs)
    }

    #[inline]
    pub fn as_dynamic() -> DynamicDitManip {
        Self::INNER
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- table sanity checks ---

    #[test]
    fn bits_table_spot_checks() {
        // lhss 0,1,2 all need 1 bit
        assert_eq!(BITS_TABLE[0], 1);
        assert_eq!(BITS_TABLE[1], 1);
        assert_eq!(BITS_TABLE[2], 1);
        // lhss 3,4 need 2 bits
        assert_eq!(BITS_TABLE[3], 2);
        assert_eq!(BITS_TABLE[4], 2);
        // lhss 5..=8 need 3 bits
        assert!(BITS_TABLE[5..=8].iter().all(|&b| b == 3));
        // lhss 9..=16 need 4 bits
        assert!(BITS_TABLE[9..=16].iter().all(|&b| b == 4));
        // lhss 17..=32 need 5 bits
        assert!(BITS_TABLE[17..=32].iter().all(|&b| b == 5));
        // lhss 33..=64 need 6 bits
        assert!(BITS_TABLE[33..=64].iter().all(|&b| b == 6));
        // lhss 65..=128 need 7 bits
        assert!(BITS_TABLE[65..=128].iter().all(|&b| b == 7));
        // lhss 129..=255 need 8 bits
        assert!(BITS_TABLE[129..=255].iter().all(|&b| b == 8));
    }

    #[test]
    fn mask_table_spot_checks() {
        assert_eq!(MASK_TABLE[2], 0b1);
        assert_eq!(MASK_TABLE[3], 0b11);
        assert_eq!(MASK_TABLE[4], 0b11);
        assert_eq!(MASK_TABLE[5], 0b111);
        assert_eq!(MASK_TABLE[8], 0b111);
        assert_eq!(MASK_TABLE[9], 0b1111);
        assert_eq!(MASK_TABLE[16], 0b1111);
        assert_eq!(MASK_TABLE[17], 0b11111);
        assert_eq!(MASK_TABLE[255], 0b1111_1111);
    }

    // --- get_dit / set_dit round-trips ---

    #[test]
    fn get_set_roundtrip_lhss2_u32() {
        let manip = DynamicDitManip::new(2); // spin-1/2: 1 bit per site
        // 8 sites packed: ...0b10110100 = sites [0..8] = [0,0,1,0,1,1,0,1] (LSB first)
        let s: u32 = 0b10110100;
        assert_eq!(manip.get_dit(s, 0), 0);
        assert_eq!(manip.get_dit(s, 1), 0);
        assert_eq!(manip.get_dit(s, 2), 1);
        assert_eq!(manip.get_dit(s, 3), 0);

        // round-trip: flip site 2 from 1 to 0
        let s2 = manip.set_dit(s, 0, 2);
        assert_eq!(manip.get_dit(s2, 2), 0);
        // other sites unchanged
        assert_eq!(manip.get_dit(s2, 3), manip.get_dit(s, 3));
    }

    #[test]
    fn get_set_roundtrip_lhss3_u64() {
        let manip = DynamicDitManip::new(3); // spin-1: 2 bits per site
        // pack three sites: [1, 2, 0] → bits = 0b00_10_01 = 0x09
        let mut s: u64 = 0;
        s = manip.set_dit(s, 1, 0); // site 0 = 1
        s = manip.set_dit(s, 2, 1); // site 1 = 2
        s = manip.set_dit(s, 0, 2); // site 2 = 0
        assert_eq!(manip.get_dit(s, 0), 1);
        assert_eq!(manip.get_dit(s, 1), 2);
        assert_eq!(manip.get_dit(s, 2), 0);
    }

    #[test]
    fn get_set_roundtrip_lhss4_u32() {
        let manip = DynamicDitManip::new(4); // 2 bits per site (same as lhss=3)
        for val in 0..4 {
            let s = manip.set_dit(0u32, val, 5); // site 5
            assert_eq!(manip.get_dit(s, 5), val);
            // neighbouring sites untouched
            assert_eq!(manip.get_dit(s, 4), 0);
            assert_eq!(manip.get_dit(s, 6), 0);
        }
    }

    // --- DitManip (const generic) matches DynamicDitManip ---

    #[test]
    fn dit_manip_const_matches_dynamic_lhss2() {
        let dyn_manip = DynamicDitManip::new(2);
        let s: u32 = 0b1011_0010;
        for i in 0..8 {
            assert_eq!(
                DitManip::<2>::get_dit(s, i),
                dyn_manip.get_dit(s, i),
                "site {i}"
            );
        }
    }

    #[test]
    fn dit_manip_const_matches_dynamic_lhss5() {
        let dyn_manip = DynamicDitManip::new(5);
        let mut s: u64 = 0;
        s = dyn_manip.set_dit(s, 3, 0);
        s = dyn_manip.set_dit(s, 1, 1);
        s = dyn_manip.set_dit(s, 4, 2);
        for i in 0..3 {
            assert_eq!(
                DitManip::<5>::get_dit(s, i),
                dyn_manip.get_dit(s, i),
                "site {i}"
            );
        }
    }

    // --- get_sub_state / set_sub_state round-trip ---

    #[test]
    fn sub_state_roundtrip_lhss3() {
        let manip = DynamicDitManip::new(3);
        // pack [0, 1, 2] into three sites
        let mut s: u64 = 0;
        s = manip.set_dit(s, 0, 0);
        s = manip.set_dit(s, 1, 1);
        s = manip.set_dit(s, 2, 2);
        let locs = [2, 1, 0]; // last entry = LSB
        // get_sub_state: 2^0 * dit(0) + 3^1 * dit(1) + 3^2 * dit(2)
        //              = 1*0 + 3*1 + 9*2 = 21
        let sub = manip.get_sub_state(s, &locs);
        assert_eq!(sub, 21);

        // set it back onto a blank state and verify
        let s2 = manip.set_sub_state(0u64, sub, &locs);
        assert_eq!(manip.get_dit(s2, 0), 0);
        assert_eq!(manip.get_dit(s2, 1), 1);
        assert_eq!(manip.get_dit(s2, 2), 2);
    }
}
