//! [`BitInt`]: abstraction over the basis-state integer widths used
//! throughout QuSpin (`u32`, `u64`, and `ruint::Uint<BITS, LIMBS>`).
//!
//! Both the trait and its concrete impls live here because the Rust orphan
//! rule forbids `impl BitInt for Uint<N, LIMBS>` in any downstream crate
//! (both trait and type would be foreign). Keeping the impls alongside the
//! trait definition is also cleaner architecturally — `quspin-types`
//! already holds all the fundamental trait abstractions for the workspace.
//!
//! `quspin-bitbasis` re-exports `BitInt` so existing
//! `quspin_bitbasis::BitInt` imports keep resolving.

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};

use ruint::Uint;

/// Trait abstracting over all supported basis-state integer widths:
/// `u32`, `u64`, and `ruint::Uint<BITS, LIMBS>`.
///
/// Associated constants mirror `bit_info<I>` from the original C++
/// implementation.
pub trait BitInt:
    Copy
    + Send
    + Sync
    + Default
    + Debug
    + Eq
    + Ord
    + Hash
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
{
    const BITS: u32;
    const LD_BITS: u32;
    const BYTES: u32;

    /// Widen a `u64` into `Self`. Truncates for narrower types (u32),
    /// zero-extends for wider types (`ruint::Uint`).
    fn from_u64(v: u64) -> Self;

    /// Narrow `self` to `usize`. Only valid when `self` is known to be small
    /// (e.g. after masking to a dit value); no bounds checking in release
    /// mode.
    fn to_usize(self) -> usize;

    /// Count the number of set bits (Hamming weight / popcount).
    fn count_ones(self) -> u32;
}

// --- u32 ---

impl BitInt for u32 {
    const BITS: u32 = 32;
    const LD_BITS: u32 = 5;
    const BYTES: u32 = 4;

    #[inline]
    fn from_u64(v: u64) -> Self {
        v as u32
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn count_ones(self) -> u32 {
        u32::count_ones(self)
    }
}

// --- u64 ---

impl BitInt for u64 {
    const BITS: u32 = 64;
    const LD_BITS: u32 = 6;
    const BYTES: u32 = 8;

    #[inline]
    fn from_u64(v: u64) -> Self {
        v
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn count_ones(self) -> u32 {
        u64::count_ones(self)
    }
}

// --- u128 ---
//
// Not a native register width on any mainstream target -- LLVM lowers it to
// a 64-bit register pair -- but it is a first-class type the optimiser
// models directly, rather than an array of limbs behind a generic. That
// makes it a strictly better choice than `Uint<128, 2>` for the 65..=128
// bit tier: 4.1x on the Benes orbit loop, 2.6x on the dit sweep, measured
// at identical physics. Repointing the dispatch tier is tracked in #123.

impl BitInt for u128 {
    const BITS: u32 = 128;
    const LD_BITS: u32 = 7;
    const BYTES: u32 = 16;

    #[inline]
    fn from_u64(v: u64) -> Self {
        v as u128
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn count_ones(self) -> u32 {
        u128::count_ones(self)
    }
}

// --- ruint::Uint<N, LIMBS> ---

impl<const N: usize, const LIMBS: usize> BitInt for Uint<N, LIMBS> {
    const BITS: u32 = N as u32;
    const LD_BITS: u32 = (N as u32).trailing_zeros();
    const BYTES: u32 = (N / 8) as u32;

    #[inline]
    fn from_u64(v: u64) -> Self {
        // Build the little-endian limb array: limb 0 = v, rest = 0.
        let mut limbs = [0u64; LIMBS];
        if LIMBS > 0 {
            limbs[0] = v;
        }
        Uint::from_limbs(limbs)
    }

    #[inline]
    fn to_usize(self) -> usize {
        // Limbs are little-endian (limb 0 is least significant 64-bit chunk).
        // Safe because this is only called after masking to a small dit value.
        *self.as_limbs().first().unwrap_or(&0) as usize
    }

    #[inline]
    fn count_ones(self) -> u32 {
        // ruint::Uint::count_ones returns usize; cast to u32.
        Uint::<N, LIMBS>::count_ones(&self) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type U128 = Uint<128, 2>;
    type U256 = Uint<256, 4>;
    type U1024 = Uint<1024, 16>;

    // --- associated constants ---

    #[test]
    fn constants_u32() {
        assert_eq!(<u32 as BitInt>::BITS, 32);
        assert_eq!(<u32 as BitInt>::LD_BITS, 5);
        assert_eq!(<u32 as BitInt>::BYTES, 4);
    }

    #[test]
    fn constants_u64() {
        assert_eq!(<u64 as BitInt>::BITS, 64);
        assert_eq!(<u64 as BitInt>::LD_BITS, 6);
        assert_eq!(<u64 as BitInt>::BYTES, 8);
    }

    #[test]
    fn constants_u128() {
        assert_eq!(<U128 as BitInt>::BITS, 128);
        assert_eq!(<U128 as BitInt>::LD_BITS, 7);
        assert_eq!(<U128 as BitInt>::BYTES, 16);
    }

    #[test]
    fn constants_u256() {
        assert_eq!(<U256 as BitInt>::BITS, 256);
        assert_eq!(<U256 as BitInt>::LD_BITS, 8);
        assert_eq!(<U256 as BitInt>::BYTES, 32);
    }

    #[test]
    fn constants_u1024() {
        assert_eq!(<U1024 as BitInt>::BITS, 1024);
        assert_eq!(<U1024 as BitInt>::LD_BITS, 10);
        assert_eq!(<U1024 as BitInt>::BYTES, 128);
    }

    // --- from_u64 / to_usize ---

    #[test]
    fn from_u64_and_to_usize() {
        assert_eq!(u32::from_u64(42).to_usize(), 42);
        assert_eq!(u64::from_u64(42).to_usize(), 42);
        assert_eq!(U128::from_u64(42).to_usize(), 42);
        assert_eq!(U256::from_u64(42).to_usize(), 42);
    }

    // --- bitwise ops on primitives ---

    #[test]
    fn bitops_u32() {
        let a: u32 = 0b1100;
        let b: u32 = 0b1010;
        assert_eq!(a & b, 0b1000);
        assert_eq!(a | b, 0b1110);
        assert_eq!(a ^ b, 0b0110);
        assert_eq!(a << 2usize, 0b11_0000);
        assert_eq!(a >> 1usize, 0b0110);
        assert_eq!(!0u32, u32::MAX);
    }

    #[test]
    fn bitops_u64() {
        let a: u64 = 0b1100;
        let b: u64 = 0b1010;
        assert_eq!(a & b, 0b1000);
        assert_eq!(a | b, 0b1110);
        assert_eq!(a ^ b, 0b0110);
        assert_eq!(a << 3usize, 0b110_0000);
        assert_eq!(!0u64, u64::MAX);
    }

    // --- bitwise ops on ruint::Uint (ported from cpp/test/bitset.cpp) ---

    #[test]
    fn bitset_shift_u128() {
        assert_eq!(U128::from(1230u64) << 36usize, U128::from(1230u64 << 36));
        assert_eq!(
            U128::from(2346020u64) << 23usize,
            U128::from(2346020u64 << 23)
        );
        assert_eq!(
            U128::from(1232132u64) << 42usize,
            U128::from(1232132u64 << 42)
        );
        assert_eq!(U128::from(1230u64) >> 2usize, U128::from(1230u64 >> 2));
    }

    #[test]
    fn bitset_bitwise_u128() {
        let x = U128::from(1230u64);
        let y = U128::from(123u64);
        assert_eq!(x & y, U128::from(1230u64 & 123u64));
        assert_eq!(x | y, U128::from(1230u64 | 123u64));
        assert_eq!(x ^ y, U128::from(1230u64 ^ 123u64));
    }

    // --- native u128 ---
    //
    // The `U128` alias above is `Uint<128, 2>`, and the dispatch ladder
    // still routes the 65..=128 bit tier through that type, so nothing else
    // in the suite touches `impl BitInt for u128`. These cover it directly,
    // including values above bit 64 where a one-limb implementation would
    // silently lose the high half.

    #[test]
    fn constants_native_u128() {
        assert_eq!(<u128 as BitInt>::BITS, 128);
        assert_eq!(<u128 as BitInt>::LD_BITS, 7);
        assert_eq!(<u128 as BitInt>::BYTES, 16);
    }

    #[test]
    fn native_u128_from_u64_zero_extends() {
        assert_eq!(<u128 as BitInt>::from_u64(0), 0u128);
        assert_eq!(<u128 as BitInt>::from_u64(42), 42u128);
        // u64::MAX must zero-extend, not sign-extend.
        assert_eq!(<u128 as BitInt>::from_u64(u64::MAX), u64::MAX as u128);
        assert_eq!(<u128 as BitInt>::from_u64(42).to_usize(), 42);
    }

    #[test]
    fn native_u128_count_ones_spans_both_halves() {
        assert_eq!(BitInt::count_ones(0u128), 0);
        assert_eq!(BitInt::count_ones(u128::MAX), 128);
        // One bit in each 64-bit half.
        let x = (1u128 << 3) | (1u128 << 100);
        assert_eq!(BitInt::count_ones(x), 2);
        // Entirely in the high half, where a u64-backed impl would see 0.
        assert_eq!(BitInt::count_ones(u128::MAX << 64), 64);
    }

    #[test]
    fn native_u128_shifts_cross_the_64_bit_boundary() {
        assert_eq!(1u128 << 64usize, 1u128 << 64);
        assert_eq!((1u128 << 127usize) >> 127usize, 1u128);
        // A value in the low half shifted into the high half and back.
        let v = 0xDEAD_BEEF_CAFE_F00Du128;
        assert_eq!((v << 64usize) >> 64usize, v);
        assert_eq!((v << 64usize).to_usize(), 0, "low limb must be cleared");
    }

    #[test]
    fn native_u128_bitops() {
        let a = (0b1100u128 << 64) | 0b1100;
        let b = (0b1010u128 << 64) | 0b1010;
        assert_eq!(a & b, (0b1000u128 << 64) | 0b1000);
        assert_eq!(a | b, (0b1110u128 << 64) | 0b1110);
        assert_eq!(a ^ b, (0b0110u128 << 64) | 0b0110);
        assert_eq!(!0u128, u128::MAX);
    }

    /// Native `u128` and `Uint<128, 2>` must agree bit for bit, so the
    /// dispatch tier can be repointed without changing behaviour.
    #[test]
    fn native_u128_agrees_with_uint128() {
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng
        };
        for _ in 0..200 {
            let (lo, hi) = (next(), next());
            let n: u128 = ((hi as u128) << 64) | lo as u128;
            let u = (U128::from(hi) << 64usize) | U128::from(lo);
            assert_eq!(u.as_limbs(), &[lo, hi], "fixture mismatch");

            assert_eq!(BitInt::count_ones(n), BitInt::count_ones(u));
            for sh in [0usize, 1, 63, 64, 65, 127] {
                let ns = n << sh;
                let us = u << sh;
                assert_eq!(
                    [ns as u64, (ns >> 64) as u64],
                    *us.as_limbs(),
                    "shl {n:#x} << {sh}"
                );
                let nr = n >> sh;
                let ur = u >> sh;
                assert_eq!(
                    [nr as u64, (nr >> 64) as u64],
                    *ur.as_limbs(),
                    "shr {n:#x} >> {sh}"
                );
            }
        }
    }

    #[test]
    fn bitset_not_u128() {
        // ~Uint<128>(0) should equal ~u128::from(0u64), i.e. all bits set
        assert_eq!(!U128::from(0u64), U128::MAX);
    }
}
