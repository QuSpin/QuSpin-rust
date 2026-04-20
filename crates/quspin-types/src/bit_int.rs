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

    #[test]
    fn bitset_not_u128() {
        // ~Uint<128>(0) should equal ~u128::from(0u64), i.e. all bits set
        assert_eq!(!U128::from(0u64), U128::MAX);
    }
}
