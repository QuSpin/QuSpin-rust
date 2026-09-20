//! Shared fixtures and the runtime-limb prototype, used by `largeint.rs`
//! and `benes_stages.rs`.

// Limb indexing is deliberate here: the loop index addresses two arrays at
// different offsets, which iterators cannot express as clearly.
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use quspin_types::BitInt;

/// xorshift64* — deterministic, no dev-dependency on `rand`.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

// ===========================================================================
// Dyn<CAP> -- runtime limb count, CAP limbs of inline capacity
// ===========================================================================
//
// Invariant: `l[i] == 0` for all `i >= n`, with `n` an upper bound rather
// than an exact count -- that keeps the operators cheap (no renormalisation
// scan) while staying sound, because every operator propagates a bound that
// is still valid.
//
// `Eq`/`Ord`/`Hash` run over all CAP limbs so identity never depends on the
// bound: two equal values may carry different `n`.

#[derive(Clone, Copy, Debug)]
pub struct Dyn<const CAP: usize> {
    pub l: [u64; CAP],
    pub n: u16,
}

impl<const CAP: usize> Default for Dyn<CAP> {
    #[inline]
    fn default() -> Self {
        Dyn {
            l: [0u64; CAP],
            n: 0,
        }
    }
}

impl<const CAP: usize> PartialEq for Dyn<CAP> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.l == other.l
    }
}
impl<const CAP: usize> Eq for Dyn<CAP> {}
impl<const CAP: usize> std::hash::Hash for Dyn<CAP> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.l.hash(state);
    }
}
impl<const CAP: usize> Ord for Dyn<CAP> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for i in (0..CAP).rev() {
            match self.l[i].cmp(&other.l[i]) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}
impl<const CAP: usize> PartialOrd for Dyn<CAP> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

macro_rules! dyn_bitop {
    ($tr:ident, $m:ident, $op:tt, $bound:expr) => {
        impl<const CAP: usize> std::ops::$tr for Dyn<CAP> {
            type Output = Self;
            #[inline]
            fn $m(self, rhs: Self) -> Self {
                let f: fn(u16, u16) -> u16 = $bound;
                let n = f(self.n, rhs.n);
                let mut o = [0u64; CAP];
                for i in 0..(n as usize) {
                    o[i] = self.l[i] $op rhs.l[i];
                }
                Dyn { l: o, n }
            }
        }
    };
}
// AND cannot set a bit where either side is zero -> min.
dyn_bitop!(BitAnd, bitand, &, |a, b| a.min(b));
// OR/XOR can set a bit wherever either side has one -> max.
dyn_bitop!(BitOr, bitor, |, |a, b| a.max(b));
dyn_bitop!(BitXor, bitxor, ^, |a, b| a.max(b));

impl<const CAP: usize> std::ops::Not for Dyn<CAP> {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        // Complement sets the high limbs, so the bound goes to full width.
        let mut o = [0u64; CAP];
        for i in 0..CAP {
            o[i] = !self.l[i];
        }
        Dyn {
            l: o,
            n: CAP as u16,
        }
    }
}

impl<const CAP: usize> std::ops::Shl<usize> for Dyn<CAP> {
    type Output = Self;
    #[inline]
    fn shl(self, s: usize) -> Self {
        let (ls, bs) = (s / 64, s % 64);
        let n = ((self.n as usize + ls + 1).min(CAP)) as u16;
        let mut o = [0u64; CAP];
        for i in (0..(n as usize)).rev() {
            if i < ls {
                break;
            }
            let lo = self.l[i - ls];
            let hi = if bs == 0 || i == ls {
                0
            } else {
                self.l[i - ls - 1] >> (64 - bs)
            };
            o[i] = (lo << bs) | hi;
        }
        Dyn { l: o, n }
    }
}

impl<const CAP: usize> std::ops::Shr<usize> for Dyn<CAP> {
    type Output = Self;
    #[inline]
    fn shr(self, s: usize) -> Self {
        let (ls, bs) = (s / 64, s % 64);
        let n = self.n;
        let mut o = [0u64; CAP];
        for i in 0..(n as usize) {
            if i + ls >= CAP {
                break;
            }
            let lo = self.l[i + ls];
            let hi = if bs == 0 || i + ls + 1 >= CAP {
                0
            } else {
                self.l[i + ls + 1] << (64 - bs)
            };
            o[i] = (lo >> bs) | hi;
        }
        Dyn { l: o, n }
    }
}

macro_rules! impl_bitint_dyn {
    ($cap:expr, $bits:expr, $ld:expr) => {
        impl BitInt for Dyn<$cap> {
            const BITS: u32 = $bits;
            const LD_BITS: u32 = $ld;
            const BYTES: u32 = $bits / 8;
            #[inline]
            fn from_u64(v: u64) -> Self {
                let mut l = [0u64; $cap];
                l[0] = v;
                Dyn {
                    l,
                    n: if v == 0 { 0 } else { 1 },
                }
            }
            #[inline]
            fn to_usize(self) -> usize {
                self.l[0] as usize
            }
            #[inline]
            fn count_ones(self) -> u32 {
                self.l[..self.n as usize]
                    .iter()
                    .map(|x| x.count_ones())
                    .sum()
            }
        }
    };
}
impl_bitint_dyn!(8, 512, 9);
impl_bitint_dyn!(16, 1024, 10);
impl_bitint_dyn!(32, 2048, 11);
impl_bitint_dyn!(64, 4096, 12);

// ===========================================================================
// Shared fixtures
// ===========================================================================

/// A cyclic site shift on `n_sites` bits — the canonical translation
/// symmetry, and the permutation `BenesLatticeElement` is built from.
/// Note `gen_benes` requires the target slice to be `B::BITS` long, so the
/// network is always container-width regardless of `n_sites`.
pub fn cyclic_network<B: BitInt>(n_sites: usize) -> quspin_bitbasis::benes::BenesNetwork<B> {
    let mut tgt: Vec<Option<usize>> = vec![None; B::BITS as usize];
    for dst in 0..n_sites {
        tgt[dst] = Some((dst + n_sites - 1) % n_sites);
    }
    quspin_bitbasis::benes::gen_benes::<B>(&tgt)
}

/// `count` states occupying the low `n_sites` bits, spread across limbs.
pub fn states<B: BitInt>(n_sites: usize, count: usize) -> Vec<B> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut s = B::from_u64(0);
        for w in 0..n_sites.div_ceil(64) {
            let bits_here = (n_sites - w * 64).min(64);
            let mask = if bits_here == 64 {
                u64::MAX
            } else {
                (1u64 << bits_here) - 1
            };
            s = s | (B::from_u64(rng.next() & mask) << (w * 64));
        }
        out.push(s);
    }
    out
}
