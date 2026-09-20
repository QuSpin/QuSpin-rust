//! Three-way experiment: where does the multi-limb cost actually come from?
//!
//! `bitint_width.rs` shows `Uint<256,4>` is 28.7x slower than `u64` on the
//! Benes hot loop *at identical physics*. That could be either of two very
//! different things, and the fix differs completely:
//!
//!   (a) multi-limb arithmetic is inherently expensive, or
//!   (b) `ruint`'s runtime-shift codegen is expensive.
//!
//! So this file compares three 256-bit-capacity types on the same work:
//!
//!   `Uint<256,4>` -- ruint, compile-time 4 limbs        (the status quo)
//!   `Fixed4`      -- hand-rolled, compile-time 4 limbs   (isolates (b))
//!   `DynLimb`     -- hand-rolled, *runtime* limb count   (the proposal)
//!
//! `Uint<256,4>` vs `Fixed4` answers (a) vs (b).
//! `Fixed4` vs `DynLimb` is the cost of making the width a runtime value --
//! the question that decides whether one representation can replace the
//! whole `u32/u64/U128/U256` ladder.
//!
//! `u64` is included as the floor: whatever replaces the ladder has to be
//! judged against what a 32-site problem costs today.

// Limb indexing is deliberate here: the loop index addresses two arrays at
// different offsets, which iterators cannot express as clearly.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use quspin_bitbasis::benes::{BenesNetwork, gen_benes};
use quspin_bitbasis::manip::DynamicDitManip;
use quspin_types::BitInt;
use ruint::Uint;

type U256 = Uint<256, 4>;

const N_SITES: usize = 32;
const N_STATES: usize = 4096;
const CAP: usize = 4; // limbs of inline capacity == 256 bits

// ===========================================================================
// Fixed4 -- hand-rolled, compile-time 4 limbs
// ===========================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Fixed4 {
    l: [u64; CAP],
}

impl Ord for Fixed4 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Limbs are little-endian: compare most-significant first.
        for i in (0..CAP).rev() {
            match self.l[i].cmp(&other.l[i]) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}
impl PartialOrd for Fixed4 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

macro_rules! fixed_bitop {
    ($tr:ident, $m:ident, $op:tt) => {
        impl std::ops::$tr for Fixed4 {
            type Output = Self;
            #[inline]
            fn $m(self, rhs: Self) -> Self {
                let mut o = [0u64; CAP];
                for i in 0..CAP {
                    o[i] = self.l[i] $op rhs.l[i];
                }
                Fixed4 { l: o }
            }
        }
    };
}
fixed_bitop!(BitAnd, bitand, &);
fixed_bitop!(BitOr, bitor, |);
fixed_bitop!(BitXor, bitxor, ^);

impl std::ops::Not for Fixed4 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        let mut o = [0u64; CAP];
        for i in 0..CAP {
            o[i] = !self.l[i];
        }
        Fixed4 { l: o }
    }
}

impl std::ops::Shl<usize> for Fixed4 {
    type Output = Self;
    #[inline]
    fn shl(self, s: usize) -> Self {
        let mut o = [0u64; CAP];
        let (ls, bs) = (s / 64, s % 64);
        for i in (0..CAP).rev() {
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
        Fixed4 { l: o }
    }
}

impl std::ops::Shr<usize> for Fixed4 {
    type Output = Self;
    #[inline]
    fn shr(self, s: usize) -> Self {
        let mut o = [0u64; CAP];
        let (ls, bs) = (s / 64, s % 64);
        for i in 0..CAP {
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
        Fixed4 { l: o }
    }
}

impl BitInt for Fixed4 {
    const BITS: u32 = 256;
    const LD_BITS: u32 = 8;
    const BYTES: u32 = 32;
    #[inline]
    fn from_u64(v: u64) -> Self {
        let mut l = [0u64; CAP];
        l[0] = v;
        Fixed4 { l }
    }
    #[inline]
    fn to_usize(self) -> usize {
        self.l[0] as usize
    }
    #[inline]
    fn count_ones(self) -> u32 {
        self.l.iter().map(|x| x.count_ones()).sum()
    }
}

// ===========================================================================
// DynLimb -- hand-rolled, runtime limb count
// ===========================================================================
//
// Invariant: `l[i] == 0` for all `i >= n`. `n` is an *upper bound* on the
// significant limbs, not an exact count -- that keeps the operators cheap
// (no renormalisation scan) while staying sound, because every operator
// below propagates a bound that is still valid.
//
// `Eq`/`Ord`/`Hash` deliberately run over all CAP limbs: `n` is only a
// bound, so two equal values may carry different `n`, and the hash must not
// depend on it. Those three therefore cost the same as `Fixed4` -- the win
// is concentrated in the bit operators, which is exactly where the Benes
// and dit-sweep loops live.

#[derive(Clone, Copy, Debug, Default)]
pub struct DynLimb {
    l: [u64; CAP],
    n: u8,
}

impl PartialEq for DynLimb {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.l == other.l
    }
}
impl Eq for DynLimb {}
impl std::hash::Hash for DynLimb {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.l.hash(state);
    }
}
impl Ord for DynLimb {
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
impl PartialOrd for DynLimb {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

macro_rules! dyn_bitop {
    ($tr:ident, $m:ident, $op:tt, $bound:expr) => {
        impl std::ops::$tr for DynLimb {
            type Output = Self;
            #[inline]
            fn $m(self, rhs: Self) -> Self {
                let f: fn(u8, u8) -> u8 = $bound;
                let n = f(self.n, rhs.n);
                let mut o = [0u64; CAP];
                for i in 0..(n as usize) {
                    o[i] = self.l[i] $op rhs.l[i];
                }
                DynLimb { l: o, n }
            }
        }
    };
}
// AND cannot produce a set bit where either side is zero -> min.
dyn_bitop!(BitAnd, bitand, &, |a, b| a.min(b));
// OR/XOR can produce a set bit wherever either side has one -> max.
dyn_bitop!(BitOr, bitor, |, |a, b| a.max(b));
dyn_bitop!(BitXor, bitxor, ^, |a, b| a.max(b));

impl std::ops::Not for DynLimb {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        // Complement sets the high limbs, so the bound goes to full width.
        let mut o = [0u64; CAP];
        for i in 0..CAP {
            o[i] = !self.l[i];
        }
        DynLimb { l: o, n: CAP as u8 }
    }
}

impl std::ops::Shl<usize> for DynLimb {
    type Output = Self;
    #[inline]
    fn shl(self, s: usize) -> Self {
        let (ls, bs) = (s / 64, s % 64);
        let n = ((self.n as usize + ls + 1).min(CAP)) as u8;
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
        DynLimb { l: o, n }
    }
}

impl std::ops::Shr<usize> for DynLimb {
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
        DynLimb { l: o, n }
    }
}

impl BitInt for DynLimb {
    const BITS: u32 = 256;
    const LD_BITS: u32 = 8;
    const BYTES: u32 = 32;
    #[inline]
    fn from_u64(v: u64) -> Self {
        let mut l = [0u64; CAP];
        l[0] = v;
        DynLimb {
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

// ===========================================================================
// Fixtures (mirrors bitint_width.rs)
// ===========================================================================

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn states<B: BitInt>() -> Vec<B> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mask = (1u64 << N_SITES) - 1;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(N_STATES);
    while out.len() < N_STATES {
        let v = rng.next() & mask;
        if seen.insert(v) {
            out.push(B::from_u64(v));
        }
    }
    out
}

fn cyclic_network<B: BitInt>() -> BenesNetwork<B> {
    let mut tgt: Vec<Option<usize>> = vec![None; B::BITS as usize];
    for dst in 0..N_SITES {
        tgt[dst] = Some((dst + N_SITES - 1) % N_SITES);
    }
    gen_benes::<B>(&tgt)
}

type G<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

// ===========================================================================
// Groups
// ===========================================================================

fn bench_benes(c: &mut Criterion) {
    let mut g = c.benchmark_group("mw_benes");
    g.throughput(Throughput::Elements(N_STATES as u64));
    fn run<B: BitInt>(g: &mut G<'_>, name: &str) {
        let net = cyclic_network::<B>();
        let st = states::<B>();
        g.bench_with_input(BenchmarkId::from_parameter(name), &st, |b, st| {
            b.iter(|| {
                let mut acc = B::from_u64(0);
                for &s in st {
                    acc = acc ^ net.apply(black_box(s));
                }
                black_box(acc)
            })
        });
    }
    run::<u64>(&mut g, "u64");
    run::<U256>(&mut g, "Uint<256>");
    run::<Fixed4>(&mut g, "Fixed4");
    run::<DynLimb>(&mut g, "DynLimb");
    g.finish();
}

fn bench_dit_sweep(c: &mut Criterion) {
    let mut g = c.benchmark_group("mw_dit_sweep");
    g.throughput(Throughput::Elements((N_STATES * N_SITES) as u64));
    fn run<B: BitInt>(g: &mut G<'_>, name: &str) {
        let manip = DynamicDitManip::new(2);
        let st = states::<B>();
        g.bench_with_input(BenchmarkId::from_parameter(name), &st, |b, st| {
            b.iter(|| {
                let mut acc = 0usize;
                for &s in st {
                    let mut t = s;
                    for i in 0..N_SITES {
                        let d = manip.get_dit(black_box(t), i);
                        acc += d;
                        t = manip.set_dit(t, d ^ 1, i);
                    }
                    acc += t.count_ones() as usize;
                }
                black_box(acc)
            })
        });
    }
    run::<u64>(&mut g, "u64");
    run::<U256>(&mut g, "Uint<256>");
    run::<Fixed4>(&mut g, "Fixed4");
    run::<DynLimb>(&mut g, "DynLimb");
    g.finish();
}

fn bench_hashmap(c: &mut Criterion) {
    let mut g = c.benchmark_group("mw_hashmap");
    g.throughput(Throughput::Elements(N_STATES as u64));
    fn run<B: BitInt>(g: &mut G<'_>, name: &str) {
        let st = states::<B>();
        g.bench_with_input(BenchmarkId::from_parameter(name), &st, |b, st| {
            b.iter(|| {
                let mut m: HashMap<B, usize> = HashMap::with_capacity(st.len());
                for (i, &s) in st.iter().enumerate() {
                    m.insert(s, i);
                }
                let mut acc = 0usize;
                for &s in st {
                    acc += m[&black_box(s)];
                }
                black_box(acc)
            })
        });
    }
    run::<u64>(&mut g, "u64");
    run::<U256>(&mut g, "Uint<256>");
    run::<Fixed4>(&mut g, "Fixed4");
    run::<DynLimb>(&mut g, "DynLimb");
    g.finish();
}

fn bench_sort_search(c: &mut Criterion) {
    let mut g = c.benchmark_group("mw_sort_search");
    g.throughput(Throughput::Elements(N_STATES as u64));
    fn run<B: BitInt>(g: &mut G<'_>, name: &str) {
        let st = states::<B>();
        g.bench_with_input(BenchmarkId::from_parameter(name), &st, |b, st| {
            b.iter(|| {
                let mut v = st.clone();
                v.sort_unstable();
                let mut acc = 0usize;
                for &s in st {
                    acc += v.binary_search(&black_box(s)).unwrap_or(0);
                }
                black_box(acc)
            })
        });
    }
    run::<u64>(&mut g, "u64");
    run::<U256>(&mut g, "Uint<256>");
    run::<Fixed4>(&mut g, "Fixed4");
    run::<DynLimb>(&mut g, "DynLimb");
    g.finish();
}

criterion_group!(
    benches,
    bench_benes,
    bench_dit_sweep,
    bench_hashmap,
    bench_sort_search
);

// `harness = false` means `#[cfg(test)]` modules never run, so correctness is
// wired into `main` as a gate instead: the benchmarks cannot be reported
// without the prototypes first proving they agree with `u64` and `ruint`.
fn main() {
    tests::verify_all();
    eprintln!("prototype correctness: OK (Fixed4, DynLimb vs u64 + Uint<256,4>)");
    benches();
    Criterion::default().configure_from_args().final_summary();
}

// ===========================================================================
// Correctness -- the benchmark is meaningless if the prototypes are wrong.
// Both new types are cross-checked against `u64` (and against `Uint<256,4>`,
// which exercises bits above limb 0) on every operator the hot loops use.
// ===========================================================================

mod tests {
    use super::*;

    pub fn verify_all() {
        agrees_with_u64();
        shifts_agree_with_ruint_across_limbs();
        benes_agrees_across_representations();
        dit_ops_agree();
        dynlimb_identity_ignores_bound();
    }

    fn rng_vals() -> Vec<u64> {
        let mut r = Rng(12345);
        let mut v = vec![0u64, 1, u64::MAX, 0xDEAD_BEEF_CAFE_F00D];
        for _ in 0..200 {
            v.push(r.next());
        }
        v
    }

    /// Low-limb agreement with `u64` for every operator, at every shift that
    /// keeps the result inside 64 bits.
    fn agrees_with_u64() {
        for &a in &rng_vals() {
            for &b in &rng_vals()[..16] {
                let (fa, fb) = (Fixed4::from_u64(a), Fixed4::from_u64(b));
                let (da, db) = (DynLimb::from_u64(a), DynLimb::from_u64(b));

                assert_eq!((fa & fb).to_usize() as u64, a & b, "and {a:#x} {b:#x}");
                assert_eq!((da & db).to_usize() as u64, a & b, "and {a:#x} {b:#x}");
                assert_eq!((fa | fb).to_usize() as u64, a | b, "or {a:#x} {b:#x}");
                assert_eq!((da | db).to_usize() as u64, a | b, "or {a:#x} {b:#x}");
                assert_eq!((fa ^ fb).to_usize() as u64, a ^ b, "xor {a:#x} {b:#x}");
                assert_eq!((da ^ db).to_usize() as u64, a ^ b, "xor {a:#x} {b:#x}");
                assert_eq!((!fa).to_usize() as u64, !a, "not {a:#x}");
                assert_eq!((!da).to_usize() as u64, !a, "not {a:#x}");

                assert_eq!(BitInt::count_ones(fa), a.count_ones(), "popcnt {a:#x}");
                assert_eq!(BitInt::count_ones(da), a.count_ones(), "popcnt {a:#x}");
                assert_eq!((a < b), (da < db), "ord {a:#x} {b:#x}");
                assert_eq!((a < b), (fa < fb), "ord {a:#x} {b:#x}");
            }
        }
    }

    /// Shifts must agree with `Uint<256,4>` across limb boundaries -- the
    /// part `u64` cannot check, and where `DynLimb`'s `n` bound is at risk.
    fn shifts_agree_with_ruint_across_limbs() {
        for &a in &rng_vals() {
            let (ua, fa, da) = (U256::from_u64(a), Fixed4::from_u64(a), DynLimb::from_u64(a));
            for s in [0usize, 1, 7, 31, 63, 64, 65, 100, 127, 128, 191, 192, 255] {
                let (ul, fl, dl) = (ua << s, fa << s, da << s);
                assert_eq!(fl.l, *ul.as_limbs(), "shl {a:#x} << {s}");
                assert_eq!(dl.l, *ul.as_limbs(), "shl {a:#x} << {s}");
                // DynLimb's bound must never understate the real extent.
                for i in (dl.n as usize)..CAP {
                    assert_eq!(dl.l[i], 0, "shl bound violated: {a:#x} << {s}");
                }

                let wide = ua << 150; // push bits high, then shift back down
                let fw = fa << 150;
                let dw = da << 150;
                let ur: U256 = wide >> s;
                let (fr, dr) = (fw >> s, dw >> s);
                assert_eq!(fr.l, *ur.as_limbs(), "shr {a:#x} >> {s}");
                assert_eq!(dr.l, *ur.as_limbs(), "shr {a:#x} >> {s}");
                for i in (dr.n as usize)..CAP {
                    assert_eq!(dr.l[i], 0, "shr bound violated: {a:#x} >> {s}");
                }
            }
        }
    }

    /// The whole point: a Benes permutation must produce identical results
    /// in all four representations.
    fn benes_agrees_across_representations() {
        let n64 = cyclic_network::<u64>();
        let nu = cyclic_network::<U256>();
        let nf = cyclic_network::<Fixed4>();
        let nd = cyclic_network::<DynLimb>();
        let mask = (1u64 << N_SITES) - 1;
        let mut r = Rng(999);
        for _ in 0..500 {
            let v = r.next() & mask;
            let want = n64.apply(v);
            assert_eq!(
                nu.apply(U256::from_u64(v)).as_limbs()[0],
                want,
                "ruint {v:#x}"
            );
            assert_eq!(nf.apply(Fixed4::from_u64(v)).l[0], want, "Fixed4 {v:#x}");
            assert_eq!(nd.apply(DynLimb::from_u64(v)).l[0], want, "DynLimb {v:#x}");
        }
    }

    /// `get_dit`/`set_dit` round-trips must match across representations.
    fn dit_ops_agree() {
        let manip = DynamicDitManip::new(2);
        let mask = (1u64 << N_SITES) - 1;
        let mut r = Rng(4242);
        for _ in 0..500 {
            let v = r.next() & mask;
            let (mut tf, mut td) = (Fixed4::from_u64(v), DynLimb::from_u64(v));
            let mut t64 = v;
            for i in 0..N_SITES {
                let d = manip.get_dit(t64, i);
                assert_eq!(manip.get_dit(tf, i), d, "get {v:#x}@{i}");
                assert_eq!(manip.get_dit(td, i), d, "get {v:#x}@{i}");
                t64 = manip.set_dit(t64, d ^ 1, i);
                tf = manip.set_dit(tf, d ^ 1, i);
                td = manip.set_dit(td, d ^ 1, i);
                assert_eq!(tf.l[0], t64, "set {v:#x}@{i}");
                assert_eq!(td.l[0], t64, "set {v:#x}@{i}");
            }
        }
    }

    /// Equal values must hash and compare equal regardless of how they were
    /// built -- `DynLimb`'s `n` is a bound, not part of its identity.
    fn dynlimb_identity_ignores_bound() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = DynLimb::from_u64(0xFF);
        // Same value, but routed through `!` so its bound is CAP, not 1.
        let b = !!DynLimb::from_u64(0xFF);
        assert_eq!(a, b);
        assert_ne!(a.n, b.n, "fixture must actually differ in bound");
        let h = |x: &DynLimb| {
            let mut s = DefaultHasher::new();
            x.hash(&mut s);
            s.finish()
        };
        assert_eq!(h(&a), h(&b), "hash must not depend on the bound");
        assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
    }
}
