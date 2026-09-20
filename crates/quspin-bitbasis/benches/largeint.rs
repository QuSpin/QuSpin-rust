//! Does a runtime limb count beat static `Uint` *above* native width?
//!
//! `multiword.rs` compared everything at 256 bits and concluded the runtime
//! limb count loses -- but it judged `DynLimb` against a hand-rolled
//! *fixed* 4-limb type. Against `ruint`, which is what the code actually
//! uses, `DynLimb` was already ahead on the Benes loop (590 vs 968 us).
//!
//! That reframes the question. `u64` is untouchable and the ladder should
//! keep it. But the `large-int` ladder -- `Uint<512..8192>`, five gated
//! widths, the bulk of the monomorphization -- is a different decision: if
//! one runtime-limb type matches or beats static `Uint` across that range,
//! it could replace all five.
//!
//! So this sweeps 512 / 1024 / 2048 / 4096 bits and compares, at matched
//! capacity:
//!
//!   `Uint<W, W/64>`  -- static, the status quo
//!   `Dyn<W/64>`      -- runtime limb count, same inline capacity
//!
//! Two site counts per width, because they exercise opposite regimes:
//!
//!   NARROW (64 sites)  -- state occupies 1 limb of a wide container. This
//!                         is the one-type scenario, where the runtime
//!                         bound has something to skip.
//!   DENSE  (W/2 sites) -- state fills half the container, so the bound is
//!                         saturated and there is nothing to skip. This is
//!                         the case that has to not regress.

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

const N_STATES: usize = 1024;

// ===========================================================================
// Dyn<CAP> -- runtime limb count, CAP limbs of inline capacity
// ===========================================================================
//
// Same invariant as `multiword.rs`: `l[i] == 0` for all `i >= n`, with `n`
// an upper bound rather than an exact count. `Eq`/`Ord`/`Hash` run over all
// CAP limbs so identity never depends on the bound.

#[derive(Clone, Copy, Debug)]
pub struct Dyn<const CAP: usize> {
    l: [u64; CAP],
    n: u16,
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
dyn_bitop!(BitAnd, bitand, &, |a, b| a.min(b));
dyn_bitop!(BitOr, bitor, |, |a, b| a.max(b));
dyn_bitop!(BitXor, bitxor, ^, |a, b| a.max(b));

impl<const CAP: usize> std::ops::Not for Dyn<CAP> {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
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
// Fixtures
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

/// States occupying the low `n_sites` bits, spread across limbs.
fn states<B: BitInt>(n_sites: usize) -> Vec<B> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(N_STATES);
    for _ in 0..N_STATES {
        let mut s = B::from_u64(0);
        // Fill every 64-bit window the site count reaches.
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

fn cyclic_network<B: BitInt>(n_sites: usize) -> BenesNetwork<B> {
    let mut tgt: Vec<Option<usize>> = vec![None; B::BITS as usize];
    for dst in 0..n_sites {
        tgt[dst] = Some((dst + n_sites - 1) % n_sites);
    }
    gen_benes::<B>(&tgt)
}

type G<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn run_benes<B: BitInt>(g: &mut G<'_>, name: &str, n_sites: usize) {
    let net = cyclic_network::<B>(n_sites);
    let st = states::<B>(n_sites);
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

fn run_dit<B: BitInt>(g: &mut G<'_>, name: &str, n_sites: usize) {
    let manip = DynamicDitManip::new(2);
    let st = states::<B>(n_sites);
    g.bench_with_input(BenchmarkId::from_parameter(name), &st, |b, st| {
        b.iter(|| {
            let mut acc = 0usize;
            for &s in st {
                let mut t = s;
                for i in 0..n_sites {
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

fn run_map<B: BitInt>(g: &mut G<'_>, name: &str, n_sites: usize) {
    let st = states::<B>(n_sites);
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

// ===========================================================================
// Sweeps
// ===========================================================================

macro_rules! sweep {
    ($fname:ident, $group:literal, $runner:ident, $sites:expr) => {
        fn $fname(c: &mut Criterion) {
            macro_rules! width {
                ($w:literal, $limbs:literal) => {{
                    let n_sites: usize = $sites($w);
                    let mut g = c.benchmark_group(concat!($group, "/", $w));
                    g.throughput(Throughput::Elements(N_STATES as u64));
                    $runner::<Uint<$w, $limbs>>(&mut g, "Uint", n_sites);
                    $runner::<Dyn<$limbs>>(&mut g, "Dyn", n_sites);
                    g.finish();
                }};
            }
            width!(512, 8);
            width!(1024, 16);
            width!(2048, 32);
            width!(4096, 64);
        }
    };
}

// NARROW: 64 sites regardless of container -- the one-type scenario.
sweep!(benes_narrow, "benes_narrow", run_benes, |_w| 64usize);
sweep!(dit_narrow, "dit_narrow", run_dit, |_w| 64usize);
sweep!(map_narrow, "map_narrow", run_map, |_w| 64usize);

// DENSE: half the container -- the bound is saturated, nothing to skip.
sweep!(benes_dense, "benes_dense", run_benes, |w: usize| w / 2);
sweep!(dit_dense, "dit_dense", run_dit, |w: usize| w / 2);

criterion_group!(
    benches,
    benes_narrow,
    dit_narrow,
    map_narrow,
    benes_dense,
    dit_dense
);

fn main() {
    verify();
    eprintln!("prototype correctness: OK (Dyn<8..64> vs Uint<512..4096>)");
    eprintln!(
        "size_of: Uint<512>={} Dyn<8>={} | Uint<4096>={} Dyn<64>={}",
        std::mem::size_of::<Uint<512, 8>>(),
        std::mem::size_of::<Dyn<8>>(),
        std::mem::size_of::<Uint<4096, 64>>(),
        std::mem::size_of::<Dyn<64>>(),
    );
    benches();
    Criterion::default().configure_from_args().final_summary();
}

// ===========================================================================
// Correctness gate -- `Dyn<CAP>` must agree with `Uint<W, CAP>` exactly.
// ===========================================================================

fn verify() {
    verify_width::<512, 8>();
    verify_width::<1024, 16>();
    verify_width::<2048, 32>();
    verify_width::<4096, 64>();
    verify_benes::<512, 8>();
    verify_benes::<4096, 64>();
}

fn verify_width<const W: usize, const L: usize>()
where
    Uint<W, L>: BitInt,
    Dyn<L>: BitInt,
{
    let mut r = Rng(7);
    let shifts = [0usize, 1, 63, 64, 65, 127, 128, 255, 256, W / 2, W - 1];
    for _ in 0..40 {
        let (a, b) = (r.next(), r.next());
        let (ua, ub) = (Uint::<W, L>::from_u64(a), Uint::<W, L>::from_u64(b));
        let (da, db) = (Dyn::<L>::from_u64(a), Dyn::<L>::from_u64(b));

        assert_eq!((da & db).l, *(ua & ub).as_limbs(), "and");
        assert_eq!((da | db).l, *(ua | ub).as_limbs(), "or");
        assert_eq!((da ^ db).l, *(ua ^ ub).as_limbs(), "xor");
        assert_eq!((!da).l, *(!ua).as_limbs(), "not");
        assert_eq!(BitInt::count_ones(da), BitInt::count_ones(ua), "popcount");
        assert_eq!(da.cmp(&db), ua.cmp(&ub), "ord");

        for s in shifts {
            let (ul, dl) = (ua << s, da << s);
            assert_eq!(dl.l, *ul.as_limbs(), "shl {a:#x} << {s} @ W={W}");
            for i in (dl.n as usize)..L {
                assert_eq!(dl.l[i], 0, "shl bound violated @ W={W}");
            }
            // Push high, shift back down: exercises `shr` across limbs.
            let (uw, dw) = (ua << (W / 2), da << (W / 2));
            let (ur, dr) = (uw >> s, dw >> s);
            assert_eq!(dr.l, *ur.as_limbs(), "shr {a:#x} >> {s} @ W={W}");
            for i in (dr.n as usize)..L {
                assert_eq!(dr.l[i], 0, "shr bound violated @ W={W}");
            }
        }
    }
}

fn verify_benes<const W: usize, const L: usize>()
where
    Uint<W, L>: BitInt,
    Dyn<L>: BitInt,
{
    for &n_sites in &[64usize, W / 2] {
        let nu = cyclic_network::<Uint<W, L>>(n_sites);
        let nd = cyclic_network::<Dyn<L>>(n_sites);
        let su = states::<Uint<W, L>>(n_sites);
        let sd = states::<Dyn<L>>(n_sites);
        assert_eq!(su.len(), sd.len());
        for (u, d) in su.iter().zip(sd.iter()) {
            assert_eq!(*u.as_limbs(), d.l, "fixture mismatch @ W={W}");
            assert_eq!(
                *nu.apply(*u).as_limbs(),
                nd.apply(*d).l,
                "benes mismatch @ W={W} n_sites={n_sites}"
            );
        }
    }
}
