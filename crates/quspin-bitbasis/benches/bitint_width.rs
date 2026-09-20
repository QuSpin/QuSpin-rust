//! What does carrying a wider `BitInt` cost, at fixed physics?
//!
//! Every group below runs the *same* work — same site count, same state
//! count, same permutation — while varying only the integer type the state
//! is stored in. That isolates the representation cost from the problem
//! size, which is the number that matters when deciding whether one
//! representation can serve every width.
//!
//! The four groups are the operations a basis build actually spends its
//! time in:
//!
//! - `benes`      — lattice-symmetry orbit application (`BenesNetwork::apply`)
//! - `dit_sweep`  — `get_dit`/`set_dit` across all sites (operator apply)
//! - `hashmap`    — the `index_map: HashMap<B, usize>` probe in `Subspace`
//! - `sort_search`— `states: Vec<B>` ordering + lookup

// Limb indexing is deliberate here: the loop index addresses two arrays at
// different offsets, which iterators cannot express as clearly.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use quspin_bitbasis::benes::{BenesNetwork, gen_benes};
use quspin_bitbasis::manip::DynamicDitManip;
use quspin_types::BitInt;
use ruint::Uint;

type U128 = Uint<128, 2>;
type U256 = Uint<256, 4>;

/// Sites used for every group. Deliberately small enough to fit a `u64`, so
/// the wider types are doing the *same* physics in a bigger container —
/// exactly the situation a single shared representation would create.
const N_SITES: usize = 32;
/// Number of distinct basis states per sample.
const N_STATES: usize = 4096;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// xorshift64* — deterministic, no dev-dependency on `rand`.
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

/// `N_STATES` distinct states occupying the low `N_SITES` bits.
fn states<B: BitInt>() -> Vec<B> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mask = if N_SITES >= 64 {
        u64::MAX
    } else {
        (1u64 << N_SITES) - 1
    };
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

/// A cyclic site shift on `N_SITES` bits — the canonical translation
/// symmetry, and the permutation `BenesLatticeElement` is built from.
fn cyclic_network<B: BitInt>() -> BenesNetwork<B> {
    let mut tgt: Vec<Option<usize>> = vec![None; B::BITS as usize];
    for dst in 0..N_SITES {
        tgt[dst] = Some((dst + N_SITES - 1) % N_SITES);
    }
    gen_benes::<B>(&tgt)
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

fn bench_benes(c: &mut Criterion) {
    let mut g = c.benchmark_group("benes");
    g.throughput(Throughput::Elements(N_STATES as u64));

    fn run<B: BitInt>(
        g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        name: &str,
    ) {
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
    run::<u128>(&mut g, "u128");
    run::<U128>(&mut g, "Uint<128>");
    run::<U256>(&mut g, "Uint<256>");
    g.finish();
}

fn bench_dit_sweep(c: &mut Criterion) {
    let mut g = c.benchmark_group("dit_sweep");
    g.throughput(Throughput::Elements((N_STATES * N_SITES) as u64));

    fn run<B: BitInt>(
        g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        name: &str,
    ) {
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
    run::<u128>(&mut g, "u128");
    run::<U128>(&mut g, "Uint<128>");
    run::<U256>(&mut g, "Uint<256>");
    g.finish();
}

fn bench_hashmap(c: &mut Criterion) {
    let mut g = c.benchmark_group("hashmap_build_probe");
    g.throughput(Throughput::Elements(N_STATES as u64));

    fn run<B: BitInt>(
        g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        name: &str,
    ) {
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
    run::<u128>(&mut g, "u128");
    run::<U128>(&mut g, "Uint<128>");
    run::<U256>(&mut g, "Uint<256>");
    g.finish();
}

fn bench_sort_search(c: &mut Criterion) {
    let mut g = c.benchmark_group("sort_search");
    g.throughput(Throughput::Elements(N_STATES as u64));

    fn run<B: BitInt>(
        g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        name: &str,
    ) {
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
    run::<u128>(&mut g, "u128");
    run::<U128>(&mut g, "Uint<128>");
    run::<U256>(&mut g, "Uint<256>");
    g.finish();
}

criterion_group!(
    benches,
    bench_benes,
    bench_dit_sweep,
    bench_hashmap,
    bench_sort_search
);
criterion_main!(benches);
