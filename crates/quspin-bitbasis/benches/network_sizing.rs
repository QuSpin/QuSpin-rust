//! End-to-end effect of sizing the Benes network to the permutation.
//!
//! `benes_stages.rs` measured a reconstructed butterfly chain to isolate the
//! stage count. This measures the real library path instead: `gen_benes`
//! over a full-width `B::BITS` slice (what the code did before) against the
//! same permutation routed over `n_sites.next_power_of_two()` slots (what it
//! does now), applied through `BenesNetwork::apply`.
//!
//! Both networks compute the same permutation on the low bits -- the gate
//! below checks that on every configuration before any timing is reported --
//! so the difference is purely the stage count.

#![allow(clippy::needless_range_loop)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use quspin_bitbasis::benes::{BenesNetwork, gen_benes, gen_benes_for};
use quspin_types::BitInt;
use ruint::Uint;

mod common;
use common::states;

const N_STATES: usize = 1024;

/// The cyclic site shift, as a site permutation.
fn cyclic(n_sites: usize) -> Vec<usize> {
    (0..n_sites).map(|d| (d + n_sites - 1) % n_sites).collect()
}

/// The old behaviour: pad the permutation out to `B::BITS` and route over
/// the whole container.
fn full_width_network<B: BitInt>(n_sites: usize) -> BenesNetwork<B> {
    let mut c_tgt: Vec<Option<usize>> = (0..B::BITS as usize).map(Some).collect();
    for (dst, src) in cyclic(n_sites).into_iter().enumerate() {
        c_tgt[dst] = Some(src);
    }
    gen_benes::<B>(&c_tgt)
}

type G<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn run<B: BitInt>(g: &mut G<'_>, width: &str, n_sites: usize) {
    let st = states::<B>(n_sites, N_STATES);
    let full = full_width_network::<B>(n_sites);
    let short = gen_benes_for::<B>(&cyclic(n_sites));

    for (label, net) in [("full", &full), ("sized", &short)] {
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{width}/{label}")),
            &st,
            |b, st| {
                b.iter(|| {
                    let mut acc = B::from_u64(0);
                    for &s in st {
                        acc = acc ^ net.apply(black_box(s));
                    }
                    black_box(acc)
                })
            },
        );
    }
}

/// 64-site lattice symmetry, held in progressively wider containers. The
/// gap between `full` and `sized` is what the change buys.
fn bench_sizing(c: &mut Criterion) {
    let mut g = c.benchmark_group("network_sizing/64_sites");
    g.throughput(Throughput::Elements(N_STATES as u64));
    run::<u64>(&mut g, "u64", 64);
    run::<Uint<256, 4>>(&mut g, "U256", 64);
    run::<Uint<512, 8>>(&mut g, "U512", 64);
    run::<Uint<1024, 16>>(&mut g, "U1024", 64);
    run::<Uint<4096, 64>>(&mut g, "U4096", 64);
    g.finish();
}

/// A spin-1/2 chain at the widths `large-int` gates: sites scale with the
/// container, so the network shortens by only one or two stages. This is
/// the case that shows the change is not free money everywhere.
fn bench_dense(c: &mut Criterion) {
    let mut g = c.benchmark_group("network_sizing/dense");
    g.throughput(Throughput::Elements(N_STATES as u64));
    run::<Uint<512, 8>>(&mut g, "U512/300sites", 300);
    run::<Uint<1024, 16>>(&mut g, "U1024/600sites", 600);
    run::<Uint<4096, 64>>(&mut g, "U4096/2500sites", 2500);
    g.finish();
}

criterion_group!(benches, bench_sizing, bench_dense);

fn main() {
    verify();
    eprintln!("sized network == full-width network on all configurations: OK");
    benches();
    Criterion::default().configure_from_args().final_summary();
}

/// The two networks must agree on every state, or the speedup is not a
/// speedup. Checked across container widths and site counts, including the
/// non-power-of-two counts that exercise the identity padding.
fn verify() {
    check::<u64>(&[4, 8, 16, 20, 33, 64]);
    check::<Uint<256, 4>>(&[4, 16, 20, 64, 100, 256]);
    check::<Uint<512, 8>>(&[16, 64, 300, 512]);
    check::<Uint<4096, 64>>(&[16, 64, 2500]);
}

fn check<B: BitInt>(site_counts: &[usize]) {
    for &n in site_counts {
        if n > B::BITS as usize {
            continue;
        }
        let full = full_width_network::<B>(n);
        let short = gen_benes_for::<B>(&cyclic(n));
        for s in states::<B>(n, 256) {
            assert_eq!(
                short.apply(s),
                full.apply(s),
                "sized/full mismatch: BITS={} n_sites={n}",
                B::BITS
            );
        }
    }
}
