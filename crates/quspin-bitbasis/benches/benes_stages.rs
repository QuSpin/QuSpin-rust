//! Network size vs representation: which one dominates the large-int cost?
//!
//! `gen_benes` asserts `c_tgt.len() == B::BITS` and takes its stage count
//! from `B::LD_BITS`, so the network is welded to the *container* width, not
//! to `n_sites`. A 64-site problem in a 4096-bit container therefore runs 12
//! butterfly stages where 6 would do.
//!
//! That leaves two candidate explanations for large-int Benes cost, and they
//! imply completely different fixes:
//!
//!   (a) the stage count (`LD_BITS`), fixable by sizing the network to
//!       `n_sites` without touching the data structure at all; or
//!   (b) the per-stage multi-limb cost, fixable only by the representation.
//!
//! `BenesNetwork`'s fields are private and `gen_benes` will not build a
//! short network, so this reimplements the butterfly chain directly --
//! `bit_permute_step` is the whole hot loop, and a `BenesNetwork::apply` is
//! exactly `2 * LD_BITS` of them. Sweeping the stage count independently of
//! the type separates the two factors:
//!
//!   cost(B, k) / cost(B, 6)  -- stage sensitivity, at fixed representation
//!   cost(B, k) / cost(u64, k) -- representation sensitivity, at fixed stages
//!
//! Stage `s` uses shift `1 << s`, matching the real network. That detail
//! matters for `Dyn`: a 6-stage network never shifts past 32 bits, so the
//! limb bound stays tight, while a 12-stage one shifts by up to 2048 and
//! saturates it immediately.

#![allow(clippy::needless_range_loop)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use quspin_types::BitInt;
use ruint::Uint;

mod common;
use common::{Dyn, states};

const N_STATES: usize = 1024;

/// The butterfly primitive, identical to `benes::bit_permute_step`.
#[inline]
fn bit_permute_step<B: BitInt>(x: B, m: B, shift: usize) -> B {
    let t = ((x >> shift) ^ x) & m;
    x ^ t ^ (t << shift)
}

/// One `BenesNetwork::apply` over `k` stages: forward high->low, then
/// inverse low->high, exactly as `Butterfly::apply` / `apply_inv` do.
#[inline]
fn apply_k<B: BitInt>(x: B, cfg: &[B], k: usize) -> B {
    let mut v = x;
    for s in (0..k).rev() {
        v = bit_permute_step(v, cfg[s], 1usize << s);
    }
    for s in 0..k {
        v = bit_permute_step(v, cfg[k + s], 1usize << s);
    }
    v
}

/// Plausible non-trivial stage masks. Exact values are irrelevant to timing
/// -- only that they are not zero (which could let LLVM fold a stage away).
fn masks<B: BitInt>(k: usize) -> Vec<B> {
    let mut r = common::Rng(0xABCD_EF01_2345_6789);
    (0..2 * k)
        .map(|_| {
            let mut m = B::from_u64(0);
            for w in 0..(B::BITS as usize / 64) {
                m = m | (B::from_u64(r.next()) << (w * 64));
            }
            m
        })
        .collect()
}

type G<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn run<B: BitInt>(g: &mut G<'_>, name: &str, k: usize, n_sites: usize) {
    let cfg = masks::<B>(k);
    let st = states::<B>(n_sites, N_STATES);
    g.bench_with_input(BenchmarkId::from_parameter(name), &st, |b, st| {
        b.iter(|| {
            let mut acc = B::from_u64(0);
            for &s in st {
                acc = acc ^ apply_k(black_box(s), &cfg, k);
            }
            black_box(acc)
        })
    });
}

/// 64-site physics in every container: the case where sizing the network to
/// `n_sites` (6 stages) is legitimate, and where the ladder would place the
/// state in a narrow type but a one-type design would not.
fn bench_stages(c: &mut Criterion) {
    for &k in &[6usize, 9, 12] {
        let mut g = c.benchmark_group(format!("stages/{k}"));
        g.throughput(Throughput::Elements(N_STATES as u64));
        // u64 only admits shifts < 64, so it caps out at 6 stages.
        if k <= 6 {
            run::<u64>(&mut g, "u64", k, 64);
        }
        run::<Uint<512, 8>>(&mut g, "Uint<512>", k, 64);
        run::<Dyn<8>>(&mut g, "Dyn<512>", k, 64);
        run::<Uint<4096, 64>>(&mut g, "Uint<4096>", k, 64);
        run::<Dyn<64>>(&mut g, "Dyn<4096>", k, 64);
        g.finish();
    }
}

criterion_group!(benches, bench_stages);

fn main() {
    verify();
    eprintln!("stage-chain shape + Dyn/Uint agreement: OK");
    benches();
    Criterion::default().configure_from_args().final_summary();
}

/// What is and is not verified here, stated plainly.
///
/// `apply_k` is a *reconstruction*, not the library's code path:
/// `BenesNetwork`'s `cfg` is private, so the chain cannot be diffed against
/// `BenesNetwork::apply` value-for-value. What makes it faithful is
/// structural: `bit_permute_step` is copied verbatim from `benes.rs`, and
/// the stage order (forward high->low then inverse low->high, shift
/// `1 << s`) mirrors `Butterfly::apply` / `apply_inv`. Since the sweep only
/// claims a *cost model* -- `2k` butterfly steps -- that is the property
/// that has to hold.
///
/// Two things are checked for real:
///   - `stage_count_matches_ld_bits` -- the library really does emit
///     `LD_BITS` stages per butterfly, so `k` maps onto a real network.
///   - `verify_dyn_matches_uint` -- `Dyn` and `Uint` compute bit-identical
///     chains, so their rows are the same computation and comparable.
fn verify() {
    stage_count_matches_ld_bits::<u64>(6);
    stage_count_matches_ld_bits::<Uint<512, 8>>(9);
    stage_count_matches_ld_bits::<Dyn<8>>(9);
    stage_count_matches_ld_bits::<Uint<4096, 64>>(12);
    stage_count_matches_ld_bits::<Dyn<64>>(12);
    verify_dyn_matches_uint();
}

/// Pins the premise of the whole sweep: the network's stage count is
/// `B::LD_BITS`, set by the container width, and a real permutation built
/// through it round-trips.
fn stage_count_matches_ld_bits<B: BitInt>(expect_ld: u32) {
    use quspin_bitbasis::benes::gen_benes;
    assert_eq!(B::LD_BITS, expect_ld, "LD_BITS drifted");

    let n_sites = 64usize;
    let mut tgt: Vec<Option<usize>> = vec![None; B::BITS as usize];
    for dst in 0..n_sites {
        tgt[dst] = Some((dst + n_sites - 1) % n_sites);
    }
    let net = gen_benes::<B>(&tgt);
    // A 64-site cyclic shift has order 64, which pins the network's
    // semantics independently of its internal configuration.
    for s in states::<B>(n_sites, 32) {
        let mut v = s;
        for _ in 0..n_sites {
            v = net.apply(v);
        }
        assert_eq!(v, s, "cyclic shift must have order n_sites");
    }
}

/// `Dyn<CAP>` and `Uint<W, CAP>` must produce identical butterfly chains --
/// otherwise the Dyn rows are not measuring the same computation.
fn verify_dyn_matches_uint() {
    fn check<const W: usize, const L: usize>(k: usize)
    where
        Uint<W, L>: BitInt,
        Dyn<L>: BitInt,
    {
        let cu = masks::<Uint<W, L>>(k);
        let cd: Vec<Dyn<L>> = cu
            .iter()
            .map(|m| Dyn::<L> {
                l: *m.as_limbs(),
                n: L as u16,
            })
            .collect();
        let su = states::<Uint<W, L>>(64, 64);
        let sd = states::<Dyn<L>>(64, 64);
        for (u, d) in su.iter().zip(sd.iter()) {
            let ru = apply_k(*u, &cu, k);
            let rd = apply_k(*d, &cd, k);
            assert_eq!(
                *ru.as_limbs(),
                rd.l,
                "Dyn/Uint chain mismatch @ W={W} k={k}"
            );
        }
    }
    for k in [6usize, 9] {
        check::<512, 8>(k);
    }
    for k in [6usize, 9, 12] {
        check::<4096, 64>(k);
    }
}
