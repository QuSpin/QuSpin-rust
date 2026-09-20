//! Build-time cost of `build_from_basis`.
//!
//! Type-erasing `B`/`S`/`C` out of the build path (see
//! `qmatrix::rowsource`) cut the crate's rlib by 17.7%, but it trades three
//! things that only a runtime measurement can settle:
//!
//!   - **against it:** `RawEntry` is 32 bytes where `Entry<f32, i64, u8>`
//!     is 16, so narrow element types pay more intermediate write traffic;
//!   - **against it:** one virtual call per row through `&dyn RowSource`;
//!   - **for it:** coalescing became sort-then-merge, `O(k log k)`, where
//!     the old linear `find` per emitted term was `O(k^2)` in the row's
//!     non-zero count.
//!
//! The third term grows with row density and the first two do not, so the
//! groups below deliberately span both regimes:
//!
//!   `chain`      nearest-neighbour XX, ~n_sites non-zeros per row --
//!                sparse rows, where the `O(k^2)` term is small and the
//!                intermediate size dominates. The case that can regress.
//!   `all_to_all` every (i, j) pair, ~n_sites^2/2 per row -- dense rows,
//!                where the coalescing complexity should dominate.
//!
//! Each runs at `f32` (worst case for the size argument: smallest `Entry`)
//! and `Complex<f64>` (where `Entry` and `RawEntry` are comparable), and at
//! dimensions either side of `PARALLEL_DIM_THRESHOLD = 256` so both the
//! serial and rayon paths are covered.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex;
use quspin_basis::BasisSpace;
use quspin_basis::space::FullSpace;
use quspin_matrix::qmatrix::QMatrix;
use quspin_matrix::qmatrix::build::build_from_basis;
use quspin_operator::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
use smallvec::smallvec;

/// Nearest-neighbour XX ring: `n_sites` terms, so ~`n_sites` non-zeros per
/// row and essentially no duplicate `(col, cindex)` pairs to coalesce.
fn xx_chain(n_sites: usize) -> HardcoreOperator<u8> {
    let mut terms = Vec::new();
    for i in 0..n_sites {
        let j = (i + 1) % n_sites;
        let ops = smallvec![(HardcoreOp::X, i as u32), (HardcoreOp::X, j as u32)];
        terms.push(OpEntry::new(0u8, Complex::new(1.0, 0.0), ops));
    }
    HardcoreOperator::new(terms)
}

/// All-to-all XX: `n_sites*(n_sites-1)/2` terms. Rows are dense and many
/// terms collide on the same column, which is what exercises coalescing.
fn xx_all_to_all(n_sites: usize) -> HardcoreOperator<u8> {
    let mut terms = Vec::new();
    for i in 0..n_sites {
        for j in (i + 1)..n_sites {
            let ops = smallvec![(HardcoreOp::X, i as u32), (HardcoreOp::X, j as u32)];
            terms.push(OpEntry::new(0u8, Complex::new(1.0, 0.0), ops));
        }
    }
    HardcoreOperator::new(terms)
}

type G<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn run<M>(g: &mut G<'_>, label: &str, ham: &HardcoreOperator<u8>, n_sites: usize)
where
    M: quspin_types::Primitive,
{
    let basis = FullSpace::<u64>::new(2, n_sites, false);
    g.throughput(Throughput::Elements(basis.size() as u64));
    g.bench_function(BenchmarkId::from_parameter(label), |b| {
        b.iter(|| {
            let m: QMatrix<M, i64, u8> = build_from_basis(black_box(ham), black_box(&basis));
            black_box(m.nnz())
        })
    });
}

fn bench_chain(c: &mut Criterion) {
    // n_sites = 8 -> dim 256 (first parallel dim); 14 -> dim 16384.
    for &n in &[8usize, 12, 14] {
        let mut g = c.benchmark_group(format!("chain/{n}sites"));
        let ham = xx_chain(n);
        run::<f32>(&mut g, "f32", &ham, n);
        run::<Complex<f64>>(&mut g, "c64", &ham, n);
        g.finish();
    }
}

fn bench_all_to_all(c: &mut Criterion) {
    for &n in &[8usize, 10, 12] {
        let mut g = c.benchmark_group(format!("all_to_all/{n}sites"));
        let ham = xx_all_to_all(n);
        run::<f32>(&mut g, "f32", &ham, n);
        run::<Complex<f64>>(&mut g, "c64", &ham, n);
        g.finish();
    }
}

/// Below `PARALLEL_DIM_THRESHOLD` the rayon path is skipped entirely, so
/// this isolates the virtual call and the intermediate size from any
/// parallel-machinery effect.
fn bench_serial(c: &mut Criterion) {
    let mut g = c.benchmark_group("serial_dim128");
    let ham = xx_chain(7); // dim = 128 < 256
    run::<f32>(&mut g, "chain/f32", &ham, 7);
    run::<Complex<f64>>(&mut g, "chain/c64", &ham, 7);
    let dense = xx_all_to_all(7);
    run::<f32>(&mut g, "all_to_all/f32", &dense, 7);
    run::<Complex<f64>>(&mut g, "all_to_all/c64", &dense, 7);
    g.finish();
}

criterion_group!(benches, bench_chain, bench_all_to_all, bench_serial);
criterion_main!(benches);
