//! Count the connected bond clusters of the square lattice and their
//! topological classes, order by order.
//!
//! ```sh
//! cargo run --release -p quspin-nlce --example bond_census [max_order] [out_file]
//! ```
//!
//! With `out_file`, the cluster DAG is also saved as a [`ClusterSet`] for
//! reuse with any model (see `heisenberg_partial_sums … load <file>`).

use quspin_nlce::*;
use std::time::Instant;

fn main() -> Result<(), NlceError> {
    let max_order: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("max_order must be an integer"))
        .unwrap_or(12);
    let generator = BondGenerator::new(SquareLattice, max_order);

    let t0 = Instant::now();
    let census = generator.census()?;
    let t_census = t0.elapsed();
    let mut topologies = vec![0usize; max_order + 1];
    for code in census.embeddings.keys() {
        let edges = quspin_nlce::canon::from_code(code).n_edges();
        topologies[edges] += 1;
    }
    println!(
        "{:>5} {:>14} {:>11}",
        "bonds", "embeddings/site", "topologies"
    );
    for (n, (fixed, topo)) in census
        .fixed_clusters
        .iter()
        .zip(&topologies)
        .enumerate()
        .skip(1)
    {
        println!("{n:>5} {fixed:>14} {topo:>11}");
    }
    println!(
        "total {:>14} {:>11}   (enumeration + canonicalisation: {:.2?})",
        census.fixed_clusters.iter().sum::<u64>(),
        census.embeddings.len(),
        t_census
    );

    let t1 = Instant::now();
    let clusters = generator.clusters_from_census(&census)?;
    let n_sub: usize = clusters.iter().map(|c| c.subclusters.len()).sum();
    println!(
        "cluster DAG: {} types, {n_sub} (sub-cluster, cluster) pairs, built in {:.2?}",
        clusters.len(),
        t1.elapsed()
    );
    if let Some(path) = std::env::args().nth(2) {
        let set = ClusterSet {
            description: format!("topological bond expansion, square lattice, {max_order} bonds"),
            clusters,
        };
        let t2 = Instant::now();
        set.save(&path)?;
        let bytes = std::fs::metadata(&path)?.len();
        println!(
            "saved to {path} ({:.1} MB) in {:.2?}",
            bytes as f64 / 1e6,
            t2.elapsed()
        );
        let t3 = Instant::now();
        let loaded = ClusterSet::load(&path)?;
        println!(
            "reloaded {} clusters in {:.2?}",
            loaded.clusters.len(),
            t3.elapsed()
        );
    }
    Ok(())
}
