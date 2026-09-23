//! Exact inclusion–exclusion identity on a finite open cluster:
//! `P(R) = Σ_{s ⊆ R} M(s, R) W(s)` (the `s = R` term has `M = 1`), with
//! `M(s, R)` counted here by brute force independently of the generator.

use quspin_nlce::*;
use std::collections::{BTreeMap, HashSet};

/// Number of distinct site sets of shape `ms × ns` (either orientation)
/// inside a `w × h` box, by enumeration.
fn brute_force_embeddings(ms: usize, ns: usize, w: usize, h: usize) -> u64 {
    let mut sets: HashSet<Vec<(usize, usize)>> = HashSet::new();
    for (sw, sh) in [(ns, ms), (ms, ns)] {
        if sw > w || sh > h {
            continue;
        }
        for x0 in 0..=w - sw {
            for y0 in 0..=h - sh {
                let mut s: Vec<(usize, usize)> = (0..sw)
                    .flat_map(|x| (0..sh).map(move |y| (x0 + x, y0 + y)))
                    .collect();
                s.sort();
                sets.insert(s);
            }
        }
    }
    sets.len() as u64
}

fn identity_check(w: usize, h: usize) {
    let temps: Vec<f64> = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0].to_vec();
    let solver = ExactDiagSolver::new(temps).unwrap().with_magnetization();
    let model = Xxz::heisenberg(1.0);

    // Every rectangle that fits in the w × h box (closed under sub-rectangles).
    let clusters: Vec<ClusterType> = RectangleGenerator::new(SquareLattice, w + h)
        .clusters()
        .unwrap()
        .into_iter()
        .filter(|c| match c.key.topology {
            Topology::Rectangle { m, n } => {
                let (m, n) = (m as usize, n as usize);
                (n <= w && m <= h) || (n <= h && m <= w)
            }
            _ => false,
        })
        .collect();
    let target = ClusterKey::rectangle(w as u32, h as u32);

    let props: BTreeMap<ClusterKey, Thermo> = clusters
        .iter()
        .map(|c| (c.key.clone(), solver.solve(&c.graph, &model).unwrap()))
        .collect();
    let result = combine(&clusters, &props).unwrap();

    // Generator multiplicities agree with brute-force counts.
    let target_cluster = clusters.iter().find(|c| c.key == target).unwrap();
    for c in &clusters {
        if c.key == target {
            continue;
        }
        let Topology::Rectangle { m, n } = c.key.topology else {
            unreachable!()
        };
        let want = brute_force_embeddings(m as usize, n as usize, w.max(h), w.min(h));
        let got = target_cluster
            .subclusters
            .iter()
            .find(|(k, _)| *k == c.key)
            .map_or(0, |(_, v)| *v);
        assert_eq!(got, want, "M({}, {target})", c.key);
    }

    // Σ_s M(s, R) W(s) reproduces the directly computed P(R).
    let mut recon = props[&target].zeros_like();
    for c in &clusters {
        let Topology::Rectangle { m, n } = c.key.topology else {
            unreachable!()
        };
        let mult = if c.key == target {
            1
        } else {
            brute_force_embeddings(m as usize, n as usize, w.max(h), w.min(h))
        };
        recon.axpy(mult as f64, &result.cache.weights[&c.key]);
    }
    let direct = &props[&target];
    let scale = direct.max_abs_diff(&direct.zeros_like()).max(1.0);
    let err = recon.max_abs_diff(direct);
    assert!(
        err <= 1e-12 * scale,
        "identity violated on {target}: max error {err:e} (scale {scale:e})"
    );
}

#[test]
fn identity_3x4() {
    identity_check(4, 3);
}

/// The full 4 × 4 check (largest symmetry block ≈ 2900 states). Run with
/// `cargo test -p quspin-nlce --release -- --ignored`.
#[test]
#[ignore = "slow: diagonalises the 16-site cluster"]
fn identity_4x4() {
    identity_check(4, 4);
}
