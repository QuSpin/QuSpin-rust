//! Topological expansions over different node kinds, lattices and labels.

use quspin_nlce::canon::{SmallGraph, canonical_form};
use quspin_nlce::*;

fn solver(temps: &[f64]) -> ExactDiagSolver {
    ExactDiagSolver::new(temps.to_vec()).unwrap()
}

/// Lattice constant of the (labelled) topology with these vertex labels and
/// edges, 0 if absent.
fn l(cs: &[ClusterType], vl: &[u32], e: &[(usize, usize, u32)]) -> f64 {
    let code = canonical_form(&SmallGraph::from_labeled_edges(vl, e), false).code;
    cs.iter()
        .find(|c| c.key.topology == Topology::Canonical(code.clone()))
        .map_or(0.0, |c| c.lattice_constant)
}

#[test]
fn site_expansion_counts_polyominoes() {
    // Fixed polyominoes, OEIS A001168.
    let census = SiteGenerator::new(SquareLattice, 7).census().unwrap();
    assert_eq!(census.fixed_clusters, vec![0, 1, 2, 6, 19, 63, 216, 760]);
}

#[test]
fn site_expansion_on_chain_matches_rectangles() {
    let temps = [0.3, 1.0, 3.0];
    let s = solver(&temps);
    let model = Xxz::heisenberg(1.0);
    let site = run_nlce(&SiteGenerator::new(ChainLattice, 9), &s, &model).unwrap();
    let rect = run_nlce(&RectangleGenerator::new(ChainLattice, 10), &s, &model).unwrap();
    // Site orders 1..=9 (paths of n sites), rectangle orders 2..=10.
    assert_eq!(site.partial_sums.len(), rect.partial_sums.len());
    for ((os, a), (or, b)) in site.partial_sums.iter().zip(&rect.partial_sums) {
        assert_eq!(os + 1, *or);
        assert!(
            a.max_abs_diff(b) < 1e-12,
            "order {os}: {:e}",
            a.max_abs_diff(b)
        );
    }
}

#[test]
fn site_expansion_agrees_with_rectangles_at_high_temperature() {
    let temps = [2.0, 5.0, 20.0];
    let tol = [3e-4, 1e-6, 1e-9];
    let s = solver(&temps);
    let model = Xxz::heisenberg(1.0);
    let site = run_nlce(&SiteGenerator::new(SquareLattice, 9), &s, &model).unwrap();
    let rect = run_nlce(&RectangleGenerator::new(SquareLattice, 7), &s, &model).unwrap();
    let (a, b) = (
        site.last_partial_sum().unwrap(),
        rect.last_partial_sum().unwrap(),
    );
    for (i, &t) in temps.iter().enumerate() {
        let d = (a.energy[i] - b.energy[i])
            .abs()
            .max((a.specific_heat[i] - b.specific_heat[i]).abs());
        assert!(d < tol[i], "T={t}: site vs rectangle differ by {d:e}");
    }
}

#[test]
fn lattice_constants_on_other_lattices() {
    // Triangular: 3 bonds, 15 two-bond paths and 2 triangles per site.
    let tb = BondGenerator::new(TriangularLattice, 3).clusters().unwrap();
    assert_eq!(l(&tb, &[0, 0], &[(0, 1, 0)]), 3.0);
    assert_eq!(l(&tb, &[0, 0, 0], &[(0, 1, 0), (1, 2, 0)]), 15.0);
    assert_eq!(l(&tb, &[0, 0, 0], &[(0, 1, 0), (1, 2, 0), (2, 0, 0)]), 2.0);
    // Site expansion: three mutually adjacent sites induce a triangle; open
    // 3-site lines are straight (3 per site) or bent by 120° (6 per site).
    let ts = SiteGenerator::new(TriangularLattice, 3).clusters().unwrap();
    assert_eq!(l(&ts, &[0, 0, 0], &[(0, 1, 0), (1, 2, 0), (2, 0, 0)]), 2.0);
    assert_eq!(l(&ts, &[0, 0, 0], &[(0, 1, 0), (1, 2, 0)]), 9.0);
    // Site clusters of the triangular lattice are fixed polyhexes (A001207).
    let census = SiteGenerator::new(TriangularLattice, 5).census().unwrap();
    assert_eq!(census.fixed_clusters, vec![0, 1, 3, 11, 44, 186]);
    // Honeycomb (2-site cell): per site 3/2 bonds, 3 two-bond paths, one
    // 3-leaf star, and 1/2 hexagon.
    let h = BondGenerator::new(HoneycombLattice, 6).clusters().unwrap();
    assert_eq!(l(&h, &[0, 0], &[(0, 1, 0)]), 1.5);
    assert_eq!(l(&h, &[0, 0, 0], &[(0, 1, 0), (1, 2, 0)]), 3.0);
    assert_eq!(l(&h, &[0; 4], &[(0, 1, 0), (0, 2, 0), (0, 3, 0)]), 1.0);
    let hexagon: Vec<(usize, usize, u32)> = (0..6).map(|i| (i, (i + 1) % 6, 0)).collect();
    assert_eq!(l(&h, &[0; 6], &hexagon), 0.5);
}

#[test]
fn honeycomb_heisenberg_high_temperature_series() {
    // z = 3, bipartite: E/N = −(3/16)(z/2) β − (3/64)(z/2) β² + O(β³).
    let temps = [10.0, 20.0];
    let r = run_nlce(
        &BondGenerator::new(HoneycombLattice, 6),
        &solver(&temps),
        &Xxz::heisenberg(1.0),
    )
    .unwrap();
    let e = &r.last_partial_sum().unwrap().energy;
    for (i, &t) in temps.iter().enumerate() {
        let b = 1.0 / t;
        let series = -9.0 / 32.0 * b - 9.0 / 128.0 * b * b;
        assert!(
            (e[i] - series).abs() < 0.2 * b.powi(3),
            "T={t}: {} vs {series}",
            e[i]
        );
    }
}

#[test]
fn j1j2_with_zero_j2_reproduces_the_square_lattice() {
    let temps = [0.5, 1.0, 3.0];
    let s = solver(&temps);
    let j1j2 = run_nlce(
        &BondGenerator::new(SquareJ1J2Lattice, 5),
        &s,
        &LabeledXxz {
            couplings: vec![(1.0, 1.0), (0.0, 0.0)],
            fields: vec![],
        },
    )
    .unwrap();
    let square = run_nlce(
        &BondGenerator::new(SquareLattice, 5),
        &s,
        &Xxz::heisenberg(1.0),
    )
    .unwrap();
    // Clusters containing a J2 bond carry zero weight.
    for ((oa, a), (ob, b)) in j1j2.partial_sums.iter().zip(&square.partial_sums) {
        assert_eq!(oa, ob);
        assert!(
            a.max_abs_diff(b) < 1e-10,
            "order {oa}: {:e}",
            a.max_abs_diff(b)
        );
    }
}

#[test]
fn j1j2_bond_labels_are_kept_apart() {
    let cs = BondGenerator::new(SquareJ1J2Lattice, 2).clusters().unwrap();
    // One J1 bond type and one J2 bond type, two of each per site.
    assert_eq!(l(&cs, &[0, 0], &[(0, 1, 0)]), 2.0);
    assert_eq!(l(&cs, &[0, 0], &[(0, 1, 1)]), 2.0);
    // Leading high-T energy −(3/8) β (J1² + J2²) needs both bond types.
    let (j1, j2) = (1.0, 0.5);
    let temps = [20.0, 40.0];
    let r = run_nlce(
        &BondGenerator::new(SquareJ1J2Lattice, 4),
        &solver(&temps),
        &LabeledXxz {
            couplings: vec![(j1, j1), (j2, j2)],
            fields: vec![],
        },
    )
    .unwrap();
    let e = &r.last_partial_sum().unwrap().energy;
    for (i, &t) in temps.iter().enumerate() {
        let b = 1.0 / t;
        let lead = -3.0 / 8.0 * b * (j1 * j1 + j2 * j2);
        assert!(
            (e[i] - lead).abs() < 0.5 * b * b,
            "T={t}: {} vs {lead}",
            e[i]
        );
    }
}

/// Square lattice with checkerboard site labels; translations that preserve
/// the labels leave a two-site unit cell.
#[derive(Clone, Copy)]
struct Checkerboard;

impl Lattice for Checkerboard {
    type Site = [i32; 2];
    fn unit_cell_sites(&self) -> Vec<[i32; 2]> {
        vec![[0, 0], [1, 0]]
    }
    fn neighbors(&self, s: [i32; 2]) -> Vec<([i32; 2], u32)> {
        SquareLattice.neighbors(s)
    }
    fn site_label(&self, [x, y]: [i32; 2]) -> u32 {
        (x + y).rem_euclid(2) as u32
    }
    fn point_group(&self) -> Vec<fn([i32; 2]) -> [i32; 2]> {
        SquareLattice.point_group()
    }
    fn translate(&self, sites: &[[i32; 2]]) -> Vec<[i32; 2]> {
        sites.to_vec()
    }
}

#[test]
fn site_labels_reach_the_model() {
    let cs = BondGenerator::new(Checkerboard, 3).clusters().unwrap();
    // Two single-site terms (one per sublattice), half a site each.
    assert_eq!(l(&cs, &[0], &[]), 0.5);
    assert_eq!(l(&cs, &[1], &[]), 0.5);
    // Free spins in a staggered field: E/N = −(h/2) tanh(βh/2) exactly.
    let h = 0.8;
    let temps = [0.3, 1.0, 4.0];
    let r = run_nlce(
        &BondGenerator::new(Checkerboard, 3),
        &solver(&temps),
        &LabeledXxz {
            couplings: vec![(0.0, 0.0)],
            fields: vec![h, -h],
        },
    )
    .unwrap();
    for (i, &t) in temps.iter().enumerate() {
        let want = -0.5 * h * (0.5 * h / t).tanh();
        let got = r.last_partial_sum().unwrap().energy[i];
        assert!((got - want).abs() < 1e-12, "T={t}: {got} vs {want}");
    }
}
