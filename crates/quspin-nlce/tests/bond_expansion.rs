//! Topological bond expansion: physics checks and consistency with the
//! rectangle expansion.

use quspin_nlce::*;

fn solver(temps: &[f64]) -> ExactDiagSolver {
    ExactDiagSolver::new(temps.to_vec()).unwrap()
}

/// On the chain both expansions use the same clusters (open paths), so
/// bond order `b` must equal rectangle order `b + 2` (a path of `b + 1`
/// sites) to rounding.
#[test]
fn chain_matches_rectangle_expansion() {
    let temps = [0.3, 1.0, 3.0];
    let s = solver(&temps);
    let model = Xxz::heisenberg(1.0);
    let bond = run_nlce(&BondGenerator::new(ChainLattice, 8), &s, &model).unwrap();
    let rect = run_nlce(&RectangleGenerator::new(ChainLattice, 10), &s, &model).unwrap();
    // Bond orders 0..=8, rectangle orders 2..=10.
    assert_eq!(bond.partial_sums.len(), rect.partial_sums.len());
    for ((ob, b), (or, r)) in bond.partial_sums.iter().zip(&rect.partial_sums) {
        assert_eq!(ob + 2, *or);
        assert!(
            b.max_abs_diff(r) < 1e-12,
            "order {ob}: {:e}",
            b.max_abs_diff(r)
        );
    }
}

/// A connected cluster with `b` bonds contributes to `ln Z` only from order
/// `β^b` on (every bond must appear in the cumulant), so the order-`n`
/// contribution must vanish at least as fast as `β^n`.
#[test]
fn order_contributions_vanish_as_beta_to_the_order() {
    // β = 0.1 and 0.05: small enough for the leading power to dominate,
    // large enough that every contribution is far above rounding.
    let temps = [10.0, 20.0];
    let result = run_nlce(
        &BondGenerator::new(SquareLattice, 6),
        &solver(&temps),
        &Xxz::heisenberg(1.0),
    )
    .unwrap();
    for (n, contrib) in &result.order_contributions {
        if *n < 2 {
            continue;
        }
        let (c10, c20) = (contrib.ln_z[0].abs(), contrib.ln_z[1].abs());
        assert!(
            c20 > 1e-12,
            "order {n} contribution {c20:e} is at the rounding floor"
        );
        // c(β) / c(β/2) >= 2^n, with a margin for subleading corrections.
        let ratio = c10 / c20;
        assert!(
            ratio > 0.9 * 2f64.powi(*n as i32),
            "order {n}: ratio {ratio} for |ΔlnZ| = {c10:e}, {c20:e}"
        );
    }
}

/// Both expansions converge to the same thermodynamic limit at high T.
#[test]
fn agrees_with_rectangle_expansion_at_high_temperature() {
    let temps = [2.0, 5.0, 20.0];
    let tol = [2e-4, 1e-6, 1e-9];
    let s = solver(&temps);
    let model = Xxz::heisenberg(1.0);
    let bond = run_nlce(&BondGenerator::new(SquareLattice, 8), &s, &model).unwrap();
    let rect = run_nlce(&RectangleGenerator::new(SquareLattice, 7), &s, &model).unwrap();
    let (b, r) = (
        bond.last_partial_sum().unwrap(),
        rect.last_partial_sum().unwrap(),
    );
    for (i, &t) in temps.iter().enumerate() {
        for (name, x, y) in [
            ("energy", &b.energy, &r.energy),
            ("entropy", &b.entropy, &r.entropy),
            ("specific heat", &b.specific_heat, &r.specific_heat),
        ] {
            let d = (x[i] - y[i]).abs();
            assert!(d < tol[i], "T={t}: {name} differs by {d:e}");
        }
    }
}

/// Classical Ising energy converges to Onsager above T_c.
#[test]
fn ising_converges_to_onsager() {
    fn onsager(t: f64) -> f64 {
        let j = 0.25;
        let k2 = 2.0 * j / t;
        let kappa = 2.0 * k2.sinh() / k2.cosh().powi(2);
        let (mut a, mut g) = (1.0, (1.0 - kappa * kappa).sqrt());
        for _ in 0..40 {
            (a, g) = (0.5 * (a + g), (a * g).sqrt());
        }
        let ell = std::f64::consts::PI / (2.0 * a);
        -j / k2.tanh() * (1.0 + 2.0 / std::f64::consts::PI * (2.0 * k2.tanh().powi(2) - 1.0) * ell)
    }
    let temps = [1.0, 2.0, 5.0];
    let tol = [2e-3, 1e-5, 1e-9];
    let result = run_nlce(
        &BondGenerator::new(SquareLattice, 8),
        &solver(&temps),
        &Xxz::ising(-1.0),
    )
    .unwrap();
    let last = result.last_partial_sum().unwrap();
    for (i, &t) in temps.iter().enumerate() {
        let err = (last.energy[i] - onsager(t)).abs();
        assert!(err < tol[i], "T={t}: |E − E_Onsager| = {err:e}");
    }
}
