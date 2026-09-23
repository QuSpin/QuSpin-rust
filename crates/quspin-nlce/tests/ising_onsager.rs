//! 2D classical Ising model: NLCE energy per site against Onsager above T_c.
//!
//! `H = J Σ S^z_i S^z_j` with `J = −1` is the σ-Ising model with coupling
//! `J_σ = 1/4`, so `T_c = J_σ · 2 / ln(1 + √2) ≈ 0.567`.

use quspin_nlce::*;
use std::f64::consts::PI;

/// Arithmetic–geometric mean (quadratic convergence: 40 steps is far past
/// machine precision for any `0 < b <= a`).
fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..40 {
        (a, b) = (0.5 * (a + b), (a * b).sqrt());
    }
    a
}

/// Onsager energy per site `u = −J_σ coth(2K) [1 + (2/π)(2 tanh²(2K) − 1) K(κ)]`
/// with `K = J_σ / T`, `κ = 2 sinh(2K) / cosh²(2K)`, and the complete
/// elliptic integral `K(κ) = π / (2 AGM(1, √(1 − κ²)))`.
fn onsager_energy(t: f64) -> f64 {
    let j = 0.25;
    let k2 = 2.0 * j / t;
    let kappa = 2.0 * k2.sinh() / k2.cosh().powi(2);
    let ell = PI / (2.0 * agm(1.0, (1.0 - kappa * kappa).sqrt()));
    -j / k2.tanh() * (1.0 + 2.0 / PI * (2.0 * k2.tanh().powi(2) - 1.0) * ell)
}

#[test]
fn onsager_reference_limits() {
    // High T: u ≈ −2 J_σ tanh(K) → −2 J_σ² / T.
    assert!((onsager_energy(1e3) + 2.0 * 0.0625 / 1e3).abs() < 1e-8);
}

#[test]
fn energy_converges_to_onsager_above_tc() {
    let tc = 0.5 / (1.0 + 2f64.sqrt()).ln();
    let temps = vec![0.7, 1.0, 1.5, 2.0, 5.0];
    let tol = [5e-3, 2e-4, 5e-6, 2e-7, 1e-11];
    assert!(temps[0] > tc);
    let solver = ExactDiagSolver::new(temps.clone()).unwrap();
    let result = run_nlce(
        &RectangleGenerator::new(SquareLattice, 7),
        &solver,
        &Xxz::ising(-1.0),
    )
    .unwrap();

    let last = result.last_partial_sum().unwrap();
    for (i, &t) in temps.iter().enumerate() {
        let err = (last.energy[i] - onsager_energy(t)).abs();
        assert!(err < tol[i], "T={t}: |E_NLCE − E_Onsager| = {err:e}");
    }
    // Monotone convergence from order 4 on, at every temperature above T_c.
    for (i, &t) in temps.iter().enumerate() {
        let errs: Vec<f64> = result
            .partial_sums
            .iter()
            .filter(|(o, _)| *o >= 4)
            .map(|(_, p)| (p.energy[i] - onsager_energy(t)).abs())
            .collect();
        for w in errs.windows(2) {
            assert!(w[1] < w[0] || w[1] < 1e-12, "T={t}: {errs:?}");
        }
    }
}

/// Wynn resummation must beat the bare sums near T_c, measured against the
/// exact Onsager energy.
#[test]
fn wynn_improves_on_bare_sums_near_tc() {
    let temps = vec![0.7, 0.8, 1.0];
    let solver = ExactDiagSolver::new(temps.clone()).unwrap();
    let result = run_nlce(
        &RectangleGenerator::new(SquareLattice, 7),
        &solver,
        &Xxz::ising(-1.0),
    )
    .unwrap();
    let sums = result.sums();
    let bare = Bare.resum(&sums).unwrap();
    let wynn = Wynn { cycles: 2 }.resum(&sums).unwrap();
    for (i, &t) in temps.iter().enumerate() {
        let e_bare = (bare.energy[i] - onsager_energy(t)).abs();
        let e_wynn = (wynn.energy[i] - onsager_energy(t)).abs();
        eprintln!("T={t}: bare {e_bare:e}, Wynn(2) {e_wynn:e}");
        assert!(
            e_wynn < 0.2 * e_bare,
            "T={t}: bare {e_bare:e}, Wynn {e_wynn:e}"
        );
    }
}
