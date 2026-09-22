//! 1D XX chain (1 × n clusters) against the free-fermion thermodynamic limit.
//!
//! Jordan–Wigner maps `H = J Σ (S^x S^x + S^y S^y)` to free fermions with
//! `ε_k = J cos k` and `S^z_tot = N_f − N/2`, so per site
//! `ln Z = ∫ ln(1 + e^{−βε})`, `E = ∫ ε f(ε)`, `χ = β ∫ f(1 − f)`.

use quspin_nlce::*;
use std::f64::consts::PI;

/// `(E, S, χ)` per site; midpoint rule is spectrally accurate for the
/// smooth periodic integrands.
fn free_fermions(t: f64) -> (f64, f64, f64) {
    let n = 4096;
    let beta = 1.0 / t;
    let (mut ln_z, mut e, mut chi) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let eps = (2.0 * PI * (i as f64 + 0.5) / n as f64).cos();
        let f = 1.0 / ((beta * eps).exp() + 1.0);
        ln_z += (1.0 + (-beta * eps).exp()).ln();
        e += eps * f;
        chi += beta * f * (1.0 - f);
    }
    let n = n as f64;
    (e / n, ln_z / n + beta * e / n, chi / n)
}

#[test]
fn converges_to_free_fermions() {
    let temps = vec![0.3, 0.5, 1.0, 2.0, 5.0];
    let tol = [1e-7, 1e-10, 1e-12, 1e-12, 1e-12];
    let solver = ExactDiagSolver::new(temps.clone())
        .unwrap()
        .with_magnetization();
    // Order m + n = 13: chains up to 12 sites.
    let result = run_nlce(
        &RectangleGenerator::new(ChainLattice, 13),
        &solver,
        &Xxz::xx(1.0),
    )
    .unwrap();
    assert_eq!(result.partial_sums.len(), 12);

    let last = result.last_partial_sum().unwrap();
    for (i, &t) in temps.iter().enumerate() {
        let (e, s, chi) = free_fermions(t);
        let de = (last.energy[i] - e).abs();
        let ds = (last.entropy[i] - s).abs();
        let dchi = (last.susceptibility.as_ref().unwrap()[i] - chi).abs();
        assert!(de < tol[i], "T={t}: energy error {de:e}");
        assert!(ds < tol[i], "T={t}: entropy error {ds:e}");
        assert!(dchi < 10.0 * tol[i], "T={t}: susceptibility error {dchi:e}");
        assert!(last.magnetization.as_ref().unwrap()[i].abs() < 1e-14);
    }

    // The error shrinks with order at the lowest temperature.
    let errs: Vec<f64> = result
        .partial_sums
        .iter()
        .map(|(_, p)| (p.energy[0] - free_fermions(temps[0]).0).abs())
        .collect();
    for w in errs[2..].windows(2) {
        assert!(w[1] < w[0], "non-monotone convergence: {errs:?}");
    }
}
