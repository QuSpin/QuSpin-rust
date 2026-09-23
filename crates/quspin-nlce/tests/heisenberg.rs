//! 2D spin-1/2 Heisenberg antiferromagnet `H = Σ S_i · S_j` on the square
//! lattice.
//!
//! Exact checks: the high-temperature series
//! `E/N = −(3/8) β − (3/32) β² + O(β³)` (`z = 4`, no triangles) and the
//! matching entropy series,
//! and order-to-order convergence at high T. Qualitative checks against the
//! known finite-T behaviour (Tang, Khatami & Rigol, arXiv:1207.3366; QMC
//! places the specific-heat maximum `C ≈ 0.46` near `T ≈ 0.6`).

use quspin_nlce::*;

fn run(temps: &[f64], order: usize) -> NlceResult<Thermo> {
    let solver = ExactDiagSolver::new(temps.to_vec()).unwrap();
    run_nlce(
        &RectangleGenerator::new(SquareLattice, order),
        &solver,
        &Xxz::heisenberg(1.0),
    )
    .unwrap()
}

#[test]
fn high_temperature_series() {
    let temps = [5.0, 10.0, 20.0, 50.0];
    let result = run(&temps, 7);
    let last = result.last_partial_sum().unwrap();
    let mut resid = Vec::new();
    for (i, &t) in temps.iter().enumerate() {
        let b = 1.0 / t;
        let r = (last.energy[i] + 0.375 * b + 3.0 / 32.0 * b * b) / b.powi(3);
        assert!(r.abs() < 0.1, "T={t}: β³ residual coefficient {r}");
        resid.push(r);
        // S = ln Z + βE = ln 2 − (3/16) β² − β³/16 + O(β⁴).
        let s_series = 2f64.ln() - 3.0 / 16.0 * b * b - b.powi(3) / 16.0;
        let ds = (last.entropy[i] - s_series).abs();
        assert!(
            ds < 0.1 * b.powi(4),
            "T={t}: entropy {} vs series {s_series}",
            last.entropy[i]
        );
    }
    // The residual tends to a constant (the exact β³ coefficient), which
    // pins the β and β² coefficients: a 1% error in the β² term would shift
    // it by ~0.05 at T = 50.
    assert!((resid[2] - resid[3]).abs() < 5e-3, "{resid:?}");
}

#[test]
fn converges_at_high_temperature() {
    let temps = [1.0, 1.5, 2.0, 5.0];
    // About 2–3× the observed order-6 → order-7 changes.
    let tol = [1.5e-2, 3e-4, 3e-5, 1e-8];
    let result = run(&temps, 7);
    let n = result.partial_sums.len();
    let (prev, last) = (&result.partial_sums[n - 2].1, &result.partial_sums[n - 1].1);
    for (i, &t) in temps.iter().enumerate() {
        for (name, a, b) in [
            ("energy", &prev.energy, &last.energy),
            ("entropy", &prev.entropy, &last.entropy),
            ("specific heat", &prev.specific_heat, &last.specific_heat),
        ] {
            let d = (a[i] - b[i]).abs();
            assert!(
                d < tol[i],
                "T={t}: {name} changes by {d:e} between the last two orders"
            );
        }
    }
}

#[test]
fn qualitative_thermodynamics() {
    let temps = [0.7, 1.0, 1.5, 2.0, 5.0];
    let result = run(&temps, 7);
    let last = result.last_partial_sum().unwrap();
    // Energy rises monotonically towards 0 and stays above E0 ≈ −0.67.
    for w in last.energy.windows(2) {
        assert!(w[0] < w[1]);
    }
    assert!(last.energy[0] > -0.67 && last.energy[4] < 0.0);
    // Specific heat: already past its maximum at T ≈ 0.7, of the QMC size.
    assert!(
        (0.35..0.55).contains(&last.specific_heat[0]),
        "{:?}",
        last.specific_heat
    );
    for w in last.specific_heat.windows(2) {
        assert!(w[0] > w[1]);
    }
    // Entropy below ln 2 and increasing.
    for w in last.entropy.windows(2) {
        assert!(w[0] < w[1] && w[1] < 2f64.ln());
    }
}

#[test]
fn cache_reuse_skips_solved_clusters() {
    let temps = [2.0];
    let solver = ExactDiagSolver::new(temps.to_vec()).unwrap();
    let model = Xxz::heisenberg(1.0);
    let small = run_nlce(&RectangleGenerator::new(SquareLattice, 5), &solver, &model).unwrap();
    let big_cached = run_nlce_cached(
        &RectangleGenerator::new(SquareLattice, 6),
        &solver,
        &model,
        small.cache,
    )
    .unwrap();
    let big = run(&temps, 6);
    for ((oa, a), (ob, b)) in big_cached.partial_sums.iter().zip(&big.partial_sums) {
        assert_eq!(oa, ob);
        assert!(a.max_abs_diff(b) < 1e-14);
    }
    assert_eq!(
        Bare.resum(
            &big.partial_sums
                .iter()
                .map(|(_, p)| p.clone())
                .collect::<Vec<_>>()
        )
        .unwrap(),
        *big.last_partial_sum().unwrap()
    );
}
