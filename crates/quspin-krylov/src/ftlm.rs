use super::eig::TridiagEigen;
use num_complex::Complex;

type C64 = Complex<f64>;

/// Partition function contribution from a single FTLM sample.
///
/// Computes `Z_r = Σ_n |c_{0,n}|² e^{-β (E_n - E_shift)}` where `c_{0,n}` is
/// the first component (overlap with starting vector) of the n-th eigenvector
/// of the tridiagonal matrix.
///
/// # Energy shift
///
/// `e_shift` is subtracted from every Ritz value before exponentiating.
/// Without it, `e^{-β E_n}` overflows to `inf` whenever `β·|E_min|` exceeds
/// ~709 — a 20-site Heisenberg chain has `E_0 ≈ -35`, so any `β ≳ 20` (an
/// ordinary low-temperature FTLM regime) produces `inf`, and the caller's
/// `Σ_r ⟨O⟩Z_r / Σ_r Z_r` comes out `NaN`.  Setting `e_shift ≈ E_0` keeps the
/// largest weight at `O(1)`.
///
/// The shift rescales `Z_r` by `e^{β·E_shift}`, which cancels in that ratio —
/// **but only if every sample uses the same `e_shift`**.  A per-sample shift
/// (e.g. each sample's own minimum Ritz value) silently biases the estimator,
/// because the samples are then no longer commensurately weighted.  Callers
/// must hold `e_shift` fixed across the whole average; the Python bindings
/// enforce this by storing it on the estimator object rather than taking it
/// per `sample()` call.
pub fn ftlm_partition(eig: &TridiagEigen, beta: f64, e_shift: f64) -> f64 {
    (0..eig.k)
        .map(|n| {
            let c0n = eig.vec_element(0, n);
            c0n * c0n * (-beta * (eig.eigenvalues[n] - e_shift)).exp()
        })
        .sum()
}

/// Observable contribution from a single FTLM sample.
///
/// Computes
/// `⟨O⟩_r · Z_r = Σ_n c_{0,n}* · e^{-β (E_n - E_shift)} · (Σ_j c_{j,n} · o_j)`
/// where `o_j = ⟨q_j|O|r⟩` are the observable matrix elements between
/// each Krylov basis vector and the starting vector.
///
/// # Arguments
/// - `eig` — eigendecomposition of the tridiagonal matrix
/// - `obs_elements` — `⟨q_j|O|r⟩` for j = 0..k (length must equal `eig.k`)
/// - `beta` — inverse temperature
/// - `e_shift` — energy shift; see [`ftlm_partition`] for why it must be the
///   same for every sample in an average
pub fn ftlm_observable(eig: &TridiagEigen, obs_elements: &[C64], beta: f64, e_shift: f64) -> C64 {
    debug_assert_eq!(obs_elements.len(), eig.k);

    (0..eig.k)
        .map(|n| {
            let c0n = C64::new(eig.vec_element(0, n), 0.0);
            let weight = (-beta * (eig.eigenvalues[n] - e_shift)).exp();

            // Σ_j c_{j,n} · o_j
            let proj: C64 = (0..eig.k)
                .map(|j| C64::new(eig.vec_element(j, n), 0.0) * obs_elements[j])
                .sum();

            c0n.conj() * C64::new(weight, 0.0) * proj
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::eig::solve_tridiagonal;
    use super::*;

    #[test]
    fn partition_beta_zero_is_one() {
        // At β=0, all Boltzmann weights are 1.
        // Z_r = Σ_n |c_{0,n}|² = 1 (eigenvectors are orthonormal,
        // first row of orthogonal matrix has unit norm).
        let eig = solve_tridiagonal(&[1.0, -1.0, 0.5], &[0.5, 0.3]);
        let z = ftlm_partition(&eig, 0.0, 0.0);
        assert!((z - 1.0).abs() < 1e-12, "Z(β=0) = {z}, expected 1.0",);
    }

    #[test]
    fn partition_positive_beta_less_than_one() {
        // At β>0, the ground state dominates but Z_r < 1 unless
        // the starting vector is exactly the ground state.
        // Actually Z_r = Σ |c_{0n}|² e^{-βEn} which can be > 1 if E < 0.
        // For positive eigenvalues, Z_r < 1.
        let eig = solve_tridiagonal(&[2.0, 3.0], &[0.5]);
        let z = ftlm_partition(&eig, 1.0, 0.0);
        // All eigenvalues positive → all weights < 1 → Z < 1
        assert!(z < 1.0, "Z = {z}");
        assert!(z > 0.0, "Z = {z}");
    }

    #[test]
    fn partition_large_beta_approaches_ground_state() {
        // At large β, the ground state contribution dominates.
        let eig = solve_tridiagonal(&[0.0, 0.0], &[1.0]);
        // Eigenvalues: ±1. Ground state: -1.
        let z_large = ftlm_partition(&eig, 100.0, 0.0);
        let z_small = ftlm_partition(&eig, 0.01, 0.0);

        // At large β, Z_r ≈ |c_{0,gs}|² e^{-β·(-1)} ≈ 0.5 * e^{100}
        // The ratio should show exponential growth with ground state
        assert!(z_large > z_small);
    }

    #[test]
    fn observable_identity_equals_partition() {
        // If O = I, then o_j = ⟨q_j|I|r⟩ = ⟨q_j|r⟩ = δ_{j,0}
        // (since r = q_0). So ⟨I⟩·Z = Z, meaning ftlm_observable = Z.
        let eig = solve_tridiagonal(&[1.0, -1.0, 0.5], &[0.5, 0.3]);
        let k = eig.k;
        let mut obs = vec![C64::default(); k];
        obs[0] = C64::new(1.0, 0.0); // ⟨q_0|I|r⟩ = 1

        let beta = 0.5;
        let z = ftlm_partition(&eig, beta, 0.0);
        let oz = ftlm_observable(&eig, &obs, beta, 0.0);

        assert!((oz.re - z).abs() < 1e-12, "⟨I⟩·Z = {oz}, Z = {z}",);
        assert!(oz.im.abs() < 1e-12, "imaginary part = {}", oz.im);
    }

    #[test]
    fn observable_at_beta_zero() {
        // At β=0, ⟨O⟩ = Tr(O)/dim. For a single sample:
        // oz_r = Σ_n c_{0,n}* · 1 · (Σ_j c_{j,n} o_j)
        //      = Σ_j o_j · (Σ_n c_{0,n}* c_{j,n})
        //      = Σ_j o_j · δ_{0,j}  (orthogonality of eigenvectors)
        //      = o_0
        let eig = solve_tridiagonal(&[0.0, 0.0], &[1.0]);
        let obs = vec![C64::new(3.0, 1.0), C64::new(7.0, -2.0)];

        let oz = ftlm_observable(&eig, &obs, 0.0, 0.0);
        // Should equal obs[0] = 3 + i
        assert!(
            (oz - obs[0]).norm() < 1e-12,
            "⟨O⟩·Z at β=0: {oz}, expected {}",
            obs[0],
        );
    }

    // --- energy shift ---

    #[test]
    fn shift_rescales_partition_by_exp_beta_shift() {
        let eig = solve_tridiagonal(&[1.0, -1.0, 0.5], &[0.5, 0.3]);
        let beta = 0.7;
        let shift = -1.25;

        let z_plain = ftlm_partition(&eig, beta, 0.0);
        let z_shifted = ftlm_partition(&eig, beta, shift);

        let expected = z_plain * (beta * shift).exp();
        assert!(
            (z_shifted - expected).abs() < 1e-12 * expected.abs().max(1.0),
            "Z(shift) = {z_shifted}, expected Z(0)·e^(βs) = {expected}",
        );
    }

    #[test]
    fn shift_cancels_in_the_observable_ratio() {
        // The whole point: <O> = Σ_r oz_r / Σ_r z_r must not depend on the
        // shift, provided every sample uses the same one.
        let eig = solve_tridiagonal(&[1.0, -1.0, 0.5], &[0.5, 0.3]);
        let obs = vec![C64::new(3.0, 1.0), C64::new(7.0, -2.0), C64::new(-1.0, 0.5)];
        let beta = 1.3;

        let ratio = |shift: f64| {
            ftlm_observable(&eig, &obs, beta, shift) / ftlm_partition(&eig, beta, shift)
        };

        let reference = ratio(0.0);
        for shift in [-5.0, -1.0, 0.25, 3.0] {
            let got = ratio(shift);
            assert!(
                (got - reference).norm() < 1e-12,
                "shift {shift} changed the ratio: {got} vs {reference}",
            );
        }
    }

    #[test]
    fn shift_prevents_overflow_at_low_temperature() {
        // Stand-in for a real low-temperature run: a ground state well below
        // zero and a β large enough that β·|E_0| > 709.
        let eig = solve_tridiagonal(&[-35.0, -30.0], &[1.0]);
        let beta = 25.0;

        let z_unshifted = ftlm_partition(&eig, beta, 0.0);
        assert!(
            z_unshifted.is_infinite(),
            "expected the unshifted weight to overflow, got {z_unshifted}",
        );

        let e_min = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let z_shifted = ftlm_partition(&eig, beta, e_min);
        assert!(
            z_shifted.is_finite() && z_shifted > 0.0,
            "shifted partition should be finite and positive, got {z_shifted}",
        );

        // And the observable ratio is still recoverable.
        let obs = vec![C64::new(2.0, 0.0), C64::new(-1.0, 0.0)];
        let oz = ftlm_observable(&eig, &obs, beta, e_min);
        assert!(
            (oz / z_shifted).norm().is_finite(),
            "shifted ratio is not finite",
        );
    }
}
