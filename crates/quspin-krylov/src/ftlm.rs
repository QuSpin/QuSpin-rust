use super::eig::TridiagEigen;
use num_complex::Complex;

type C64 = Complex<f64>;

/// Partition function contribution from a single FTLM sample.
///
/// Computes `Z_r = Σ_n |c_{0,n}|² e^{-β E_n}` where `c_{0,n}` is the
/// first component (overlap with starting vector) of the n-th eigenvector
/// of the tridiagonal matrix.
pub fn ftlm_partition(eig: &TridiagEigen, beta: f64) -> f64 {
    (0..eig.k)
        .map(|n| {
            let c0n = eig.vec_element(0, n);
            c0n * c0n * (-beta * eig.eigenvalues[n]).exp()
        })
        .sum()
}

/// Observable contribution from a single FTLM sample.
///
/// Computes `⟨O⟩_r · Z_r = Σ_n c_{0,n}* · e^{-β E_n} · (Σ_j c_{j,n} · o_j)`
/// where `o_j = ⟨q_j|O|r⟩` are the observable matrix elements between
/// each Krylov basis vector and the starting vector.
///
/// # Arguments
/// - `eig` — eigendecomposition of the tridiagonal matrix
/// - `obs_elements` — `⟨q_j|O|r⟩` for j = 0..k (length must equal `eig.k`)
/// - `beta` — inverse temperature
pub fn ftlm_observable(eig: &TridiagEigen, obs_elements: &[C64], beta: f64) -> C64 {
    debug_assert_eq!(obs_elements.len(), eig.k);

    (0..eig.k)
        .map(|n| {
            let c0n = C64::new(eig.vec_element(0, n), 0.0);
            let weight = (-beta * eig.eigenvalues[n]).exp();

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
        let z = ftlm_partition(&eig, 0.0);
        assert!((z - 1.0).abs() < 1e-12, "Z(β=0) = {z}, expected 1.0",);
    }

    #[test]
    fn partition_positive_beta_less_than_one() {
        // At β>0, the ground state dominates but Z_r < 1 unless
        // the starting vector is exactly the ground state.
        // Actually Z_r = Σ |c_{0n}|² e^{-βEn} which can be > 1 if E < 0.
        // For positive eigenvalues, Z_r < 1.
        let eig = solve_tridiagonal(&[2.0, 3.0], &[0.5]);
        let z = ftlm_partition(&eig, 1.0);
        // All eigenvalues positive → all weights < 1 → Z < 1
        assert!(z < 1.0, "Z = {z}");
        assert!(z > 0.0, "Z = {z}");
    }

    #[test]
    fn partition_large_beta_approaches_ground_state() {
        // At large β, the ground state contribution dominates.
        let eig = solve_tridiagonal(&[0.0, 0.0], &[1.0]);
        // Eigenvalues: ±1. Ground state: -1.
        let z_large = ftlm_partition(&eig, 100.0);
        let z_small = ftlm_partition(&eig, 0.01);

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
        let z = ftlm_partition(&eig, beta);
        let oz = ftlm_observable(&eig, &obs, beta);

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

        let oz = ftlm_observable(&eig, &obs, 0.0);
        // Should equal obs[0] = 3 + i
        assert!(
            (oz - obs[0]).norm() < 1e-12,
            "⟨O⟩·Z at β=0: {oz}, expected {}",
            obs[0],
        );
    }
}
