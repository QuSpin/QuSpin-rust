use super::eig::TridiagEigen;
use num_complex::Complex;

type C64 = Complex<f64>;

/// Compute Krylov-space coefficients for `e^{-βH/2}|r⟩`.
///
/// Returns coefficients `g_j = Σ_n c_{0,n} · e^{-β E_n / 2} · c_{j,n}`
/// such that `e^{-βH/2}|r⟩ = Σ_j g_j |q_j⟩`.
///
/// Usage: pass the result to `LanczosBasis::lin_comb` (or
/// `LanczosBasisIter::lin_comb`) to get the full-space vector `|φ⟩`,
/// then compute `⟨φ|O|φ⟩` for the LTLM observable estimate.
pub fn ltlm_coeffs(eig: &TridiagEigen, beta: f64) -> Vec<C64> {
    let k = eig.k;
    let mut coeffs = vec![C64::default(); k];

    for n in 0..k {
        let c0n = eig.vec_element(0, n);
        let weight = (-beta * eig.eigenvalues[n] / 2.0).exp();
        let factor = c0n * weight;

        for (j, cj) in coeffs.iter_mut().enumerate() {
            *cj += C64::new(factor * eig.vec_element(j, n), 0.0);
        }
    }

    coeffs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::eig::solve_tridiagonal;
    use super::*;

    #[test]
    fn coeffs_beta_zero_is_delta() {
        // At β=0, e^{0} = I, so g_j = Σ_n c_{0,n} c_{j,n} = δ_{0,j}
        let eig = solve_tridiagonal(&[1.0, -1.0, 0.5], &[0.5, 0.3]);
        let coeffs = ltlm_coeffs(&eig, 0.0);

        assert!(
            (coeffs[0].re - 1.0).abs() < 1e-12,
            "g[0] = {}, expected 1",
            coeffs[0],
        );
        for (j, c) in coeffs.iter().enumerate().skip(1) {
            assert!(c.norm() < 1e-12, "g[{j}] = {c}, expected 0",);
        }
    }

    #[test]
    fn coeffs_norm_squared_equals_partition() {
        // ||e^{-βH/2}|r⟩||² = ⟨r|e^{-βH}|r⟩ = Z_r (the FTLM partition).
        // Since |φ⟩ = Σ_j g_j |q_j⟩ and {q_j} are orthonormal,
        // ||φ||² = Σ_j |g_j|².
        use super::super::ftlm::ftlm_partition;

        let eig = solve_tridiagonal(&[1.0, -1.0, 0.5], &[0.5, 0.3]);
        let beta = 1.5;

        let coeffs = ltlm_coeffs(&eig, beta);
        let norm_sq: f64 = coeffs.iter().map(|c| c.norm_sqr()).sum();
        let z = ftlm_partition(&eig, beta);

        assert!((norm_sq - z).abs() < 1e-12, "||φ||² = {norm_sq}, Z = {z}",);
    }

    #[test]
    fn coeffs_are_real_for_real_system() {
        // For a real Hermitian system with real starting vector,
        // all coefficients should be real.
        let eig = solve_tridiagonal(&[2.0, -0.5], &[1.0]);
        let coeffs = ltlm_coeffs(&eig, 0.7);

        for (j, c) in coeffs.iter().enumerate() {
            assert!(c.im.abs() < 1e-14, "g[{j}].im = {}, expected 0", c.im,);
        }
    }
}
