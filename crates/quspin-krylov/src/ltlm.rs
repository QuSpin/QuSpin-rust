use super::eig::TridiagEigen;
use num_complex::Complex;

type C64 = Complex<f64>;

/// Compute Krylov-space coefficients for `e^{-β(H - E_shift)/2}|r⟩`.
///
/// Returns coefficients `g_j = Σ_n c_{0,n} · e^{-β (E_n - E_shift) / 2} · c_{j,n}`
/// such that `e^{-β(H - E_shift)/2}|r⟩ = Σ_j g_j |q_j⟩`.
///
/// Usage: pass the result to `LanczosBasis::lin_comb` (or
/// `LanczosBasisIter::lin_comb`) to get the full-space vector `|φ⟩`,
/// then compute `⟨φ|O|φ⟩` for the LTLM observable estimate.
///
/// `e_shift` scales `|φ⟩` by `e^{β·E_shift/2}` and hence `⟨φ|O|φ⟩` by
/// `e^{β·E_shift}` — matching [`ftlm_partition`](super::ftlm::ftlm_partition),
/// so the LTLM ratio is unchanged provided the same shift is used for every
/// sample.  Without it the half-exponent `e^{-β E_n/2}` still overflows at low
/// temperature, just at twice the `β` the FTLM partition does.
pub fn ltlm_coeffs(eig: &TridiagEigen, beta: f64, e_shift: f64) -> Vec<C64> {
    let k = eig.k;
    let mut coeffs = vec![C64::default(); k];

    for n in 0..k {
        let c0n = eig.vec_element(0, n);
        let weight = (-beta * (eig.eigenvalues[n] - e_shift) / 2.0).exp();
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
        let coeffs = ltlm_coeffs(&eig, 0.0, 0.0);

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

        let coeffs = ltlm_coeffs(&eig, beta, 0.0);
        let norm_sq: f64 = coeffs.iter().map(|c| c.norm_sqr()).sum();
        let z = ftlm_partition(&eig, beta, 0.0);

        assert!((norm_sq - z).abs() < 1e-12, "||φ||² = {norm_sq}, Z = {z}",);
    }

    #[test]
    fn shift_keeps_norm_squared_equal_to_the_shifted_partition() {
        // ||φ||² = Z_r must keep holding once both carry the same shift —
        // that identity is what makes the LTLM ratio shift-invariant.
        use super::super::ftlm::ftlm_partition;

        let eig = solve_tridiagonal(&[1.0, -1.0, 0.5], &[0.5, 0.3]);
        let beta = 1.5;
        let shift = -2.0;

        let coeffs = ltlm_coeffs(&eig, beta, shift);
        let norm_sq: f64 = coeffs.iter().map(|c| c.norm_sqr()).sum();
        let z = ftlm_partition(&eig, beta, shift);

        assert!(
            (norm_sq - z).abs() < 1e-12 * z.abs().max(1.0),
            "||φ||² = {norm_sq}, Z = {z}",
        );
    }

    #[test]
    fn shift_prevents_overflow_at_low_temperature() {
        // The half-exponent overflows later than FTLM's, but it does overflow.
        let eig = solve_tridiagonal(&[-35.0, -30.0], &[1.0]);
        let beta = 50.0;

        let plain = ltlm_coeffs(&eig, beta, 0.0);
        assert!(
            plain.iter().any(|c| !c.is_finite()),
            "expected the unshifted coefficients to overflow",
        );

        let e_min = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let shifted = ltlm_coeffs(&eig, beta, e_min);
        assert!(
            shifted.iter().all(|c| c.is_finite()),
            "shifted coefficients should all be finite, got {shifted:?}",
        );
    }

    #[test]
    fn coeffs_are_real_for_real_system() {
        // For a real Hermitian system with real starting vector,
        // all coefficients should be real.
        let eig = solve_tridiagonal(&[2.0, -0.5], &[1.0]);
        let coeffs = ltlm_coeffs(&eig, 0.7, 0.0);

        for (j, c) in coeffs.iter().enumerate() {
            assert!(c.im.abs() < 1e-14, "g[{j}].im = {}, expected 0", c.im,);
        }
    }
}
