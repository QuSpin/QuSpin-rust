use super::eig::TridiagEigen;
use num_complex::Complex;

type C64 = Complex<f64>;

/// Evaluate the continued fraction representation of the resolvent.
///
/// Computes `G(z) = 1 / (z - α₀ - β₀² / (z - α₁ - β₁² / ...))`
///
/// This represents `⟨v̂|(z − H)⁻¹|v̂⟩` where `α`, `β` are the Lanczos
/// tridiagonal coefficients from starting vector `v̂ = v / ‖v‖`.
/// Multiply by `‖v‖²` to get `⟨v|(z − H)⁻¹|v⟩`.
///
/// Evaluates from the bottom of the fraction upward for numerical stability.
pub fn continued_fraction(alpha: &[f64], beta: &[f64], z: C64) -> C64 {
    let k = alpha.len();
    debug_assert!(k > 0);
    debug_assert_eq!(beta.len() + 1, k);

    // Start from the deepest level
    let mut g = z - C64::new(alpha[k - 1], 0.0);
    for j in (0..k - 1).rev() {
        g = z - C64::new(alpha[j], 0.0) - C64::new(beta[j] * beta[j], 0.0) / g;
    }
    C64::new(1.0, 0.0) / g
}

/// Compute the spectral function contribution from one FTLM dynamic sample.
///
/// For a random vector `|r⟩`:
/// - **Left Lanczos** on `|r⟩` provides Boltzmann-weighted Ritz values
/// - **Right Lanczos** on `A|r⟩` provides the continued-fraction resolvent
///
/// The spectral function contribution at frequency `ω` is:
/// ```text
/// S_r(ω) = -(‖Ar‖² / π) Σ_n |c_{0n}^L|² e^{-β E_n^L}
///           × Im[G_R(ω + E_n^L + iη)]
/// ```
///
/// # Arguments
/// - `left_eig` — eigendecomposition of the left tridiagonal (from `|r⟩`)
/// - `right_alpha`, `right_beta` — tridiagonal from the right Lanczos (from `A|r⟩`)
/// - `right_norm_sq` — `‖A|r⟩‖²`
/// - `beta_temp` — inverse temperature `β`
/// - `omegas` — frequency grid
/// - `eta` — Lorentzian broadening parameter
pub fn ftlm_dynamic_spectral(
    left_eig: &TridiagEigen,
    right_alpha: &[f64],
    right_beta: &[f64],
    right_norm_sq: f64,
    beta_temp: f64,
    omegas: &[f64],
    eta: f64,
) -> Vec<f64> {
    let inv_pi = -1.0 / std::f64::consts::PI;

    omegas
        .iter()
        .map(|&omega| {
            let mut s = 0.0;
            for n in 0..left_eig.k {
                let c0n = left_eig.vec_element(0, n);
                let weight = c0n * c0n * (-beta_temp * left_eig.eigenvalues[n]).exp();
                let z = C64::new(omega + left_eig.eigenvalues[n], eta);
                let g = continued_fraction(right_alpha, right_beta, z);
                s += weight * inv_pi * (right_norm_sq * g).im;
            }
            s
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::eig::solve_tridiagonal;
    use super::*;

    #[test]
    fn continued_fraction_1x1() {
        // T = [[α₀]], G(z) = 1/(z - α₀)
        let g = continued_fraction(&[2.0], &[], C64::new(5.0, 0.1));
        let expected = C64::new(1.0, 0.0) / C64::new(3.0, 0.1);
        assert!((g - expected).norm() < 1e-14);
    }

    #[test]
    fn continued_fraction_2x2_matches_inverse() {
        // T = [[α₀, β₀], [β₀, α₁]]
        // G(z) = ⟨e₁|(z-T)⁻¹|e₁⟩ = (z - α₁) / ((z-α₀)(z-α₁) - β₀²)
        let a0 = 1.0;
        let a1 = -0.5;
        let b0 = 0.7;
        let z = C64::new(3.0, 0.2);

        let g = continued_fraction(&[a0, a1], &[b0], z);
        let expected = (z - C64::new(a1, 0.0))
            / ((z - C64::new(a0, 0.0)) * (z - C64::new(a1, 0.0)) - C64::new(b0 * b0, 0.0));

        assert!(
            (g - expected).norm() < 1e-13,
            "cf = {g}, expected = {expected}",
        );
    }

    #[test]
    fn continued_fraction_poles_at_eigenvalues() {
        // Near an eigenvalue, Im[G] should have a peak.
        // T = [[0, 1], [1, 0]], eigenvalues ±1.
        let alpha = [0.0, 0.0];
        let beta = [1.0];
        let eta = 0.01;

        // At ω=1 (eigenvalue), large |Im(G)|
        let g_on = continued_fraction(&alpha, &beta, C64::new(1.0, eta));
        // At ω=0.5 (off eigenvalue), smaller |Im(G)|
        let g_off = continued_fraction(&alpha, &beta, C64::new(0.5, eta));

        assert!(
            g_on.im.abs() > g_off.im.abs(),
            "on-resonance |Im(G)| = {} should exceed off-resonance {}",
            g_on.im.abs(),
            g_off.im.abs(),
        );
    }

    #[test]
    fn continued_fraction_sum_rule() {
        // ∫ dω (-1/π) Im G(ω + iη) = 1 for normalized starting vector.
        // Approximate with trapezoidal rule on a wide grid.
        let alpha = [0.5, -0.3, 1.0];
        let beta = [0.8, 0.4];
        let eta = 0.1;

        let n_pts = 10000;
        let omega_min = -10.0;
        let omega_max = 10.0;
        let dw = (omega_max - omega_min) / n_pts as f64;

        let integral: f64 = (0..=n_pts)
            .map(|i| {
                let omega = omega_min + i as f64 * dw;
                let g = continued_fraction(&alpha, &beta, C64::new(omega, eta));
                (-1.0 / std::f64::consts::PI) * g.im * dw
            })
            .sum();

        assert!(
            (integral - 1.0).abs() < 0.01,
            "sum rule integral = {integral}, expected 1.0",
        );
    }

    #[test]
    fn dynamic_spectral_is_nonnegative() {
        // S(ω) should be non-negative for all ω.
        let left_eig = solve_tridiagonal(&[0.0, 0.0], &[1.0]);
        let right_alpha = [0.5, -0.5];
        let right_beta = [0.3];
        let right_norm_sq = 0.5;

        let omegas: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let s = ftlm_dynamic_spectral(
            &left_eig,
            &right_alpha,
            &right_beta,
            right_norm_sq,
            1.0,
            &omegas,
            0.1,
        );

        for (i, &si) in s.iter().enumerate() {
            assert!(si >= -1e-14, "S(ω={}) = {si} is negative", omegas[i],);
        }
    }

    #[test]
    fn dynamic_spectral_beta_zero_symmetric() {
        // At β=0, all Boltzmann weights are equal, so S(ω) should have
        // a specific symmetry structure.
        let left_eig = solve_tridiagonal(&[0.0, 0.0], &[1.0]);
        let right_alpha = [0.0, 0.0];
        let right_beta = [1.0];
        let right_norm_sq = 1.0;

        let s_pos = ftlm_dynamic_spectral(
            &left_eig,
            &right_alpha,
            &right_beta,
            right_norm_sq,
            0.0,
            &[2.0],
            0.1,
        )[0];
        let s_neg = ftlm_dynamic_spectral(
            &left_eig,
            &right_alpha,
            &right_beta,
            right_norm_sq,
            0.0,
            &[-2.0],
            0.1,
        )[0];

        // At β=0, S(ω) = S(-ω) (detailed balance with equal weights)
        assert!(
            (s_pos - s_neg).abs() < 1e-12,
            "S(2) = {s_pos}, S(-2) = {s_neg}",
        );
    }
}
