//! Adaptive parameter selection for the Taylor-series expm algorithm.
//!
//! Ports `LazyOperatorNormInfo` and `_fragment_3_1` from
//! `parallel-sparse-tools/expm_multiply_parallel_core.py`, which in turn
//! calls `_fragment_3_1` imported from `scipy.sparse.linalg._expm_multiply`.
//!
//! Reference: Al-Mohy & Higham, "Computing the Action of the Matrix
//! Exponential, with an Application to Exponential Integrators" (2011).

use std::collections::HashMap;

use super::norm_est::onenorm_matrix_power_nnm;
use super::shifted_op::ShiftedOp;
use quspin_types::ExpmComputation;
use quspin_types::LinearOperator;

// ---------------------------------------------------------------------------
// θ_m table — Table A.3 of Al-Mohy & Higham (2011) / Higham (2008)
// ---------------------------------------------------------------------------
// THETA[m-1] is the threshold θ_m used for Taylor order m (m = 1..=55).
// When ||B||_1 ≤ θ_m a single partition (s = 1) achieves full precision.
#[allow(clippy::approx_constant)] // 3.14 is θ_28 from Table A.3, not π
const THETA: [f64; 55] = [
    2.29e-16, // m = 1
    2.58e-8,  // m = 2
    1.39e-5,  // m = 3
    3.40e-4,  // m = 4
    2.40e-3,  // m = 5
    9.07e-3,  // m = 6
    2.38e-2,  // m = 7
    5.00e-2,  // m = 8
    8.96e-2,  // m = 9
    1.44e-1,  // m = 10
    2.14e-1,  // m = 11
    3.00e-1,  // m = 12
    4.00e-1,  // m = 13
    5.14e-1,  // m = 14
    6.41e-1,  // m = 15
    7.80e-1,  // m = 16
    9.31e-1,  // m = 17
    1.09,     // m = 18
    1.26,     // m = 19
    1.44,     // m = 20
    1.62,     // m = 21
    1.82,     // m = 22
    2.03,     // m = 23
    2.24,     // m = 24
    2.45,     // m = 25
    2.68,     // m = 26
    2.91,     // m = 27
    3.14,     // m = 28
    3.38,     // m = 29
    3.63,     // m = 30
    3.88,     // m = 31
    4.14,     // m = 32
    4.40,     // m = 33
    4.67,     // m = 34
    4.94,     // m = 35
    5.22,     // m = 36
    5.50,     // m = 37
    5.78,     // m = 38
    6.07,     // m = 39
    6.36,     // m = 40
    6.65,     // m = 41
    6.95,     // m = 42
    7.25,     // m = 43
    7.56,     // m = 44
    7.86,     // m = 45
    8.17,     // m = 46
    8.48,     // m = 47
    8.80,     // m = 48
    9.11,     // m = 49
    9.43,     // m = 50
    9.76,     // m = 51
    10.09,    // m = 52
    10.42,    // m = 53
    10.75,    // m = 54
    11.08,    // m = 55
];

/// Look up `θ_m` for `m ∈ 1..=55`.
#[inline]
fn theta(m: usize) -> f64 {
    debug_assert!(
        (1..=55).contains(&m),
        "theta index m={m} out of range 1..=55"
    );
    THETA[m - 1]
}

// ---------------------------------------------------------------------------
// LazyNormInfo
// ---------------------------------------------------------------------------

/// Lazily evaluated norm information for the operator `B = a*(A - μI)`.
///
/// Mirrors `LazyOperatorNormInfo` in `expm_multiply_parallel_core.py`.
pub struct LazyNormInfo<'a, V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    /// The linear operator wrapping the matrix and baked-in coefficients.
    op: &'a Op,
    /// Scalar factor: `B = a*(A - μI)`.
    a: V,
    /// Diagonal shift `μ = trace(A)/n`.
    mu: V,
    /// Precomputed exact value `|a| · ||A - μI||_1`.
    onenorm_exact: f64,
    /// Cache for `d(p) = ||B^p||_1^(1/p)`.
    d_cache: HashMap<usize, f64>,
    /// Number of probe vectors used in the 1-norm estimator.
    ell: usize,
}

impl<'a, V, Op> LazyNormInfo<'a, V, Op>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    /// Construct a `LazyNormInfo`.
    ///
    /// # Arguments
    /// - `op`            — linear operator (matrix + baked-in coefficients)
    /// - `a`             — scalar factor in `B = a*(A - μI)`
    /// - `mu`            — diagonal shift (usually `trace(A) / n`)
    /// - `onenorm_exact` — precomputed value `|a| · ||A - μI||_1`
    /// - `ell`           — number of probe vectors for the 1-norm estimator
    pub fn new(op: &'a Op, a: V, mu: V, onenorm_exact: f64, ell: usize) -> Self {
        Self {
            op,
            a,
            mu,
            onenorm_exact,
            d_cache: HashMap::new(),
            ell,
        }
    }

    /// Exact 1-norm of the operator: `|a| · ||A - μI||_1`.
    ///
    /// Mirrors `LazyOperatorNormInfo.onenorm()` in the Python source.
    pub fn onenorm(&self) -> f64 {
        self.onenorm_exact
    }

    /// Lazily estimate `d(p) = ||B^p||_1^(1/p)` where `B = a*(A - μI)`.
    ///
    /// Mirrors `LazyOperatorNormInfo.d()` in the Python source.
    pub fn d(&mut self, p: usize) -> f64 {
        if let Some(&cached) = self.d_cache.get(&p) {
            return cached;
        }
        let b = ShiftedOp::new(self.op, self.a, self.mu);
        let est = onenorm_matrix_power_nnm(&b, p, self.ell);
        // Convert V::Real → f64 via round-trip through Complex<f64>.
        let est_f64 = V::from_real(est).to_complex().re;
        // d(p) = ||B^p||_1^(1/p)
        let d_p = if est_f64 == 0.0 {
            0.0
        } else {
            est_f64.powf(1.0 / p as f64)
        };
        self.d_cache.insert(p, d_p);
        d_p
    }

    /// `α(p) = max(d(p), d(p+1))`.
    ///
    /// Mirrors `LazyOperatorNormInfo.alpha()` in the Python source.
    pub fn alpha(&mut self, p: usize) -> f64 {
        let dp = self.d(p);
        let dp1 = self.d(p + 1);
        dp.max(dp1)
    }
}

// ---------------------------------------------------------------------------
// fragment_3_1
// ---------------------------------------------------------------------------

/// Compute the optimal Taylor-series parameters `(m_star, s)`.
///
/// Ports `_fragment_3_1` from `scipy.sparse.linalg._expm_multiply`, which
/// implements Fragment 3.1 of Al-Mohy & Higham (2011).
///
/// # Arguments
/// - `norm_info` — mutable reference; `d(p)` values are cached lazily
/// - `n0`        — number of columns in the input (typically 1)
/// - `tol`       — machine precision / 2 (not used in the core loop but kept
///   for API compatibility with the Python original)
/// - `m_max`     — maximum Taylor order to consider (typically 55)
///
/// # Returns
/// `(m_star, s)` where `m_star` is the optimal Taylor truncation order and
/// `s` is the number of matrix-exponential partitions (scaling count).
///
/// Returns `(0, 1)` when the matrix is small enough to skip scaling entirely.
pub fn fragment_3_1<V, Op>(
    norm_info: &mut LazyNormInfo<V, Op>,
    n0: usize,
    _tol: f64,
    m_max: usize,
) -> (usize, usize)
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    // Condition 3.13: norm is small enough that s=1 suffices for all m.
    if condition_3_13(norm_info, n0, m_max) {
        return (0, 1);
    }

    let mut best_cost: Option<usize> = None;
    let mut m_star = 1_usize;
    let mut s_star = 1_usize;

    // Outer loop over p (controls which alpha estimate we use).
    // For a given p, the relevant m range is [p*(p-1)-1, m_max].
    // When p*(p-1)-1 > m_max the inner range is empty and we stop.
    for p in 2_usize.. {
        let m_start = p.saturating_mul(p - 1).saturating_sub(1).max(1);
        if m_start > m_max {
            break;
        }

        // alpha(p) is the same for all m in this p-slice; compute it once.
        let alpha = norm_info.alpha(p);

        for m in m_start..=m_max {
            let theta_m = theta(m);
            // s = ceil(alpha / theta_m), minimum 1.
            let s = if alpha <= 0.0 {
                1_usize
            } else {
                ((alpha / theta_m).ceil() as usize).max(1)
            };
            let cost = m * s;
            if best_cost.is_none() || cost < best_cost.unwrap() {
                best_cost = Some(cost);
                m_star = m;
                s_star = s;
            }
        }
    }

    (m_star, s_star)
}

// ---------------------------------------------------------------------------
// condition_3_13  (internal helper)
// ---------------------------------------------------------------------------

/// Condition 3.13 from Al-Mohy & Higham (2011).
///
/// Returns `true` **only** when the operator 1-norm is exactly zero, i.e.
/// `a = 0` or `A = μI`.  In that case there are no Taylor terms to compute
/// and the result is simply `exp(a·μ) · f` — the algorithm's `m_star = 0`
/// fast path.
///
/// For any non-zero norm the optimisation loop in [`fragment_3_1`] determines
/// the correct `(m_star, s)` pair.  The wider threshold from the paper (which
/// says s = 1 suffices, not that m_star = 0 is valid) is deliberately not
/// used here: returning `(0, 1)` with a non-zero norm would drop all Taylor
/// corrections and give an incorrect result.
fn condition_3_13<V, Op>(norm_info: &mut LazyNormInfo<V, Op>, _n0: usize, _m_max: usize) -> bool
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    norm_info.onenorm() == 0.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theta_lookup() {
        // Spot-check a few known values from the paper.
        assert!((theta(1) - 2.29e-16).abs() < 1e-18);
        assert!((theta(10) - 1.44e-1).abs() < 1e-10);
        assert!((theta(55) - 11.08).abs() < 1e-6);
    }
}
