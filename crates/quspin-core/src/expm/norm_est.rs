use super::compute::ExpmComputation;
use super::linear_operator::LinearOperator;

/// Estimate `‖(a·(A − μI))^p‖_1` using a randomised block 1-norm estimator.
///
/// Implements a simplified variant of the Higham–Tisseur (2000) block algorithm
/// that alternates forward (`B = a*(A - μI)`) and backward (`B* = ā*(A* - μ̄I)`)
/// matrix–vector products to iteratively refine the estimate.
///
/// # Arguments
/// - `op`  — linear operator implementing `A · x` and `A^T · x`
/// - `a`   — global scalar factor in `B = a*(A - μI)`
/// - `mu`  — diagonal shift (usually `trace(A)/n`)
/// - `p`   — power to apply
/// - `ell` — number of probe vectors (typically 2)
///
/// Returns a lower bound on `‖B^p‖_1`.
///
/// # Precision note
///
/// Internal arithmetic runs in `V::Real`, which is `f32` when `V = f32` or
/// `Complex<f32>`.  The caller (`LazyNormInfo::d`) converts the result to
/// `f64` before parameter selection, so (m*, s) choices are made in full
/// double precision regardless of the compute dtype.  The estimator itself
/// may be slightly less accurate for `f32` inputs, but this is an acceptable
/// tradeoff — the norm estimate is only used to choose scaling parameters,
/// not as a final result.
pub fn onenorm_matrix_power_nnm<V, Op>(op: &Op, a: V, mu: V, p: usize, ell: usize) -> V::Real
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    let n = op.dim();
    if n == 0 {
        return V::Real::default();
    }
    if p == 0 {
        // B^0 = I, ||I||_1 = 1
        return V::real_from_f64(1.0);
    }

    let a_conj = V::from_complex(a.to_complex().conj());
    let mu_conj = V::from_complex(mu.to_complex().conj());
    let mut est = V::Real::default();

    // -----------------------------------------------------------------------
    // Forward pass: apply B = a*(A - μI) p times to ell probe vectors.
    // Probe vectors use alternating sign patterns (deterministic, no rng dep).
    // -----------------------------------------------------------------------
    let mut ax = vec![V::default(); n];
    let mut best_col: Option<Vec<V>> = None;

    for j in 0..ell {
        let mut x: Vec<V> = (0..n)
            .map(|i| {
                let sign = if (i + j) % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
                V::from_real(V::real_from_f64(sign))
            })
            .collect();

        apply_bp(op, a, mu, p, &mut x, &mut ax);

        let col_norm: V::Real = x
            .iter()
            .map(|v| v.abs_val())
            .fold(V::Real::default(), |a, b| a + b);
        if col_norm > est {
            est = col_norm;
            best_col = Some(x);
        }
    }

    // -----------------------------------------------------------------------
    // Refinement step (one Higham–Tisseur iteration).
    // -----------------------------------------------------------------------
    if let Some(y) = best_col {
        // s = sign(y) = y / |y|  (element-wise; treat zero as +1)
        // Uses to_complex()/from_complex() to handle both real and complex V.
        let s: Vec<V> = y
            .iter()
            .map(|v| {
                let vc = v.to_complex();
                let nrm = vc.norm();
                if nrm == 0.0 {
                    V::from_real(V::real_from_f64(1.0))
                } else {
                    V::from_complex(vc / nrm)
                }
            })
            .collect();

        // Apply B*^p to s: B* = ā*(A^T - μ̄I)
        let mut z = s;
        let mut az = vec![V::default(); n];
        apply_bp_adj(op, a_conj, mu_conj, p, &mut z, &mut az);

        // Row with the largest |z[i]| gives the best unit-vector starting point.
        let i_star = z
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.abs_val()
                    .partial_cmp(&b.1.abs_val())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut x_unit = vec![V::default(); n];
        x_unit[i_star] = V::from_real(V::real_from_f64(1.0));

        apply_bp(op, a, mu, p, &mut x_unit, &mut ax);

        let col_norm: V::Real = x_unit
            .iter()
            .map(|v| v.abs_val())
            .fold(V::Real::default(), |a, b| a + b);
        if col_norm > est {
            est = col_norm;
        }
    }

    est
}

// ---------------------------------------------------------------------------
// Helpers: apply B^p and (B*)^p in-place
// ---------------------------------------------------------------------------

/// In-place: `x ← (a*(A − μI))^p · x`.  `work` is scratch space of length `n`.
fn apply_bp<V, Op>(op: &Op, a: V, mu: V, p: usize, x: &mut [V], work: &mut [V])
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    for _ in 0..p {
        op.dot(true, x, work)
            .expect("onenorm_matrix_power_nnm: dot failed");
        // work = a * (A·x − μ·x)
        for k in 0..x.len() {
            work[k] = a * (work[k] - mu * x[k]);
        }
        x.copy_from_slice(work);
    }
}

/// In-place: `x ← (ā*(A^T − μ̄I))^p · x`.  `work` is scratch space.
fn apply_bp_adj<V, Op>(op: &Op, a_conj: V, mu_conj: V, p: usize, x: &mut [V], work: &mut [V])
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    for _ in 0..p {
        op.dot_transpose(true, x, work)
            .expect("onenorm_matrix_power_nnm: dot_transpose failed");
        for k in 0..x.len() {
            work[k] = a_conj * (work[k] - mu_conj * x[k]);
        }
        x.copy_from_slice(work);
    }
}
