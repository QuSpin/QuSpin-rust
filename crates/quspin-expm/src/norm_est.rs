use quspin_types::ExpmComputation;
use quspin_types::LinearOperator;

use crate::shifted_op::ShiftedOp;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// XOR-shift 64-bit PRNG — period 2⁶⁴−1.
/// Returns +1.0 or −1.0 with equal probability.
#[inline]
fn xorshift_sign(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if (*state >> 32) & 1 == 0 {
        1.0_f64
    } else {
        -1.0_f64
    }
}

/// Apply `B^p` (or `(B^T)^p` when `transpose`) to every column of the
/// column-major matrix `x_mat` (shape n×t).  Returns `Y` column-major n×t.
fn apply_block<V, Op>(
    b: &ShiftedOp<V, Op>,
    p: usize,
    x_mat: &[V],
    n: usize,
    t: usize,
    work: &mut Vec<V>,
    transpose: bool,
) -> Vec<V>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    work.resize(n, V::default());
    let mut y_mat = vec![V::default(); n * t];
    for col in 0..t {
        let mut col_buf: Vec<V> = x_mat[col * n..(col + 1) * n].to_vec();
        if transpose {
            b.apply_pow_in_place_transpose(p, &mut col_buf, work);
        } else {
            b.apply_pow_in_place(p, &mut col_buf, work);
        }
        y_mat[col * n..(col + 1) * n].copy_from_slice(&col_buf);
    }
    y_mat
}

/// 1-norm of each column of the column-major n×t matrix `y_mat`.
fn col_onenorms<V: ExpmComputation>(y_mat: &[V], n: usize, t: usize) -> Vec<V::Real> {
    (0..t)
        .map(|col| {
            y_mat[col * n..(col + 1) * n]
                .iter()
                .map(|v| v.abs_val())
                .fold(V::Real::default(), |a, b| a + b)
        })
        .collect()
}

/// Index of the largest element.
fn argmax<R: PartialOrd + Copy>(v: &[R]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Element-wise complex sign: `v / |v|` when `|v| > 0`, else `+1`.
/// This is `sign_round_up` from the Higham–Tisseur paper.
fn sign_roundup<V: ExpmComputation>(v: V) -> V {
    let vc = v.to_complex();
    let nrm = vc.norm();
    if nrm == 0.0 {
        V::from_real(V::real_from_f64(1.0))
    } else {
        V::from_complex(vc / nrm)
    }
}

/// Apply `sign_roundup` element-wise to a flat column-major matrix.
fn sign_roundup_mat<V: ExpmComputation>(y: &[V]) -> Vec<V> {
    y.iter().map(|&v| sign_roundup(v)).collect()
}

/// Row-wise maximum absolute value across all `t` columns of `z_mat` (n×t).
fn row_max_abs<V: ExpmComputation>(z_mat: &[V], n: usize, t: usize) -> Vec<V::Real> {
    let mut h = vec![V::Real::default(); n];
    for col in 0..t {
        for i in 0..n {
            let val = z_mat[col * n + i].abs_val();
            if val > h[i] {
                h[i] = val;
            }
        }
    }
    h
}

/// True iff every column of `x_mat` is a unit-scalar multiple of some column
/// of `y_mat` — i.e. they carry the same directional information.
///
/// For real ±1 sign matrices this coincides with column equality.  For complex
/// sign matrices (all entries on the unit circle) it checks whether `x[:,cx]`
/// and `y[:,cy]` differ by a global phase factor: `x[0]/y[0]` is computed and
/// then verified against every element.
fn every_col_parallel<V: ExpmComputation>(
    x_mat: &[V],
    y_mat: &[V],
    n: usize,
    t_x: usize,
    t_y: usize,
) -> bool {
    if n == 0 {
        return true;
    }
    (0..t_x).all(|cx| {
        let xc = &x_mat[cx * n..(cx + 1) * n];
        (0..t_y).any(|cy| {
            let yc = &y_mat[cy * n..(cy + 1) * n];
            let y0 = yc[0].to_complex();
            if y0.norm() == 0.0 {
                return false;
            }
            let c = xc[0].to_complex() / y0;
            xc.iter()
                .zip(yc.iter())
                .all(|(&x, &y)| (x.to_complex() - c * y.to_complex()).norm() < 1e-10)
        })
    })
}

// ---------------------------------------------------------------------------
// Public estimator
// ---------------------------------------------------------------------------

/// Estimate `‖B^p‖_1` using Higham–Tisseur (2000) Algorithm 2.4.
///
/// This implements the iterative block 1-norm estimator used by
/// `scipy.sparse.linalg.onenormest` (and therefore `expm_multiply`).
/// It alternates forward (`B^p`) and transpose (`(B^T)^p`) matrix–vector
/// products to refine the estimate, starting from a robust all-ones column
/// that avoids the null-space failures of purely alternating probe vectors.
///
/// # Arguments
/// - `b`   — shifted operator `B = a·(A − μI)`
/// - `p`   — power to apply
/// - `ell` — block size / number of probe columns (matches scipy's `t`; 2 is
///   the default used by `expm_multiply`)
///
/// Returns a lower bound on `‖B^p‖_1`.
pub(crate) fn onenorm_matrix_power_nnm<V, Op>(b: &ShiftedOp<V, Op>, p: usize, ell: usize) -> V::Real
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    let n = b.dim();
    if n == 0 {
        return V::Real::default();
    }
    if p == 0 {
        // B^0 = I, ||I||_1 = 1
        return V::real_from_f64(1.0);
    }

    // scipy requires t < n; clamp so we never allocate more columns than rows.
    let t = ell.max(1).min(if n > 1 { n - 1 } else { 1 });

    const ITMAX: usize = 5;

    // -----------------------------------------------------------------------
    // Initialise X (column-major, n×t).
    //
    // Column 0: all-ones / n   — same as scipy's first column.
    //   For a matrix with all-positive entries this gives an exact estimate on
    //   the second iteration.  It also avoids null-space failures that affect
    //   purely alternating ±1 probes (e.g. 4-site periodic XX Hamiltonians).
    //
    // Columns 1..t: pseudo-random ±1 / n, seeded deterministically per column
    //   using an XOR-shift PRNG.  These widen the initial coverage without
    //   requiring the `rand` crate.
    // -----------------------------------------------------------------------
    let inv_n = V::from_real(V::real_from_f64(1.0 / n as f64));
    let mut x_mat = vec![V::default(); n * t];
    for slot in x_mat.iter_mut().take(n) {
        *slot = inv_n; // column 0
    }
    for col in 1..t {
        let mut rng: u64 = 0x9e3779b97f4a7c15_u64 ^ ((col as u64).wrapping_mul(0x6c62272e07bb0142));
        if rng == 0 {
            rng = 1;
        }
        for i in 0..n {
            let sign = xorshift_sign(&mut rng);
            x_mat[col * n + i] = V::from_real(V::real_from_f64(sign / n as f64));
        }
    }

    // -----------------------------------------------------------------------
    // Algorithm 2.4 — Higham & Tisseur (2000)
    // -----------------------------------------------------------------------
    let mut est = V::Real::default();
    let mut est_old = V::Real::default();
    let mut s_mat = vec![V::default(); n * t]; // S_{k-1}, starts as zeros
    let mut ind: Vec<usize> = Vec::new();
    let mut ind_hist: Vec<usize> = Vec::with_capacity(t * (ITMAX + 1));
    let mut ind_best: usize = 0;
    let mut work = vec![V::default(); n];

    // We iterate for k = 1, 2, …, ITMAX+1.  The hard limit is hit when
    // k > ITMAX (scipy: `if k > itmax: break`), which lets iteration ITMAX+1
    // run its forward pass and termination check (1) before exiting.
    for k in 1..=(ITMAX + 1) {
        // --- (A) Forward pass: Y = B^p X ---
        let y_mat = apply_block(b, p, &x_mat, n, t, &mut work, false);

        // --- Column 1-norms; pick the best column ---
        let mags = col_onenorms::<V>(&y_mat, n, t);
        let best_j = argmax(&mags);
        est = mags[best_j];

        // Track which unit-vector index gave the best estimate (k ≥ 2 only,
        // since `ind` is populated at the end of k=1).
        if k >= 2 && best_j < ind.len() {
            ind_best = ind[best_j];
        }

        // --- Termination (1): estimate did not improve ---
        if k >= 2 && est <= est_old {
            est = est_old;
            break;
        }
        est_old = est;

        // --- Build sign matrix S = sign_roundup(Y); save S_old ---
        // Hard limit check comes first (scipy: `S_old = S; if k > itmax: break;
        // S = sign_round_up(Y)`), so we don't compute the new S on the final
        // iteration.
        let s_old = s_mat;
        if k > ITMAX {
            break;
        }
        s_mat = sign_roundup_mat::<V>(&y_mat);

        // --- Termination (2): S has converged (every col parallel to S_old) ---
        if every_col_parallel::<V>(&s_mat, &s_old, n, t, t) {
            break;
        }

        // --- (B) Transpose pass: Z = (B^T)^p S ---
        let z_mat = apply_block(b, p, &s_mat, n, t, &mut work, true);

        // --- h[i] = max_col |z_{i,col}| ---
        let h = row_max_abs::<V>(&z_mat, n, t);

        // --- Termination (4): transpose pass confirms ind_best is optimal ---
        if k >= 2 {
            let max_h = h
                .iter()
                .copied()
                .fold(V::Real::default(), |a, b| if b > a { b } else { a });
            if max_h == h[ind_best] {
                break;
            }
        }

        // --- Sort rows by h descending; keep top (t + |ind_hist|) ---
        let take = (t + ind_hist.len()).min(n);
        let mut ind_sorted: Vec<usize> = (0..n).collect();
        ind_sorted.sort_unstable_by(|&a, &b| {
            h[b].partial_cmp(&h[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let ind_all = &ind_sorted[..take];

        // --- Termination (5): top-t candidates all already visited ---
        if t > 1 {
            let top_t = &ind_all[..t.min(ind_all.len())];
            if top_t.iter().all(|i| ind_hist.contains(i)) {
                break;
            }
        }

        // --- Reorder: unvisited first (preserving h-order), visited last ---
        let (unvisited, visited): (Vec<usize>, Vec<usize>) =
            ind_all.iter().partition(|&&i| !ind_hist.contains(&i));
        ind = unvisited.into_iter().chain(visited).collect();

        // --- X[:, j] = e_{ind[j]}: probe the most promising unit vectors ---
        for col in 0..t {
            let xc = &mut x_mat[col * n..(col + 1) * n];
            xc.iter_mut().for_each(|v| *v = V::default());
            xc[ind[col]] = V::from_real(V::real_from_f64(1.0));
        }

        // --- Record visited indices ---
        for &i in ind.iter().take(t) {
            if !ind_hist.contains(&i) {
                ind_hist.push(i);
            }
        }
    }

    est
}
