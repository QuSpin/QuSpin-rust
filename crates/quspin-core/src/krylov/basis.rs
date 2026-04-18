use num_complex::Complex;
use quspin_types::QuSpinError;

type C64 = Complex<f64>;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// L2 norm of a complex vector.
fn l2_norm(v: &[C64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Inner product <a|b> = Σ conj(a_i) * b_i.
fn inner(a: &[C64], b: &[C64]) -> C64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai.conj() * bi).sum()
}

// ---------------------------------------------------------------------------
// LanczosBasis — stores all K basis vectors
// ---------------------------------------------------------------------------

/// Stored Lanczos basis with full re-orthogonalization.
///
/// Stores all K orthonormal basis vectors Q and the tridiagonal matrix
/// T = Q^H · H · Q with diagonal `alpha` and off-diagonal `beta`.
pub struct LanczosBasis {
    /// Orthonormal basis vectors, flat row-major: q[i] lives at `[i*dim..(i+1)*dim]`.
    q: Vec<C64>,
    /// Diagonal of the tridiagonal matrix T (length k_actual).
    alpha: Vec<f64>,
    /// Off-diagonal of T (length k_actual - 1).
    beta: Vec<f64>,
    /// Dimension of the full Hilbert space.
    dim: usize,
}

impl LanczosBasis {
    /// Build a K-step Lanczos basis with full re-orthogonalization via MGS.
    ///
    /// # Arguments
    /// - `matvec` — applies `H|v⟩` in-place: `matvec(input, output)` computes `output = H * input`
    /// - `v0` — initial vector (will be normalized)
    /// - `k` — number of Lanczos steps requested
    ///
    /// The actual basis may be smaller than `k` if an invariant subspace is found.
    pub fn build(
        matvec: &mut impl FnMut(&[C64], &mut [C64]) -> Result<(), QuSpinError>,
        v0: &[C64],
        k: usize,
    ) -> Result<Self, QuSpinError> {
        let dim = v0.len();
        if dim == 0 {
            return Err(QuSpinError::ValueError("v0 must be non-empty".to_string()));
        }
        if k == 0 {
            return Err(QuSpinError::ValueError("k must be at least 1".to_string()));
        }

        let k = k.min(dim);

        let mut q = Vec::with_capacity(k * dim);
        let mut alpha = Vec::with_capacity(k);
        let mut beta: Vec<f64> = Vec::with_capacity(k.saturating_sub(1));
        let mut w = vec![C64::default(); dim];

        // Normalize v0 and store as first basis vector
        let norm0 = l2_norm(v0);
        if norm0 < f64::EPSILON {
            return Err(QuSpinError::ValueError("v0 has zero norm".to_string()));
        }
        let inv_norm0 = 1.0 / norm0;
        q.extend(v0.iter().map(|&c| c * inv_norm0));

        for j in 0..k {
            let qj = &q[j * dim..(j + 1) * dim];

            // w = H * q_j
            matvec(qj, &mut w)?;

            // Three-term recurrence: subtract projections onto q_j and q_{j-1}
            let aj = inner(qj, &w).re; // Hermitian → real
            alpha.push(aj);

            for i in 0..dim {
                w[i] -= C64::new(aj, 0.0) * qj[i];
            }
            if j > 0 {
                let bprev = beta[j - 1];
                let qprev = &q[(j - 1) * dim..j * dim];
                for i in 0..dim {
                    w[i] -= C64::new(bprev, 0.0) * qprev[i];
                }
            }

            // Full re-orthogonalization via MGS (two passes for stability)
            for _pass in 0..2 {
                for l in 0..=j {
                    let ql = &q[l * dim..(l + 1) * dim];
                    let h = inner(ql, &w);
                    for i in 0..dim {
                        w[i] -= h * ql[i];
                    }
                }
            }

            // Compute norm for next basis vector
            let bj = l2_norm(&w);

            if j < k - 1 {
                if bj < 1e-14 * dim as f64 {
                    // Invariant subspace found — stop
                    break;
                }
                beta.push(bj);
                let inv_bj = 1.0 / bj;
                q.extend(w.iter().map(|&c| c * inv_bj));
            }
        }

        Ok(LanczosBasis {
            q,
            alpha,
            beta,
            dim,
        })
    }

    /// Number of Lanczos vectors actually computed.
    pub fn k(&self) -> usize {
        self.alpha.len()
    }

    /// Dimension of the full Hilbert space.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Diagonal of the tridiagonal matrix (length `k`).
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Off-diagonal of the tridiagonal matrix (length `k - 1`).
    pub fn beta(&self) -> &[f64] {
        &self.beta
    }

    /// Access the i-th basis vector.
    pub fn q(&self, i: usize) -> &[C64] {
        &self.q[i * self.dim..(i + 1) * self.dim]
    }

    /// Reconstruct a full-space vector from Krylov-space coefficients.
    ///
    /// Computes `result = Σ_j coeffs[j] * q_j` where `q_j` are the stored
    /// basis vectors.
    pub fn lin_comb(&self, coeffs: &[C64], result: &mut [C64]) -> Result<(), QuSpinError> {
        let k = self.k();
        if coeffs.len() != k {
            return Err(QuSpinError::ValueError(format!(
                "coeffs.len() = {} but basis has k = {}",
                coeffs.len(),
                k,
            )));
        }
        if result.len() != self.dim {
            return Err(QuSpinError::ValueError(format!(
                "result.len() = {} but dim = {}",
                result.len(),
                self.dim,
            )));
        }

        // Zero output
        result.iter_mut().for_each(|c| *c = C64::default());

        for (j, &cj) in coeffs.iter().enumerate() {
            let qj = self.q(j);
            for i in 0..self.dim {
                result[i] += cj * qj[i];
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LanczosBasisIter — generator, O(dim) memory
// ---------------------------------------------------------------------------

/// Lightweight Lanczos result that stores only the tridiagonal coefficients
/// and the initial vector — O(k + dim) after construction.
///
/// **Note:** The `build` method temporarily allocates O(k × dim) for full
/// re-orthogonalization during construction, then discards the basis vectors.
/// The persistent storage is O(k + dim) (alpha, beta, and v0).
///
/// Reconstructing vectors requires replaying the Lanczos recurrence with the
/// operator, so `lin_comb` takes the operator by reference.
pub struct LanczosBasisIter {
    /// Normalized initial vector.
    v0: Vec<C64>,
    /// Diagonal of the tridiagonal matrix (length k).
    alpha: Vec<f64>,
    /// Off-diagonal of the tridiagonal matrix (length k - 1).
    beta: Vec<f64>,
}

impl LanczosBasisIter {
    /// Build the tridiagonal decomposition.
    ///
    /// Same recurrence as [`LanczosBasis::build`] with full re-orthogonalization.
    /// Temporarily allocates O(k × dim) for the basis vectors during construction,
    /// then discards them — only `alpha`, `beta`, and `v0` are retained.
    pub fn build(
        matvec: &mut impl FnMut(&[C64], &mut [C64]) -> Result<(), QuSpinError>,
        v0: &[C64],
        k: usize,
    ) -> Result<Self, QuSpinError> {
        let dim = v0.len();
        if dim == 0 {
            return Err(QuSpinError::ValueError("v0 must be non-empty".to_string()));
        }
        if k == 0 {
            return Err(QuSpinError::ValueError("k must be at least 1".to_string()));
        }

        let k = k.min(dim);

        let mut alpha = Vec::with_capacity(k);
        let mut beta: Vec<f64> = Vec::with_capacity(k.saturating_sub(1));

        // For full re-orthogonalization we need to keep all basis vectors
        // during the build phase; they are discarded afterwards.
        let mut q = Vec::with_capacity(k * dim);
        let mut w = vec![C64::default(); dim];

        let norm0 = l2_norm(v0);
        if norm0 < f64::EPSILON {
            return Err(QuSpinError::ValueError("v0 has zero norm".to_string()));
        }
        let inv_norm0 = 1.0 / norm0;
        let v0_normed: Vec<C64> = v0.iter().map(|&c| c * inv_norm0).collect();
        q.extend_from_slice(&v0_normed);

        for j in 0..k {
            let qj = &q[j * dim..(j + 1) * dim];
            matvec(qj, &mut w)?;

            let aj = inner(qj, &w).re;
            alpha.push(aj);

            for i in 0..dim {
                w[i] -= C64::new(aj, 0.0) * qj[i];
            }
            if j > 0 {
                let bprev = beta[j - 1];
                let qprev = &q[(j - 1) * dim..j * dim];
                for i in 0..dim {
                    w[i] -= C64::new(bprev, 0.0) * qprev[i];
                }
            }

            // Full re-orthogonalization (two passes)
            for _pass in 0..2 {
                for l in 0..=j {
                    let ql = &q[l * dim..(l + 1) * dim];
                    let h = inner(ql, &w);
                    for i in 0..dim {
                        w[i] -= h * ql[i];
                    }
                }
            }

            let bj = l2_norm(&w);

            if j < k - 1 {
                if bj < 1e-14 * dim as f64 {
                    break;
                }
                beta.push(bj);
                let inv_bj = 1.0 / bj;
                q.extend(w.iter().map(|&c| c * inv_bj));
            }
        }

        Ok(LanczosBasisIter {
            v0: v0_normed,
            alpha,
            beta,
        })
    }

    /// Number of Lanczos steps computed.
    pub fn k(&self) -> usize {
        self.alpha.len()
    }

    /// Dimension of the full Hilbert space.
    pub fn dim(&self) -> usize {
        self.v0.len()
    }

    /// Diagonal of the tridiagonal matrix.
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Off-diagonal of the tridiagonal matrix.
    pub fn beta(&self) -> &[f64] {
        &self.beta
    }

    /// Replay the Lanczos recurrence, calling `f(j, q_j)` at each step.
    ///
    /// This re-derives the basis vectors using the operator without storing
    /// them all simultaneously.
    pub fn for_each(
        &self,
        matvec: &mut impl FnMut(&[C64], &mut [C64]) -> Result<(), QuSpinError>,
        mut f: impl FnMut(usize, &[C64]),
    ) -> Result<(), QuSpinError> {
        let dim = self.dim();
        let k = self.k();

        let mut v_prev = vec![C64::default(); dim];
        let mut v_curr = self.v0.clone();
        let mut w = vec![C64::default(); dim];

        for j in 0..k {
            f(j, &v_curr);

            if j < k - 1 {
                matvec(&v_curr, &mut w)?;

                let aj = self.alpha[j];
                for i in 0..dim {
                    w[i] -= C64::new(aj, 0.0) * v_curr[i];
                }
                if j > 0 {
                    let bprev = self.beta[j - 1];
                    for i in 0..dim {
                        w[i] -= C64::new(bprev, 0.0) * v_prev[i];
                    }
                }

                let bj = self.beta[j];
                let inv_bj = 1.0 / bj;

                // Shift: v_prev ← v_curr, v_curr ← w / beta_j
                std::mem::swap(&mut v_prev, &mut v_curr);
                for i in 0..dim {
                    v_curr[i] = w[i] * inv_bj;
                }
            }
        }

        Ok(())
    }

    /// Reconstruct a full-space vector from Krylov-space coefficients by
    /// replaying the recurrence.
    ///
    /// Computes `result = Σ_j coeffs[j] * q_j` where `q_j` are re-derived
    /// from the stored tridiagonal coefficients and the operator.
    pub fn lin_comb(
        &self,
        matvec: &mut impl FnMut(&[C64], &mut [C64]) -> Result<(), QuSpinError>,
        coeffs: &[C64],
        result: &mut [C64],
    ) -> Result<(), QuSpinError> {
        if coeffs.len() != self.k() {
            return Err(QuSpinError::ValueError(format!(
                "coeffs.len() = {} but basis has k = {}",
                coeffs.len(),
                self.k(),
            )));
        }
        if result.len() != self.dim() {
            return Err(QuSpinError::ValueError(format!(
                "result.len() = {} but dim = {}",
                result.len(),
                self.dim(),
            )));
        }

        result.iter_mut().for_each(|c| *c = C64::default());

        self.for_each(matvec, |j, qj| {
            let cj = coeffs[j];
            for i in 0..self.dim() {
                result[i] += cj * qj[i];
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 2×2 Pauli X matrix: H = [[0, 1], [1, 0]]
    fn pauli_x(input: &[C64], output: &mut [C64]) -> Result<(), QuSpinError> {
        output[0] = input[1];
        output[1] = input[0];
        Ok(())
    }

    /// 4×4 XX + ZZ Hamiltonian for 2-site chain.
    ///
    /// H = X⊗X + Z⊗Z in the {|00⟩, |01⟩, |10⟩, |11⟩} basis:
    /// ```text
    /// [[1, 0, 0, 1],
    ///  [0,-1, 1, 0],
    ///  [0, 1,-1, 0],
    ///  [1, 0, 0, 1]]
    /// ```
    /// Eigenvalues: -2, 0, 0, 2
    fn xx_zz_2site(input: &[C64], output: &mut [C64]) -> Result<(), QuSpinError> {
        let one = C64::new(1.0, 0.0);
        let neg = C64::new(-1.0, 0.0);
        output[0] = one * input[0] + one * input[3];
        output[1] = neg * input[1] + one * input[2];
        output[2] = one * input[1] + neg * input[2];
        output[3] = one * input[0] + one * input[3];
        Ok(())
    }

    #[test]
    fn basis_2x2_orthonormal() {
        let v0 = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let basis = LanczosBasis::build(&mut pauli_x, &v0, 2).unwrap();

        assert_eq!(basis.k(), 2);

        // Check orthonormality
        for i in 0..basis.k() {
            for j in 0..basis.k() {
                let dot = inner(basis.q(i), basis.q(j));
                if i == j {
                    assert!((dot.re - 1.0).abs() < 1e-12, "<q{i}|q{j}> = {dot}");
                } else {
                    assert!(dot.norm() < 1e-12, "<q{i}|q{j}> = {dot}");
                }
            }
        }
    }

    #[test]
    fn basis_2x2_eigenvalues() {
        let v0 = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let basis = LanczosBasis::build(&mut pauli_x, &v0, 2).unwrap();

        // For Pauli X, eigenvalues are ±1.
        // The tridiagonal matrix should reproduce these.
        assert_eq!(basis.alpha().len(), 2);
        assert_eq!(basis.beta().len(), 1);

        // T = [[alpha[0], beta[0]], [beta[0], alpha[1]]]
        // eigenvalues of T: alpha[0] ± beta[0] (if alpha[0] == alpha[1] == 0)
        let a0 = basis.alpha()[0];
        let a1 = basis.alpha()[1];
        let b0 = basis.beta()[0];
        let trace = a0 + a1;
        let det = a0 * a1 - b0 * b0;
        let disc = (trace * trace - 4.0 * det).sqrt();
        let e1 = (trace - disc) / 2.0;
        let e2 = (trace + disc) / 2.0;

        assert!((e1 - (-1.0)).abs() < 1e-12, "e1 = {e1}, expected -1");
        assert!((e2 - 1.0).abs() < 1e-12, "e2 = {e2}, expected 1");
    }

    #[test]
    fn basis_4x4_eigenvalues() {
        // Use a starting vector that spans the full space
        let v0 = vec![
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
        ];
        let basis = LanczosBasis::build(&mut xx_zz_2site, &v0, 4).unwrap();

        // XX+ZZ eigenvalues: {-2, 0, 0, 2}
        // Depending on starting vector, may not span all eigenspaces.
        // With v0 = [1,1,1,1], we should get at least the non-degenerate eigenvalues.
        assert!(basis.k() >= 2);
    }

    #[test]
    fn lin_comb_stored_and_iter_agree() {
        let v0 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.5, 0.3),
            C64::new(-0.2, 0.1),
            C64::new(0.0, 0.7),
        ];

        let stored = LanczosBasis::build(&mut xx_zz_2site, &v0, 4).unwrap();
        let iter = LanczosBasisIter::build(&mut xx_zz_2site, &v0, 4).unwrap();

        assert_eq!(stored.k(), iter.k());
        assert_eq!(stored.alpha().len(), iter.alpha().len());
        assert_eq!(stored.beta().len(), iter.beta().len());

        // Check tridiagonal coefficients agree
        for j in 0..stored.k() {
            assert!(
                (stored.alpha()[j] - iter.alpha()[j]).abs() < 1e-12,
                "alpha[{j}]: stored={}, iter={}",
                stored.alpha()[j],
                iter.alpha()[j],
            );
        }
        for j in 0..stored.beta().len() {
            assert!(
                (stored.beta()[j] - iter.beta()[j]).abs() < 1e-12,
                "beta[{j}]: stored={}, iter={}",
                stored.beta()[j],
                iter.beta()[j],
            );
        }

        // Test lin_comb with some arbitrary coefficients
        let k = stored.k();
        let coeffs: Vec<C64> = (0..k)
            .map(|j| C64::new(j as f64 * 0.3 + 0.1, -(j as f64) * 0.2))
            .collect();

        let mut result_stored = vec![C64::default(); stored.dim()];
        let mut result_iter = vec![C64::default(); iter.dim()];

        stored.lin_comb(&coeffs, &mut result_stored).unwrap();
        iter.lin_comb(&mut xx_zz_2site, &coeffs, &mut result_iter)
            .unwrap();

        for i in 0..stored.dim() {
            assert!(
                (result_stored[i] - result_iter[i]).norm() < 1e-10,
                "result[{i}]: stored={}, iter={}",
                result_stored[i],
                result_iter[i],
            );
        }
    }

    #[test]
    fn invariant_subspace_early_stop() {
        // Pauli X is 2×2, so requesting k=10 should give k=2
        let v0 = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let basis = LanczosBasis::build(&mut pauli_x, &v0, 10).unwrap();
        assert_eq!(basis.k(), 2);
    }

    #[test]
    fn zero_v0_errors() {
        let v0 = vec![C64::default(); 4];
        assert!(LanczosBasis::build(&mut xx_zz_2site, &v0, 4).is_err());
    }
}
