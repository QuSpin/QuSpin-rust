use super::basis::LanczosBasis;
use nalgebra::DMatrix;
use num_complex::Complex;
use quspin_types::QuSpinError;

type C64 = Complex<f64>;

// ---------------------------------------------------------------------------
// TridiagEigen — eigendecomposition of a tridiagonal matrix
// ---------------------------------------------------------------------------

/// Eigendecomposition of a real symmetric tridiagonal matrix.
///
/// Shared by `lanczos_eig`, FTLM, and LTLM.
pub struct TridiagEigen {
    /// Eigenvalues (length k).
    pub eigenvalues: Vec<f64>,
    /// Eigenvector matrix, column-major: element (i, j) is at `vecs[i + j * k]`.
    /// Column j is the eigenvector for `eigenvalues[j]`.
    pub vecs: Vec<f64>,
    /// Dimension of the tridiagonal matrix.
    pub k: usize,
}

impl TridiagEigen {
    /// Element (i, j) of the eigenvector matrix.
    pub fn vec_element(&self, i: usize, j: usize) -> f64 {
        self.vecs[i + j * self.k]
    }
}

/// Solve the eigenvalue problem for a real symmetric tridiagonal matrix
/// with diagonal `alpha` and off-diagonal `beta`.
pub fn solve_tridiagonal(alpha: &[f64], beta: &[f64]) -> TridiagEigen {
    let k = alpha.len();
    debug_assert_eq!(beta.len() + 1, k);

    let mut t = DMatrix::<f64>::zeros(k, k);
    for j in 0..k {
        t[(j, j)] = alpha[j];
    }
    for j in 0..beta.len() {
        t[(j, j + 1)] = beta[j];
        t[(j + 1, j)] = beta[j];
    }

    let eigen = t.symmetric_eigen();

    // Copy into flat column-major layout
    let mut vecs = vec![0.0_f64; k * k];
    for col in 0..k {
        for row in 0..k {
            vecs[row + col * k] = eigen.eigenvectors[(row, col)];
        }
    }

    TridiagEigen {
        eigenvalues: eigen.eigenvalues.as_slice().to_vec(),
        vecs,
        k,
    }
}

// ---------------------------------------------------------------------------
// Which — eigenvalue selection
// ---------------------------------------------------------------------------

/// Which eigenvalues to target.
#[derive(Debug, Clone, Copy)]
pub enum Which {
    SmallestAlgebraic,
    LargestAlgebraic,
    SmallestMagnitude,
}

// ---------------------------------------------------------------------------
// EigResult
// ---------------------------------------------------------------------------

/// Result of a Lanczos eigenvalue computation.
pub struct EigResult {
    /// Real eigenvalues (sorted according to the `Which` selector).
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors in the full Hilbert space,
    /// stored flat in row-major order: eigvec `i` is at `[i*dim..(i+1)*dim]`.
    pub eigenvectors: Vec<C64>,
    /// Residual norms `‖H·x - λ·x‖` for each Ritz pair.
    pub residuals: Vec<f64>,
    /// Dimension of the full Hilbert space.
    dim: usize,
}

impl EigResult {
    /// Number of converged eigenpairs returned.
    pub fn n_eig(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Dimension of the full Hilbert space.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Access the i-th eigenvector.
    pub fn eigenvector(&self, i: usize) -> &[C64] {
        &self.eigenvectors[i * self.dim..(i + 1) * self.dim]
    }
}

// ---------------------------------------------------------------------------
// lanczos_eig
// ---------------------------------------------------------------------------

/// Compute eigenvalues and eigenvectors using the Lanczos algorithm.
///
/// # Arguments
/// - `matvec` — applies `H|v⟩`: `matvec(input, output)` computes `output = H * input`
/// - `v0` — initial vector (will be normalized internally)
/// - `k_krylov` — Krylov subspace dimension
/// - `k_wanted` — number of eigenpairs to return
/// - `which` — eigenvalue selection criterion
/// - `tol` — convergence tolerance on residual norms; only eigenpairs with
///   residual `≤ tol` are returned (pass `f64::INFINITY` to disable)
pub fn lanczos_eig(
    matvec: &mut impl FnMut(&[C64], &mut [C64]) -> Result<(), QuSpinError>,
    v0: &[C64],
    k_krylov: usize,
    k_wanted: usize,
    which: Which,
    tol: f64,
) -> Result<EigResult, QuSpinError> {
    if k_wanted == 0 {
        return Err(QuSpinError::ValueError(
            "k_wanted must be at least 1".to_string(),
        ));
    }
    if k_krylov < k_wanted {
        return Err(QuSpinError::ValueError(format!(
            "k_krylov ({k_krylov}) must be >= k_wanted ({k_wanted})"
        )));
    }
    if !tol.is_finite() && tol != f64::INFINITY {
        return Err(QuSpinError::ValueError(format!(
            "tol must be finite or INFINITY; got {tol}"
        )));
    }

    let dim = v0.len();

    // Step 1: Build Lanczos basis
    let basis = LanczosBasis::build(matvec, v0, k_krylov)?;
    let k = basis.k();
    let k_wanted = k_wanted.min(k);

    // Step 2: Diagonalize the tridiagonal matrix
    let eig = solve_tridiagonal(basis.alpha(), basis.beta());

    // Step 3: Sort indices by the requested criterion
    let mut indices: Vec<usize> = (0..k).collect();
    match which {
        Which::SmallestAlgebraic => {
            indices.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
        }
        Which::LargestAlgebraic => {
            indices.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
        }
        Which::SmallestMagnitude => {
            indices.sort_by(|&a, &b| {
                eig.eigenvalues[a]
                    .abs()
                    .partial_cmp(&eig.eigenvalues[b].abs())
                    .unwrap()
            });
        }
    }
    indices.truncate(k_wanted);

    // Step 4: Map eigenvectors back to full space and compute residuals
    let mut eigenvalues = Vec::with_capacity(k_wanted);
    let mut eigenvectors = Vec::with_capacity(k_wanted * dim);
    let mut residuals = Vec::with_capacity(k_wanted);

    let mut full_vec = vec![C64::default(); dim];
    let mut h_full_vec = vec![C64::default(); dim];

    for &idx in &indices {
        let lam = eig.eigenvalues[idx];
        eigenvalues.push(lam);

        // Convert Krylov-space eigenvector to complex coefficients
        let krylov_coeffs: Vec<C64> = (0..k)
            .map(|j| C64::new(eig.vec_element(j, idx), 0.0))
            .collect();

        // Project back to full space: x = Q * y
        basis.lin_comb(&krylov_coeffs, &mut full_vec)?;
        eigenvectors.extend_from_slice(&full_vec);

        // Residual: ‖H·x - λ·x‖
        matvec(&full_vec, &mut h_full_vec)?;
        let residual: f64 = h_full_vec
            .iter()
            .zip(full_vec.iter())
            .map(|(hx, x)| (hx - C64::new(lam, 0.0) * x).norm_sqr())
            .sum::<f64>()
            .sqrt();

        // Filter by tolerance
        if residual > tol {
            // Remove the eigenvector we just appended
            eigenvectors.truncate(eigenvectors.len() - dim);
            eigenvalues.pop();
            continue;
        }
        residuals.push(residual);
    }

    Ok(EigResult {
        eigenvalues,
        eigenvectors,
        residuals,
        dim,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 2×2 Pauli X: eigenvalues ±1.
    fn pauli_x(input: &[C64], output: &mut [C64]) -> Result<(), QuSpinError> {
        output[0] = input[1];
        output[1] = input[0];
        Ok(())
    }

    /// 4×4 XX + ZZ Hamiltonian (2-site chain).
    /// Eigenvalues: -2, 0, 0, 2.
    fn xx_zz_2site(input: &[C64], output: &mut [C64]) -> Result<(), QuSpinError> {
        let one = C64::new(1.0, 0.0);
        let neg = C64::new(-1.0, 0.0);
        output[0] = one * input[0] + one * input[3];
        output[1] = neg * input[1] + one * input[2];
        output[2] = one * input[1] + neg * input[2];
        output[3] = one * input[0] + one * input[3];
        Ok(())
    }

    // -- solve_tridiagonal tests --

    #[test]
    fn tridiag_2x2_eigenvalues() {
        // T = [[0, 1], [1, 0]] → eigenvalues ±1
        let eig = solve_tridiagonal(&[0.0, 0.0], &[1.0]);
        let mut vals: Vec<f64> = eig.eigenvalues.clone();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - (-1.0)).abs() < 1e-12);
        assert!((vals[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tridiag_eigenvectors_orthonormal() {
        let eig = solve_tridiagonal(&[1.0, -1.0, 0.5], &[0.5, 0.3]);
        for i in 0..eig.k {
            for j in 0..eig.k {
                let dot: f64 = (0..eig.k)
                    .map(|r| eig.vec_element(r, i) * eig.vec_element(r, j))
                    .sum();
                if i == j {
                    assert!((dot - 1.0).abs() < 1e-12, "⟨v{i}|v{j}⟩ = {dot}");
                } else {
                    assert!(dot.abs() < 1e-12, "⟨v{i}|v{j}⟩ = {dot}");
                }
            }
        }
    }

    // -- lanczos_eig tests --

    #[test]
    fn pauli_x_ground_state() {
        let v0 = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let result = lanczos_eig(&mut pauli_x, &v0, 2, 1, Which::SmallestAlgebraic, 1e-12).unwrap();

        assert_eq!(result.n_eig(), 1);
        assert!(
            (result.eigenvalues[0] - (-1.0)).abs() < 1e-10,
            "ground state energy = {}, expected -1",
            result.eigenvalues[0],
        );
        assert!(
            result.residuals[0] < 1e-10,
            "residual = {}",
            result.residuals[0],
        );
    }

    #[test]
    fn pauli_x_both_eigenvalues() {
        let v0 = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let result = lanczos_eig(&mut pauli_x, &v0, 2, 2, Which::SmallestAlgebraic, 1e-12).unwrap();

        assert_eq!(result.n_eig(), 2);
        assert!(
            (result.eigenvalues[0] - (-1.0)).abs() < 1e-10,
            "e0 = {}",
            result.eigenvalues[0],
        );
        assert!(
            (result.eigenvalues[1] - 1.0).abs() < 1e-10,
            "e1 = {}",
            result.eigenvalues[1],
        );
    }

    #[test]
    fn xx_zz_ground_state_energy() {
        let v0 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.5, 0.3),
            C64::new(-0.2, 0.1),
            C64::new(0.0, 0.7),
        ];
        let result =
            lanczos_eig(&mut xx_zz_2site, &v0, 4, 1, Which::SmallestAlgebraic, 1e-12).unwrap();

        assert!(
            (result.eigenvalues[0] - (-2.0)).abs() < 1e-10,
            "ground state energy = {}, expected -2",
            result.eigenvalues[0],
        );
        assert!(
            result.residuals[0] < 1e-10,
            "residual = {}",
            result.residuals[0],
        );
    }

    #[test]
    fn xx_zz_largest_3() {
        // XX+ZZ has eigenvalues {-2, 0, 0, 2} but minimal polynomial is
        // x³ - 4x (degree 3), so the Krylov subspace is 3-dimensional
        // and finds the distinct eigenvalues {-2, 0, 2}.
        let v0 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.5, 0.3),
            C64::new(-0.2, 0.1),
            C64::new(0.0, 0.7),
        ];
        let result =
            lanczos_eig(&mut xx_zz_2site, &v0, 4, 3, Which::LargestAlgebraic, 1e-12).unwrap();

        assert_eq!(result.n_eig(), 3);
        // Sorted descending: 2, 0, -2
        assert!(
            (result.eigenvalues[0] - 2.0).abs() < 1e-10,
            "e0 = {}",
            result.eigenvalues[0],
        );
        assert!(
            result.eigenvalues[1].abs() < 1e-10,
            "e1 = {}, expected 0",
            result.eigenvalues[1],
        );
        assert!(
            (result.eigenvalues[2] - (-2.0)).abs() < 1e-10,
            "e2 = {}, expected -2",
            result.eigenvalues[2],
        );
    }

    #[test]
    fn smallest_magnitude() {
        // Krylov subspace finds distinct eigenvalues {-2, 0, 2}.
        // Smallest magnitude: 0, then ±2.
        let v0 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.5, 0.3),
            C64::new(-0.2, 0.1),
            C64::new(0.0, 0.7),
        ];
        let result =
            lanczos_eig(&mut xx_zz_2site, &v0, 4, 2, Which::SmallestMagnitude, 1e-12).unwrap();

        assert_eq!(result.n_eig(), 2);
        assert!(
            result.eigenvalues[0].abs() < 1e-10,
            "e0 = {}, expected 0",
            result.eigenvalues[0],
        );
        assert!(
            (result.eigenvalues[1].abs() - 2.0).abs() < 1e-10,
            "e1 = {}, expected ±2",
            result.eigenvalues[1],
        );
    }

    #[test]
    fn residuals_are_small() {
        let v0 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.5, 0.3),
            C64::new(-0.2, 0.1),
            C64::new(0.0, 0.7),
        ];
        let result =
            lanczos_eig(&mut xx_zz_2site, &v0, 4, 4, Which::SmallestAlgebraic, 1e-12).unwrap();

        for (i, &r) in result.residuals.iter().enumerate() {
            assert!(r < 1e-10, "residual[{i}] = {r}");
        }
    }

    #[test]
    fn eigenvectors_are_normalized() {
        let v0 = vec![
            C64::new(1.0, 0.0),
            C64::new(0.5, 0.3),
            C64::new(-0.2, 0.1),
            C64::new(0.0, 0.7),
        ];
        let result =
            lanczos_eig(&mut xx_zz_2site, &v0, 4, 4, Which::SmallestAlgebraic, 1e-12).unwrap();

        for i in 0..result.n_eig() {
            let v = result.eigenvector(i);
            let norm: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "eigenvector {i} norm = {norm}",);
        }
    }

    #[test]
    fn k_wanted_greater_than_k_krylov_errors() {
        let v0 = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        assert!(lanczos_eig(&mut pauli_x, &v0, 1, 2, Which::SmallestAlgebraic, 1e-12).is_err());
    }
}
