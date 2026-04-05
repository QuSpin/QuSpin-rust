use crate::error::Error;
use crate::hamiltonian::PyHamiltonian;
use num_complex::Complex;
use numpy::{Complex64, PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use quspin_core::error::QuSpinError;
use quspin_core::hamiltonian::HamiltonianInner;
use quspin_core::krylov::{
    basis::{LanczosBasis, LanczosBasisIter},
    eig::{self, Which},
    ftlm, ftlm_dynamic, ltlm,
};
use std::sync::Arc;

type C64 = Complex<f64>;

/// Convert a Python "which" string to the Rust enum.
fn parse_which(which: &str) -> PyResult<Which> {
    match which {
        "SA" | "sa" => Ok(Which::SmallestAlgebraic),
        "LA" | "la" => Ok(Which::LargestAlgebraic),
        "SM" | "sm" => Ok(Which::SmallestMagnitude),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown 'which' value: {which:?}; expected \"SA\", \"LA\", or \"SM\""
        ))),
    }
}

/// Extract a complex128 numpy array into a Vec<Complex<f64>>.
fn extract_c64_vec(arr: &Bound<'_, PyArray1<Complex64>>) -> Vec<C64> {
    unsafe {
        arr.as_array()
            .iter()
            .map(|c| C64::new(c.re, c.im))
            .collect()
    }
}

/// Build a matvec closure from a HamiltonianInner at a fixed time.
fn make_matvec(
    inner: &Arc<HamiltonianInner>,
    time: f64,
) -> impl FnMut(&[C64], &mut [C64]) -> Result<(), QuSpinError> + '_ {
    move |input: &[C64], output: &mut [C64]| inner.dot(true, time, input, output)
}

// ---------------------------------------------------------------------------
// EigSolver
// ---------------------------------------------------------------------------

/// Lanczos eigenvalue solver.
///
/// Wraps a `Hamiltonian` and computes eigenvalues/eigenvectors using the
/// Lanczos algorithm with full re-orthogonalization.
#[pyclass(name = "EigSolver", module = "quspin._rs")]
pub struct PyEigSolver {
    inner: Arc<HamiltonianInner>,
}

#[pymethods]
impl PyEigSolver {
    #[new]
    fn new(hamiltonian: &PyHamiltonian) -> Self {
        PyEigSolver {
            inner: Arc::clone(&hamiltonian.inner),
        }
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Compute eigenvalues and eigenvectors.
    ///
    /// Args:
    ///     v0:       Initial vector, shape ``(dim,)``.
    ///     k_krylov: Krylov subspace dimension.
    ///     k_wanted: Number of eigenpairs to return (default 1).
    ///     which:    ``"SA"`` (smallest algebraic), ``"LA"`` (largest),
    ///               or ``"SM"`` (smallest magnitude). Default ``"SA"``.
    ///     tol:      Convergence tolerance (default 1e-10).
    ///     time:     Evaluation time for time-dependent coefficients (default 0.0).
    ///
    /// Returns:
    ///     ``(eigenvalues, eigenvectors, residuals)`` where eigenvalues has
    ///     shape ``(k_wanted,)``, eigenvectors has shape ``(k_wanted, dim)``,
    ///     and residuals has shape ``(k_wanted,)``.
    #[pyo3(signature = (v0, k_krylov, k_wanted = 1, which = "SA", tol = 1e-10, time = 0.0))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        v0: &Bound<'py, PyArray1<Complex64>>,
        k_krylov: usize,
        k_wanted: usize,
        which: &str,
        tol: f64,
        time: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<Complex64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let n = self.inner.dim();
        let v0_vec = extract_c64_vec(v0);
        if v0_vec.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "v0 must have length {n}"
            )));
        }
        let which_enum = parse_which(which)?;

        let inner = Arc::clone(&self.inner);
        let result = py.allow_threads(move || {
            eig::lanczos_eig(
                &mut make_matvec(&inner, time),
                &v0_vec,
                k_krylov,
                k_wanted,
                which_enum,
                tol,
            )
        });
        let result = result.map_err(Error::from)?;

        let eigenvalues = result.eigenvalues.to_pyarray(py);
        let residuals = result.residuals.to_pyarray(py);

        let n_eig = result.n_eig();
        let dim = result.dim();
        let rows: Vec<Vec<Complex64>> = (0..n_eig)
            .map(|i| {
                result
                    .eigenvector(i)
                    .iter()
                    .map(|c| Complex64::new(c.re, c.im))
                    .collect()
            })
            .collect();
        let eigenvectors = PyArray2::from_vec2(py, &rows).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to create ({n_eig}, {dim}) array"
            ))
        })?;

        Ok((eigenvalues, eigenvectors, residuals))
    }

    fn __repr__(&self) -> String {
        format!("EigSolver(dim={})", self.inner.dim())
    }
}

// ---------------------------------------------------------------------------
// FTLM
// ---------------------------------------------------------------------------

/// Finite Temperature Lanczos Method.
///
/// Computes thermal expectation values using quantum typicality.
#[pyclass(name = "FTLM", module = "quspin._rs")]
pub struct PyFTLM {
    inner: Arc<HamiltonianInner>,
}

#[pymethods]
impl PyFTLM {
    #[new]
    fn new(hamiltonian: &PyHamiltonian) -> Self {
        PyFTLM {
            inner: Arc::clone(&hamiltonian.inner),
        }
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Compute a single FTLM sample.
    ///
    /// Builds a Lanczos basis from ``v0`` using the Hamiltonian, then
    /// computes the partition function contribution and the observable
    /// expectation value contribution for the given inverse temperature.
    ///
    /// Args:
    ///     v0:         Random starting vector, shape ``(dim,)``.
    ///     k:          Number of Lanczos steps.
    ///     observable: ``Hamiltonian`` representing the observable operator.
    ///     beta:       Inverse temperature.
    ///     time:       Evaluation time for time-dependent coefficients (default 0.0).
    ///
    /// Returns:
    ///     ``(z_r, oz_r)`` where ``z_r`` is the partition function contribution
    ///     and ``oz_r`` is the ``⟨O⟩ · Z`` contribution (complex).
    #[pyo3(signature = (v0, k, observable, beta, time = 0.0))]
    fn sample(
        &self,
        py: Python<'_>,
        v0: &Bound<'_, PyArray1<Complex64>>,
        k: usize,
        observable: &PyHamiltonian,
        beta: f64,
        time: f64,
    ) -> PyResult<(f64, Complex64)> {
        let n = self.inner.dim();
        let v0_vec = extract_c64_vec(v0);
        if v0_vec.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "v0 must have length {n}"
            )));
        }

        let h_inner = Arc::clone(&self.inner);
        let o_inner = Arc::clone(&observable.inner);

        let result = py.allow_threads(move || -> Result<(f64, C64), QuSpinError> {
            // Build Lanczos basis
            let basis = LanczosBasis::build(&mut make_matvec(&h_inner, time), &v0_vec, k)?;

            // Solve tridiagonal eigenproblem
            let eig = eig::solve_tridiagonal(basis.alpha(), basis.beta());

            // Compute observable matrix elements: o_j = ⟨q_j|O|r⟩
            // where r = q_0 (first basis vector, which is v0 normalized)
            let q0 = basis.q(0);
            let mut o_q0 = vec![C64::default(); n];
            o_inner.dot(true, time, q0, &mut o_q0)?;

            let obs_elements: Vec<C64> = (0..basis.k())
                .map(|j| {
                    basis
                        .q(j)
                        .iter()
                        .zip(o_q0.iter())
                        .map(|(a, b)| a.conj() * b)
                        .sum()
                })
                .collect();

            let z_r = ftlm::ftlm_partition(&eig, beta);
            let oz_r = ftlm::ftlm_observable(&eig, &obs_elements, beta);

            Ok((z_r, oz_r))
        });

        let (z_r, oz_r) = result.map_err(Error::from)?;
        Ok((z_r, Complex64::new(oz_r.re, oz_r.im)))
    }

    fn __repr__(&self) -> String {
        format!("FTLM(dim={})", self.inner.dim())
    }
}

// ---------------------------------------------------------------------------
// LTLM
// ---------------------------------------------------------------------------

/// Low Temperature Lanczos Method.
///
/// Uses ``e^{-βH/2}`` on both sides for lower estimator variance at low
/// temperature.
#[pyclass(name = "LTLM", module = "quspin._rs")]
pub struct PyLTLM {
    inner: Arc<HamiltonianInner>,
}

#[pymethods]
impl PyLTLM {
    #[new]
    fn new(hamiltonian: &PyHamiltonian) -> Self {
        PyLTLM {
            inner: Arc::clone(&hamiltonian.inner),
        }
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Compute a single LTLM sample.
    ///
    /// Builds a Lanczos basis from ``v0``, computes ``|φ⟩ = e^{-βH/2}|r⟩``
    /// via the Krylov projection, then evaluates ``⟨φ|O|φ⟩``.
    ///
    /// Args:
    ///     v0:         Random starting vector, shape ``(dim,)``.
    ///     k:          Number of Lanczos steps.
    ///     observable: ``Hamiltonian`` representing the observable operator.
    ///     beta:       Inverse temperature.
    ///     time:       Evaluation time for time-dependent coefficients (default 0.0).
    ///
    /// Returns:
    ///     ``(z_r, oz_r)`` where ``z_r`` is the partition function contribution
    ///     and ``oz_r`` is ``⟨φ|O|φ⟩`` (complex).
    #[pyo3(signature = (v0, k, observable, beta, time = 0.0))]
    fn sample(
        &self,
        py: Python<'_>,
        v0: &Bound<'_, PyArray1<Complex64>>,
        k: usize,
        observable: &PyHamiltonian,
        beta: f64,
        time: f64,
    ) -> PyResult<(f64, Complex64)> {
        let n = self.inner.dim();
        let v0_vec = extract_c64_vec(v0);
        if v0_vec.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "v0 must have length {n}"
            )));
        }

        let h_inner = Arc::clone(&self.inner);
        let o_inner = Arc::clone(&observable.inner);

        let result = py.allow_threads(move || -> Result<(f64, C64), QuSpinError> {
            // Build stored Lanczos basis
            let basis = LanczosBasis::build(&mut make_matvec(&h_inner, time), &v0_vec, k)?;

            // Solve tridiagonal eigenproblem
            let eig = eig::solve_tridiagonal(basis.alpha(), basis.beta());

            // Partition function (same as FTLM)
            let z_r = ftlm::ftlm_partition(&eig, beta);

            // Compute |φ⟩ = e^{-βH/2}|r⟩ via Krylov projection
            let coeffs = ltlm::ltlm_coeffs(&eig, beta);
            let mut phi = vec![C64::default(); n];
            basis.lin_comb(&coeffs, &mut phi)?;

            // Compute O|φ⟩
            let mut o_phi = vec![C64::default(); n];
            o_inner.dot(true, time, &phi, &mut o_phi)?;

            // ⟨φ|O|φ⟩
            let oz_r: C64 = phi
                .iter()
                .zip(o_phi.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();

            Ok((z_r, oz_r))
        });

        let (z_r, oz_r) = result.map_err(Error::from)?;
        Ok((z_r, Complex64::new(oz_r.re, oz_r.im)))
    }

    fn __repr__(&self) -> String {
        format!("LTLM(dim={})", self.inner.dim())
    }
}

// ---------------------------------------------------------------------------
// FTLMDynamic
// ---------------------------------------------------------------------------

/// FTLM dynamic correlations (spectral function).
///
/// Computes the spectral function ``S(ω)`` via two independent Lanczos runs.
#[pyclass(name = "FTLMDynamic", module = "quspin._rs")]
pub struct PyFTLMDynamic {
    inner: Arc<HamiltonianInner>,
}

#[pymethods]
impl PyFTLMDynamic {
    #[new]
    fn new(hamiltonian: &PyHamiltonian) -> Self {
        PyFTLMDynamic {
            inner: Arc::clone(&hamiltonian.inner),
        }
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Compute one FTLM dynamic sample for the spectral function.
    ///
    /// Performs two Lanczos runs: one from ``v0`` (left, for Boltzmann
    /// weights) and one from ``A|v0⟩`` (right, for the continued-fraction
    /// resolvent).
    ///
    /// Args:
    ///     v0:       Random starting vector, shape ``(dim,)``.
    ///     k:        Number of Lanczos steps for each run.
    ///     operator: ``Hamiltonian`` representing the operator ``A``.
    ///     beta:     Inverse temperature.
    ///     omegas:   Frequency grid, shape ``(n_omega,)``.
    ///     eta:      Lorentzian broadening parameter.
    ///     time:     Evaluation time for time-dependent coefficients (default 0.0).
    ///
    /// Returns:
    ///     Spectral function contribution ``S_r(ω)`` as a 1-D array of shape
    ///     ``(n_omega,)``.
    #[pyo3(signature = (v0, k, operator, beta, omegas, eta, time = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        v0: &Bound<'py, PyArray1<Complex64>>,
        k: usize,
        operator: &PyHamiltonian,
        beta: f64,
        omegas: &Bound<'py, PyArray1<f64>>,
        eta: f64,
        time: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n = self.inner.dim();
        let v0_vec = extract_c64_vec(v0);
        if v0_vec.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "v0 must have length {n}"
            )));
        }
        let omegas_vec: Vec<f64> = unsafe { omegas.as_array().to_vec() };

        let h_inner = Arc::clone(&self.inner);
        let a_inner = Arc::clone(&operator.inner);

        let result = py.allow_threads(move || -> Result<Vec<f64>, QuSpinError> {
            // Left Lanczos: build basis from v0 using H
            let left_basis = LanczosBasisIter::build(&mut make_matvec(&h_inner, time), &v0_vec, k)?;
            let left_eig = eig::solve_tridiagonal(left_basis.alpha(), left_basis.beta());

            // Compute A|v0⟩ (using normalized v0 from the left basis)
            let norm0 = v0_vec.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            let v0_normed: Vec<C64> = v0_vec.iter().map(|&c| c / norm0).collect();
            let mut a_v0 = vec![C64::default(); n];
            a_inner.dot(true, time, &v0_normed, &mut a_v0)?;

            let right_norm_sq: f64 = a_v0.iter().map(|c| c.norm_sqr()).sum();

            if right_norm_sq < f64::EPSILON {
                // A|v0⟩ = 0, no spectral weight
                return Ok(vec![0.0; omegas_vec.len()]);
            }

            // Right Lanczos: build basis from A|v0⟩ using H
            let right_basis = LanczosBasisIter::build(&mut make_matvec(&h_inner, time), &a_v0, k)?;

            let spectral = ftlm_dynamic::ftlm_dynamic_spectral(
                &left_eig,
                right_basis.alpha(),
                right_basis.beta(),
                right_norm_sq,
                beta,
                &omegas_vec,
                eta,
            );

            Ok(spectral)
        });

        let spectral = result.map_err(Error::from)?;
        Ok(spectral.to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("FTLMDynamic(dim={})", self.inner.dim())
    }
}
