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

/// Reject a non-finite energy shift at construction.
///
/// A NaN shift makes every Boltzmann weight NaN; an infinite one makes them
/// all 0 or inf. Either way the failure surfaces far from its cause, so catch
/// it where the user supplied it.
fn validate_e_shift(e_shift: f64) -> PyResult<()> {
    if !e_shift.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "e_shift must be finite, got {e_shift}"
        )));
    }
    Ok(())
}

/// Build the shared "Boltzmann weight left the representable range" error.
///
/// `e^{-beta (E_n - e_shift)}` overflows once `beta * (e_shift - E_n)` exceeds
/// about 709 and underflows to zero once `beta * (E_n - e_shift)` does. Both
/// ends are failures the caller cannot detect afterwards:
///
/// - overflow gives `inf`, and `sum(oz)/sum(z)` comes out `NaN`;
/// - underflow gives exactly `0.0`, and the same ratio is `0/0`.
fn boltzmann_range_err(what: &str, detail: &str, beta: f64, e_shift: f64) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{what} at beta = {beta} with e_shift = {e_shift}; {detail} \
         (exp(-beta*(E - e_shift)) leaves the representable range once \
         beta*|E - e_shift| exceeds ~709)"
    ))
}

/// Check the partition function landed in the representable range.
///
/// Note the directions, which are opposite and easy to state backwards:
/// overflow means `e_shift` sits too far **above** the spectrum and must be
/// *lowered* toward `E_0`; underflow means it sits too far **below** and must
/// be *raised*. With the default `e_shift = 0` and a negative ground state,
/// the fix for overflow is to move `e_shift` down to `E_0` — not up.
fn check_partition_in_range(z_r: f64, beta: f64, e_shift: f64) -> PyResult<()> {
    if !z_r.is_finite() {
        return Err(boltzmann_range_err(
            &format!("partition function overflowed (z_r = {z_r})"),
            "e_shift sits too far above the spectrum; lower it towards the \
             ground-state energy when constructing the estimator",
            beta,
            e_shift,
        ));
    }
    if z_r == 0.0 {
        return Err(boltzmann_range_err(
            "partition function underflowed to zero (z_r = 0)",
            "e_shift sits too far below the spectrum, so every Boltzmann \
             weight rounded to zero; raise it towards the ground-state energy. \
             Lower is not safer than higher — set e_shift near E_0, not below \
             it, so a variational lower bound is the wrong thing to reach for",
            beta,
            e_shift,
        ));
    }
    Ok(())
}

/// Check a full `(z_r, oz_r)` sample.
///
/// `z_r` is a sum of non-negative terms so it cannot reach a finite value
/// through cancellation, but `oz_r` can be non-finite while `z_r` is not (a
/// large-norm observable), so check it too.
fn check_sample_in_range(z_r: f64, oz_r: C64, beta: f64, e_shift: f64) -> PyResult<()> {
    check_partition_in_range(z_r, beta, e_shift)?;
    if !oz_r.re.is_finite() || !oz_r.im.is_finite() {
        return Err(boltzmann_range_err(
            &format!("observable contribution is not finite (oz_r = {oz_r})"),
            "the observable's matrix elements overflowed once weighted",
            beta,
            e_shift,
        ));
    }
    Ok(())
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
        let result = py.detach(move || {
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
    e_shift: f64,
}

#[pymethods]
impl PyFTLM {
    /// Args:
    ///     hamiltonian: The system Hamiltonian.
    ///     e_shift:     Energy shift subtracted from every Ritz value before
    ///                  exponentiating (default ``0.0``).  See the ``e_shift``
    ///                  property for why you almost certainly want to set it.
    #[new]
    #[pyo3(signature = (hamiltonian, e_shift = 0.0))]
    fn new(hamiltonian: &PyHamiltonian, e_shift: f64) -> PyResult<Self> {
        validate_e_shift(e_shift)?;
        Ok(PyFTLM {
            inner: Arc::clone(&hamiltonian.inner),
            e_shift,
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Energy shift used in the Boltzmann weights ``e^{-beta (E_n - e_shift)}``.
    ///
    /// Fixed at construction rather than passed per ``sample()`` call, and
    /// deliberately so: the shift rescales every ``z_r`` and ``oz_r`` by
    /// ``e^{beta*e_shift}``, which cancels in ``sum(oz_r)/sum(z_r)`` **only if
    /// every sample used the same shift**.  Letting it vary per call (for
    /// instance, per-sample auto-shifting by that sample's own lowest Ritz
    /// value) would silently bias the estimate instead of overflowing, which
    /// is strictly worse.
    ///
    /// Set it to an estimate of the ground-state energy.  With the default of
    /// ``0.0``, ``e^{-beta E_n}`` overflows to ``inf`` once ``beta*|E_min|``
    /// exceeds about 709: a 20-site Heisenberg chain has ``E_0 ~ -35``, so any
    /// ``beta >~ 20`` returns ``inf``/``NaN``.  A cheap way to get one is a
    /// short ``EigSolver`` run; it does not need to be tight, only close.
    #[getter]
    fn e_shift(&self) -> f64 {
        self.e_shift
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
    ///     stored:     If ``True`` (default), store all Lanczos vectors for
    ///                 O(k × dim) memory but only one Lanczos build. If
    ///                 ``False``, use O(k + dim) memory by replaying the
    ///                 recurrence at the cost of extra matvecs.
    ///
    /// Returns:
    ///     ``(z_r, oz_r)`` where ``z_r`` is the partition function contribution
    ///     and ``oz_r`` is the ``⟨O⟩ · Z`` contribution (complex).
    #[pyo3(signature = (v0, k, observable, beta, time = 0.0, stored = true))]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        v0: &Bound<'_, PyArray1<Complex64>>,
        k: usize,
        observable: &PyHamiltonian,
        beta: f64,
        time: f64,
        stored: bool,
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
        let e_shift = self.e_shift;

        let result = py.detach(move || -> Result<(f64, C64), QuSpinError> {
            if stored {
                // Stored path: keep all basis vectors, single Lanczos build
                let basis = LanczosBasis::build(&mut make_matvec(&h_inner, time), &v0_vec, k)?;
                let eig = eig::solve_tridiagonal(basis.alpha(), basis.beta());

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

                let z_r = ftlm::ftlm_partition(&eig, beta, e_shift);
                let oz_r = ftlm::ftlm_observable(&eig, &obs_elements, beta, e_shift);
                Ok((z_r, oz_r))
            } else {
                // Replay path: O(k + dim) memory, extra matvecs
                let iter_basis =
                    LanczosBasisIter::build(&mut make_matvec(&h_inner, time), &v0_vec, k)?;
                let eig = eig::solve_tridiagonal(iter_basis.alpha(), iter_basis.beta());

                // Compute O|q_0⟩ once (q_0 is the normalized v0)
                let norm0 = v0_vec.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                let q0: Vec<C64> = v0_vec.iter().map(|&c| c / norm0).collect();
                let mut o_q0 = vec![C64::default(); n];
                o_inner.dot(true, time, &q0, &mut o_q0)?;

                // Replay recurrence to compute obs_elements on the fly
                let mut obs_elements = Vec::with_capacity(iter_basis.k());
                iter_basis.for_each(&mut make_matvec(&h_inner, time), |_j, q_j| {
                    let elem: C64 = q_j.iter().zip(o_q0.iter()).map(|(a, b)| a.conj() * b).sum();
                    obs_elements.push(elem);
                })?;

                let z_r = ftlm::ftlm_partition(&eig, beta, e_shift);
                let oz_r = ftlm::ftlm_observable(&eig, &obs_elements, beta, e_shift);
                Ok((z_r, oz_r))
            }
        });

        let (z_r, oz_r) = result.map_err(Error::from)?;
        check_sample_in_range(z_r, oz_r, beta, e_shift)?;
        Ok((z_r, Complex64::new(oz_r.re, oz_r.im)))
    }

    fn __repr__(&self) -> String {
        format!("FTLM(dim={}, e_shift={})", self.inner.dim(), self.e_shift)
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
    e_shift: f64,
}

#[pymethods]
impl PyLTLM {
    /// Args:
    ///     hamiltonian: The system Hamiltonian.
    ///     e_shift:     Energy shift subtracted from every Ritz value before
    ///                  exponentiating (default ``0.0``).  See ``FTLM.e_shift``.
    #[new]
    #[pyo3(signature = (hamiltonian, e_shift = 0.0))]
    fn new(hamiltonian: &PyHamiltonian, e_shift: f64) -> PyResult<Self> {
        validate_e_shift(e_shift)?;
        Ok(PyLTLM {
            inner: Arc::clone(&hamiltonian.inner),
            e_shift,
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Energy shift used in the Boltzmann weights; see ``FTLM.e_shift``.
    ///
    /// ``ltlm_coeffs`` exponentiates the half exponent
    /// ``-beta (E_n - e_shift) / 2``, but ``sample`` still computes ``z_r``
    /// with the full-exponent FTLM partition, so this class overflows at the
    /// same ``beta`` FTLM does — not at twice it.
    #[getter]
    fn e_shift(&self) -> f64 {
        self.e_shift
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
    ///     stored:     If ``True`` (default), store all Lanczos vectors for
    ///                 O(k × dim) memory but only one Lanczos build. If
    ///                 ``False``, use O(k + dim) memory by replaying the
    ///                 recurrence at the cost of extra matvecs.
    ///
    /// Returns:
    ///     ``(z_r, oz_r)`` where ``z_r`` is the partition function contribution
    ///     and ``oz_r`` is ``⟨φ|O|φ⟩`` (complex).
    #[pyo3(signature = (v0, k, observable, beta, time = 0.0, stored = true))]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        v0: &Bound<'_, PyArray1<Complex64>>,
        k: usize,
        observable: &PyHamiltonian,
        beta: f64,
        time: f64,
        stored: bool,
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
        let e_shift = self.e_shift;

        let result = py.detach(move || -> Result<(f64, C64), QuSpinError> {
            if stored {
                // Stored path: keep all basis vectors
                let basis = LanczosBasis::build(&mut make_matvec(&h_inner, time), &v0_vec, k)?;
                let eig = eig::solve_tridiagonal(basis.alpha(), basis.beta());
                let z_r = ftlm::ftlm_partition(&eig, beta, e_shift);

                let coeffs = ltlm::ltlm_coeffs(&eig, beta, e_shift);
                let mut phi = vec![C64::default(); n];
                basis.lin_comb(&coeffs, &mut phi)?;

                let mut o_phi = vec![C64::default(); n];
                o_inner.dot(true, time, &phi, &mut o_phi)?;

                let oz_r: C64 = phi
                    .iter()
                    .zip(o_phi.iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                Ok((z_r, oz_r))
            } else {
                // Replay path: O(k + dim) memory, extra matvecs
                let iter_basis =
                    LanczosBasisIter::build(&mut make_matvec(&h_inner, time), &v0_vec, k)?;
                let eig = eig::solve_tridiagonal(iter_basis.alpha(), iter_basis.beta());
                let z_r = ftlm::ftlm_partition(&eig, beta, e_shift);

                // Compute |φ⟩ = e^{-βH/2}|r⟩ via replay
                let coeffs = ltlm::ltlm_coeffs(&eig, beta, e_shift);
                let mut phi = vec![C64::default(); n];
                iter_basis.lin_comb(&mut make_matvec(&h_inner, time), &coeffs, &mut phi)?;

                let mut o_phi = vec![C64::default(); n];
                o_inner.dot(true, time, &phi, &mut o_phi)?;

                let oz_r: C64 = phi
                    .iter()
                    .zip(o_phi.iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                Ok((z_r, oz_r))
            }
        });

        let (z_r, oz_r) = result.map_err(Error::from)?;
        check_sample_in_range(z_r, oz_r, beta, e_shift)?;
        Ok((z_r, Complex64::new(oz_r.re, oz_r.im)))
    }

    fn __repr__(&self) -> String {
        format!("LTLM(dim={}, e_shift={})", self.inner.dim(), self.e_shift)
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
    e_shift: f64,
}

#[pymethods]
impl PyFTLMDynamic {
    /// Args:
    ///     hamiltonian: The system Hamiltonian.
    ///     e_shift:     Energy shift subtracted from every Ritz value before
    ///                  exponentiating (default ``0.0``).  See ``FTLM.e_shift``.
    #[new]
    #[pyo3(signature = (hamiltonian, e_shift = 0.0))]
    fn new(hamiltonian: &PyHamiltonian, e_shift: f64) -> PyResult<Self> {
        validate_e_shift(e_shift)?;
        Ok(PyFTLMDynamic {
            inner: Arc::clone(&hamiltonian.inner),
            e_shift,
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Energy shift used in the Boltzmann weights; see ``FTLM.e_shift``.
    ///
    /// Applied to the Boltzmann weight only — the resolvent pole ``omega +
    /// E_n`` is left alone, so the frequency axis does not move.  The whole
    /// returned array is scaled by ``exp(beta*e_shift)``.
    ///
    /// **Normalization caveat.**  ``sample`` returns only the unnormalized
    /// ``S_r(omega)``; the partition function that divides it comes from a
    /// separate ``FTLM`` object.  That is the one place the constructor-level
    /// shift cannot enforce consistency on its own — if the two objects are
    /// built with different ``e_shift`` values the spectral function is off by
    /// a silent factor ``exp(beta*(s_dyn - s_ftlm))``.  Build both from the
    /// same value:
    ///
    /// ```python
    /// e0 = ...                      # one estimate, used for both
    /// dyn = FTLMDynamic(ham, e_shift=e0)
    /// part = FTLM(ham, e_shift=e0)
    /// ```
    #[getter]
    fn e_shift(&self) -> f64 {
        self.e_shift
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
        let e_shift = self.e_shift;

        let result = py.detach(move || -> Result<(Vec<f64>, f64), QuSpinError> {
            // Left Lanczos: build basis from v0 using H
            let left_basis = LanczosBasisIter::build(&mut make_matvec(&h_inner, time), &v0_vec, k)?;
            let left_eig = eig::solve_tridiagonal(left_basis.alpha(), left_basis.beta());

            // The Boltzmann weights that scale the whole spectral function are
            // exactly the terms of this partition, so guard it the way the
            // FTLM/LTLM samples guard theirs. Checking the *output* array
            // instead would miss underflow: an e_shift below the left Ritz
            // values zeroes every weight, and the resulting all-zero array is
            // indistinguishable from a legitimately empty frequency window.
            // Computed before the zero-operator early return below, so a
            // degenerate partition is reported even when A|v0⟩ = 0.
            let z_left = ftlm::ftlm_partition(&left_eig, beta, e_shift);

            // Compute A|v0⟩ (using normalized v0 from the left basis)
            let norm0 = v0_vec.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            let v0_normed: Vec<C64> = v0_vec.iter().map(|&c| c / norm0).collect();
            let mut a_v0 = vec![C64::default(); n];
            a_inner.dot(true, time, &v0_normed, &mut a_v0)?;

            let right_norm_sq: f64 = a_v0.iter().map(|c| c.norm_sqr()).sum();

            if right_norm_sq < f64::EPSILON {
                // A|v0⟩ = 0, no spectral weight. This is a genuine physical
                // zero, not a numerical one.
                return Ok((vec![0.0; omegas_vec.len()], z_left));
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
                e_shift,
            );

            Ok((spectral, z_left))
        });

        let (spectral, z_left) = result.map_err(Error::from)?;
        // Catches both ends, including the underflow the output array cannot
        // show (an all-zero spectral function is also what an empty frequency
        // window legitimately produces).
        check_partition_in_range(z_left, beta, e_shift)?;
        // A finite, non-zero partition still leaves the operator norm free, so
        // an overflow can enter through `right_norm_sq` — the analogue of the
        // `oz_r` check on the other two estimators.
        if let Some(bad) = spectral.iter().find(|s| !s.is_finite()) {
            return Err(boltzmann_range_err(
                &format!("spectral function is not finite (S = {bad})"),
                "the operator's matrix elements overflowed once weighted",
                beta,
                e_shift,
            ));
        }
        Ok(spectral.to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "FTLMDynamic(dim={}, e_shift={})",
            self.inner.dim(),
            self.e_shift
        )
    }
}
