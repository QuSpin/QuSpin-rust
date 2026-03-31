use crate::hamiltonian::PyHamiltonian;
use nalgebra::DVector;
use num_complex::Complex;
use numpy::{Complex64, PyArrayMethods};
use numpy::{PyArray1, PyArray2, ToPyArray};
use ode_solvers::System;
use pyo3::prelude::*;
use quspin_core::qmatrix::QMatrixInner;

/// Return type of `PySchrodingerEq::integrate_dense`.
type DenseOutput<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<Complex64>>);

// ---------------------------------------------------------------------------
// Inline ODE system  (only lives during `integrate`, never stored)
// ---------------------------------------------------------------------------

struct SchrodingerSystem<'a, 'py> {
    py: Python<'py>,
    matrix: &'a QMatrixInner,
    coeff_fns: &'a [PyObject],
}

impl<'a, 'py> System<f64, DVector<f64>> for SchrodingerSystem<'a, 'py> {
    fn system(&self, t: f64, y: &DVector<f64>, dy: &mut DVector<f64>) {
        // Evaluate Python coefficient functions at time t
        let coeffs = crate::hamiltonian::eval_coeffs(self.py, self.coeff_fns, t)
            .expect("SchrodingerEq: coefficient function call failed");

        // Re-interpret interleaved real/imaginary as Complex<f64> slices
        let psi: &[Complex<f64>] = bytemuck::cast_slice(y.as_slice());
        let dpsi: &mut [Complex<f64>] = bytemuck::cast_slice_mut(dy.as_mut_slice());

        self.matrix
            .dot(true, &coeffs, psi, dpsi)
            .expect("SchrodingerEq: dot product failed");

        // Apply -i: d|psi>/dt = -i*H|psi>  <->  -i*(a+ib) = b - ia
        for c in dpsi.iter_mut() {
            *c = Complex::new(c.im, -c.re);
        }
    }
}

// ---------------------------------------------------------------------------
// PySchrodingerEq
// ---------------------------------------------------------------------------

/// Python-facing Schrödinger equation integrator.
///
/// Wraps a `Hamiltonian` and drives the `ode_solvers` Dopri5 integrator.
/// The state vector is stored in interleaved real/imaginary format internally
/// but exposed to Python as a complex128 1-D array.
#[pyclass(name = "SchrodingerEq", module = "quspin._rs")]
pub struct PySchrodingerEq {
    matrix: QMatrixInner,
    coeff_fns: Vec<PyObject>,
}

#[pymethods]
impl PySchrodingerEq {
    /// Construct from a `Hamiltonian`.
    #[new]
    fn new(py: Python<'_>, hamiltonian: &PyHamiltonian) -> Self {
        PySchrodingerEq {
            matrix: hamiltonian.matrix.clone(),
            coeff_fns: hamiltonian
                .coeff_fns
                .iter()
                .map(|f| f.clone_ref(py))
                .collect(),
        }
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    #[getter]
    fn dim(&self) -> usize {
        self.matrix.dim()
    }

    fn __repr__(&self) -> String {
        format!(
            "SchrodingerEq(dim={}, dtype={})",
            self.matrix.dim(),
            self.matrix.dtype_name(),
        )
    }

    // ------------------------------------------------------------------
    // Integration
    // ------------------------------------------------------------------

    /// Integrate from `t0` to `t_end` starting from initial state `y0`.
    ///
    /// Args:
    ///     t0:    Initial time.
    ///     t_end: Final time.
    ///     y0:    Initial state — 1-D complex128 array of length `dim`.
    ///     rtol:  Relative tolerance for Dopri5 (default 1e-10).
    ///     atol:  Absolute tolerance for Dopri5 (default 1e-12).
    ///
    /// Returns:
    ///     Final state as a 1-D complex128 numpy array of length `dim`.
    #[pyo3(signature = (t0, t_end, y0, rtol = 1e-10, atol = 1e-12))]
    fn integrate<'py>(
        &self,
        py: Python<'py>,
        t0: f64,
        t_end: f64,
        y0: &Bound<'py, PyArray1<Complex64>>,
        rtol: f64,
        atol: f64,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let n = self.matrix.dim();
        if unsafe { y0.as_array().len() } != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "y0 must have length {n}"
            )));
        }

        // Pack initial state into interleaved real/imaginary DVector<f64>
        let mut packed = DVector::zeros(2 * n);
        {
            let arr = unsafe { y0.as_array() };
            for (k, c) in arr.iter().enumerate() {
                packed[2 * k] = c.re;
                packed[2 * k + 1] = c.im;
            }
        }

        // Build ODE system (borrows self without cloning PyObjects)
        let system = SchrodingerSystem {
            py,
            matrix: &self.matrix,
            coeff_fns: &self.coeff_fns,
        };

        // Run Dopri5 integrator
        let h0 = (t_end - t0).abs() * 1e-4;
        let mut stepper =
            ode_solvers::dopri5::Dopri5::new(system, t0, t_end, h0, packed, rtol, atol);
        stepper.integrate().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ODE integration failed: {e:?}"))
        })?;

        // Extract final state
        let yf = stepper
            .y_out()
            .last()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no ODE output produced"))?;
        let result: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(yf[2 * k], yf[2 * k + 1]))
            .collect();
        Ok(result.to_pyarray(py))
    }

    // ------------------------------------------------------------------
    // Dense output: all time-points
    // ------------------------------------------------------------------

    /// Integrate and return all accepted time-step outputs.
    ///
    /// Returns:
    ///     (times, states) where `times` is a 1-D float64 array and
    ///     `states` is a 2-D complex128 array of shape `(n_steps, dim)`.
    #[pyo3(signature = (t0, t_end, y0, rtol = 1e-10, atol = 1e-12))]
    fn integrate_dense<'py>(
        &self,
        py: Python<'py>,
        t0: f64,
        t_end: f64,
        y0: &Bound<'py, PyArray1<Complex64>>,
        rtol: f64,
        atol: f64,
    ) -> PyResult<DenseOutput<'py>> {
        let n = self.matrix.dim();
        if unsafe { y0.as_array().len() } != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "y0 must have length {n}"
            )));
        }

        let mut packed = DVector::zeros(2 * n);
        {
            let arr = unsafe { y0.as_array() };
            for (k, c) in arr.iter().enumerate() {
                packed[2 * k] = c.re;
                packed[2 * k + 1] = c.im;
            }
        }

        let system = SchrodingerSystem {
            py,
            matrix: &self.matrix,
            coeff_fns: &self.coeff_fns,
        };

        let h0 = (t_end - t0).abs() * 1e-4;
        let mut stepper =
            ode_solvers::dopri5::Dopri5::new(system, t0, t_end, h0, packed, rtol, atol);
        stepper.integrate().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ODE integration failed: {e:?}"))
        })?;

        let times: Vec<f64> = stepper.x_out().to_vec();
        let n_steps = times.len();

        let rows: Vec<Vec<Complex64>> = stepper
            .y_out()
            .iter()
            .map(|yf| {
                (0..n)
                    .map(|k| Complex64::new(yf[2 * k], yf[2 * k + 1]))
                    .collect()
            })
            .collect();

        let times_arr = times.to_pyarray(py);
        let states_arr = PyArray2::from_vec2(py, &rows)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let _ = n_steps;
        Ok((times_arr, states_arr))
    }
}
