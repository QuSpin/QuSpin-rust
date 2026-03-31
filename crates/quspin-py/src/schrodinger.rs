use crate::hamiltonian::PyHamiltonian;
use nalgebra::DVector;
use num_complex::Complex;
use numpy::{Complex64, PyArrayMethods};
use numpy::{PyArray1, PyArray2, ToPyArray};
use ode_solvers::System;
use pyo3::prelude::*;
use quspin_core::hamiltonian::HamiltonianInner;
use std::sync::Arc;

/// Return type of `PySchrodingerEq::integrate_dense`.
type DenseOutput<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<Complex64>>);

// ---------------------------------------------------------------------------
// ODE system — wraps `HamiltonianInner`, GIL-free during integration
// ---------------------------------------------------------------------------

/// Implements `dy/dt = -i H(t) y` for the `ode_solvers` Dopri5 stepper.
///
/// State `y` is stored as interleaved real/imaginary floats so that
/// `DVector<f64>` can be used directly.  `HamiltonianInner::dot` is called
/// without holding the GIL; Python coefficient callables re-acquire it
/// via `Python::with_gil` (re-entrant) as needed.
struct SchrodingerSystem {
    inner: Arc<HamiltonianInner>,
}

impl System<f64, DVector<f64>> for SchrodingerSystem {
    fn system(&self, t: f64, y: &DVector<f64>, dy: &mut DVector<f64>) {
        // Safety: `DVector<f64>` has exactly `2*dim` elements; each pair
        // encodes one `Complex<f64>` in native endian — same layout as
        // `[f64; 2]`.  `bytemuck::cast_slice` verifies alignment and size.
        let psi: &[Complex<f64>] = bytemuck::cast_slice(y.as_slice());
        let dpsi: &mut [Complex<f64>] = bytemuck::cast_slice_mut(dy.as_mut_slice());

        self.inner
            .dot(true, t, psi, dpsi)
            .expect("SchrodingerEq: dot product failed");

        // d|ψ⟩/dt = −i H|ψ⟩  →  multiply by −i: (a+ib)·(−i) = b − ia
        for c in dpsi.iter_mut() {
            *c = Complex::new(c.im, -c.re);
        }
    }
}

// ---------------------------------------------------------------------------
// PySchrodingerEq
// ---------------------------------------------------------------------------

/// Python-facing Schrödinger equation integrator (Dopri5).
///
/// Shares the `Arc<HamiltonianInner>` from the `Hamiltonian` that was passed
/// to the constructor — no data is copied.  The GIL is released for the
/// entire duration of the ODE solve; Python coefficient callbacks re-acquire
/// it briefly per evaluation step via `Python::with_gil`.
#[pyclass(name = "SchrodingerEq", module = "quspin._rs")]
pub struct PySchrodingerEq {
    inner: Arc<HamiltonianInner>,
}

#[pymethods]
impl PySchrodingerEq {
    /// Construct from a `Hamiltonian`.
    #[new]
    fn new(_py: Python<'_>, hamiltonian: &PyHamiltonian) -> Self {
        PySchrodingerEq {
            inner: Arc::clone(&hamiltonian.inner),
        }
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __repr__(&self) -> String {
        format!(
            "SchrodingerEq(dim={}, dtype={})",
            self.inner.dim(),
            self.inner.dtype_name(),
        )
    }

    // ------------------------------------------------------------------
    // Integration
    // ------------------------------------------------------------------

    /// Integrate from `t0` to `t_end` starting from `y0`; return final state.
    ///
    /// The GIL is released for the entire ODE solve.
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
        let n = self.inner.dim();
        if unsafe { y0.as_array().len() } != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "y0 must have length {n}"
            )));
        }

        // Pack initial state into interleaved-real/imaginary DVector<f64>.
        let mut packed = DVector::zeros(2 * n);
        {
            let arr = unsafe { y0.as_array() };
            for (k, c) in arr.iter().enumerate() {
                packed[2 * k] = c.re;
                packed[2 * k + 1] = c.im;
            }
        }

        let inner = Arc::clone(&self.inner);

        // Release the GIL for the entire ODE solve.
        let yf_flat = py.allow_threads(move || -> Result<Vec<f64>, String> {
            let system = SchrodingerSystem { inner };
            let h0 = (t_end - t0).abs() * 1e-4;
            let mut stepper =
                ode_solvers::dopri5::Dopri5::new(system, t0, t_end, h0, packed, rtol, atol);
            stepper.integrate().map_err(|e| format!("{e:?}"))?;
            let yf = stepper
                .y_out()
                .last()
                .ok_or_else(|| "no ODE output produced".to_string())?;
            Ok(yf.as_slice().to_vec())
        });

        let yf_flat = yf_flat.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ODE integration failed: {e}"))
        })?;
        let result: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(yf_flat[2 * k], yf_flat[2 * k + 1]))
            .collect();
        Ok(result.to_pyarray(py))
    }

    // ------------------------------------------------------------------
    // Dense output: all accepted time-steps
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
        let n = self.inner.dim();
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

        let inner = Arc::clone(&self.inner);

        // (times: Vec<f64>, states_flat: Vec<f64>) — flat row-major (n_steps × 2n)
        let result = py.allow_threads(move || -> Result<(Vec<f64>, Vec<f64>), String> {
            let system = SchrodingerSystem { inner };
            let h0 = (t_end - t0).abs() * 1e-4;
            let mut stepper =
                ode_solvers::dopri5::Dopri5::new(system, t0, t_end, h0, packed, rtol, atol);
            stepper.integrate().map_err(|e| format!("{e:?}"))?;
            let times: Vec<f64> = stepper.x_out().to_vec();
            let flat: Vec<f64> = stepper
                .y_out()
                .iter()
                .flat_map(|yf| yf.as_slice().to_vec())
                .collect();
            Ok((times, flat))
        });

        let (times, states_flat) = result.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ODE integration failed: {e}"))
        })?;

        let n_steps = times.len();
        let times_arr = times.to_pyarray(py);

        // Build (n_steps, dim) complex128 array from flat interleaved buffer.
        let rows: Vec<Vec<Complex64>> = (0..n_steps)
            .map(|step| {
                (0..n)
                    .map(|k| {
                        Complex64::new(
                            states_flat[step * 2 * n + 2 * k],
                            states_flat[step * 2 * n + 2 * k + 1],
                        )
                    })
                    .collect()
            })
            .collect();
        let states_arr = PyArray2::from_vec2(py, &rows)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok((times_arr, states_arr))
    }
}
