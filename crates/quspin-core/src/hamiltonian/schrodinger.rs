use crate::hamiltonian::ham::Hamiltonian;
use crate::primitive::Primitive;
use crate::qmatrix::matrix::{CIndex, Index};
use nalgebra::DVector;
use num_complex::Complex;
use ode_solvers::System;

/// `ode_solvers`-compatible wrapper for time-evolving a quantum state.
///
/// ## State layout
/// Interleaved real/imaginary: `y = [re₀, im₀, re₁, im₁, …]`
/// so `y.len() == 2 * hamiltonian.dim()`.
///
/// ## Panics (debug)
/// `new` panics if `y.len() != 2 * hamiltonian.dim()` inside `system`.
pub struct SchrodingerEq<M: Primitive, I: Index, C: CIndex> {
    hamiltonian: Hamiltonian<M, I, C>,
}

impl<M: Primitive, I: Index, C: CIndex> SchrodingerEq<M, I, C> {
    pub fn new(hamiltonian: Hamiltonian<M, I, C>) -> Self {
        SchrodingerEq { hamiltonian }
    }

    pub fn hamiltonian(&self) -> &Hamiltonian<M, I, C> {
        &self.hamiltonian
    }

    pub fn dim(&self) -> usize {
        self.hamiltonian.dim()
    }

    /// Pack a complex state slice into the interleaved `DVector<f64>` format.
    pub fn pack(psi: &[Complex<f64>]) -> DVector<f64> {
        let mut y = DVector::zeros(2 * psi.len());
        for (k, c) in psi.iter().enumerate() {
            y[2 * k] = c.re;
            y[2 * k + 1] = c.im;
        }
        y
    }

    /// Unpack an interleaved `DVector<f64>` into a `Vec<Complex<f64>>`.
    pub fn unpack(y: &DVector<f64>) -> Vec<Complex<f64>> {
        (0..y.len() / 2)
            .map(|k| Complex::new(y[2 * k], y[2 * k + 1]))
            .collect()
    }
}

impl<M, I, C> System<f64, DVector<f64>> for SchrodingerEq<M, I, C>
where
    M: Primitive,
    I: Index,
    C: CIndex,
{
    fn system(&self, t: f64, y: &DVector<f64>, dy: &mut DVector<f64>) {
        debug_assert_eq!(
            y.len(),
            2 * self.hamiltonian.dim(),
            "SchrodingerEq: y.len() ({}) must equal 2 * dim ({})",
            y.len(),
            2 * self.hamiltonian.dim(),
        );

        let psi: &[Complex<f64>] = bytemuck::cast_slice(y.as_slice());
        let dpsi: &mut [Complex<f64>] = bytemuck::cast_slice_mut(dy.as_mut_slice());

        self.hamiltonian
            .dot(true, t, psi, dpsi)
            .expect("SchrodingerEq::system: Hamiltonian::dot failed");

        // Apply -i: -i·(a + ib) = b - ia
        for c in dpsi.iter_mut() {
            *c = Complex::new(c.im, -c.re);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::space::FullSpace;
    use crate::operator::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
    use crate::qmatrix::build::build_from_basis;
    use num_complex::Complex;
    use ode_solvers::dopri5::Dopri5;
    use smallvec::smallvec;

    /// Pauli X on a 1-site basis: H = X = [[0,1],[1,0]]
    fn pauli_x_hamiltonian() -> SchrodingerEq<f64, i64, u8> {
        let ops = smallvec![(HardcoreOp::X, 0u32)];
        let terms = vec![OpEntry::new(0u8, Complex::new(1.0, 0.0), ops)];
        let op = HardcoreOperator::new(terms);
        let basis = FullSpace::<u32>::new(2, 1, false);
        let mat = build_from_basis::<_, u32, f64, i64, u8, _>(&op, &basis);
        let ham = Hamiltonian::new(mat, vec![]).unwrap();
        SchrodingerEq::new(ham)
    }

    #[test]
    fn diff_at_basis_state_zero() {
        // |ψ⟩ = |0⟩ = [1, 0]
        // H|ψ⟩ = X|0⟩ = |1⟩ = [0, 1]
        // dψ/dt = -i·H|ψ⟩ = -i·[0, 1] = [0, -i] → re/im interleaved: [0, 0, 0, -1]
        let eq = pauli_x_hamiltonian();
        let psi0 = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let y = SchrodingerEq::<f64, i64, u8>::pack(&psi0);
        let mut dy = DVector::zeros(4);
        eq.system(0.0, &y, &mut dy);

        let dpsi = SchrodingerEq::<f64, i64, u8>::unpack(&dy);
        assert!((dpsi[0] - Complex::new(0.0, 0.0)).norm() < 1e-12);
        assert!((dpsi[1] - Complex::new(0.0, -1.0)).norm() < 1e-12);
    }

    #[test]
    fn diff_at_basis_state_one() {
        // |ψ⟩ = |1⟩ = [0, 1]
        // H|ψ⟩ = X|1⟩ = |0⟩ = [1, 0]
        // dψ/dt = -i·[1, 0] = [-i, 0]
        let eq = pauli_x_hamiltonian();
        let psi1 = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        let y = SchrodingerEq::<f64, i64, u8>::pack(&psi1);
        let mut dy = DVector::zeros(4);
        eq.system(0.0, &y, &mut dy);

        let dpsi = SchrodingerEq::<f64, i64, u8>::unpack(&dy);
        assert!((dpsi[0] - Complex::new(0.0, -1.0)).norm() < 1e-12);
        assert!((dpsi[1] - Complex::new(0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn diff_is_linear_in_state() {
        // diff(α·y) = α·diff(y)
        let eq = pauli_x_hamiltonian();
        let alpha = Complex::new(2.0, 1.0);
        let psi = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];

        let y = SchrodingerEq::<f64, i64, u8>::pack(&psi);
        let psi_scaled: Vec<_> = psi.iter().map(|c| c * alpha).collect();
        let y_scaled = SchrodingerEq::<f64, i64, u8>::pack(&psi_scaled);

        let mut dy = DVector::zeros(4);
        let mut dy_scaled = DVector::zeros(4);
        eq.system(0.0, &y, &mut dy);
        eq.system(0.0, &y_scaled, &mut dy_scaled);

        let dpsi = SchrodingerEq::<f64, i64, u8>::unpack(&dy);
        let dpsi_scaled = SchrodingerEq::<f64, i64, u8>::unpack(&dy_scaled);

        for (a, b) in dpsi.iter().zip(dpsi_scaled.iter()) {
            assert!(
                (a * alpha - b).norm() < 1e-12,
                "linearity: {a}*alpha={}, got {b}",
                a * alpha
            );
        }
    }

    #[test]
    fn evolve_pauli_x_half_pi() {
        // |0⟩ under H=X evolved to t=π/2 should give -i|1⟩.
        // Analytical: ψ(t) = cos(t)|0⟩ - i·sin(t)|1⟩
        //
        // Note: ode_solvers stores the state at each accepted step boundary.
        // The last stored point may be just before t_end (the final trimmed
        // step is not re-pushed to y_out).  We therefore check accuracy
        // against the analytical solution at t_actual = x_out().last(), and
        // separately assert that t_actual is within 1e-3 of t_end.
        let eq = pauli_x_hamiltonian();
        let psi0 = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let y0 = SchrodingerEq::<f64, i64, u8>::pack(&psi0);

        let t0 = 0.0_f64;
        let t_end = std::f64::consts::FRAC_PI_2;
        let mut stepper = Dopri5::new(eq, t0, t_end, 1e-4, y0, 1e-10, 1e-12);
        stepper.integrate().expect("integration failed");

        let t_actual = *stepper.x_out().last().expect("no output");
        let yf = stepper.y_out().last().unwrap();
        let psif = SchrodingerEq::<f64, i64, u8>::unpack(yf);

        assert!(
            (t_actual - t_end).abs() < 1e-3,
            "integration stopped at t={t_actual}, expected t≈{t_end}"
        );

        // Verify numerical solution matches analytical at t_actual (tol 1e-8)
        let expected_0 = Complex::new(t_actual.cos(), 0.0);
        let expected_1 = Complex::new(0.0, -t_actual.sin());
        assert!(
            (psif[0] - expected_0).norm() < 1e-8,
            "ψ[0] = {}, expected {}",
            psif[0],
            expected_0
        );
        assert!(
            (psif[1] - expected_1).norm() < 1e-8,
            "ψ[1] = {}, expected {}",
            psif[1],
            expected_1
        );
    }
}
