"""Tests for SchrodingerEq (ODE-based time evolution)."""

import math

import numpy as np

from quspin_rs._rs import (
    Hamiltonian,
    PauliOperator,
    QMatrix,
    SchrodingerEq,
    SpinBasis,
    Static,
)


class TestSchrodingerEq:
    def _pauli_x_eq(self) -> SchrodingerEq:
        """H = X on a 1-site spin-1/2 basis."""
        op = PauliOperator([("X", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        ham = Hamiltonian(mat, [Static()])
        return SchrodingerEq(ham)

    def test_dim(self):
        assert self._pauli_x_eq().dim == 2

    def test_integrate_returns_correct_shape(self):
        eq = self._pauli_x_eq()
        y0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        yf = eq.integrate(0.0, math.pi / 2, y0)
        assert yf.shape == (2,)

    def test_integrate_pauli_x_half_pi(self):
        """Under H=X, |0> → cos(t)|0> - i sin(t)|1>.
        At t=pi/2: yf ≈ [0, -i] = [0, -i*1]."""
        eq = self._pauli_x_eq()
        y0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        yf = eq.integrate(0.0, math.pi / 2, y0, rtol=1e-10, atol=1e-12)
        assert abs(yf[0]) < 1e-6, f"expected yf[0]≈0, got {yf[0]}"
        assert abs(yf[1] - (-1j)) < 1e-6, f"expected yf[1]≈-i, got {yf[1]}"

    def test_integrate_norm_conserved(self):
        eq = self._pauli_x_eq()
        y0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        yf = eq.integrate(0.0, 1.0, y0)
        assert abs(np.linalg.norm(yf) - 1.0) < 1e-8

    def test_integrate_dense_shapes(self):
        eq = self._pauli_x_eq()
        y0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        times, states = eq.integrate_dense(0.0, 1.0, y0)
        assert times.ndim == 1
        assert states.ndim == 2
        assert states.shape[1] == 2
        assert states.shape[0] == len(times)
