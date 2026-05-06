"""Tests for Hamiltonian construction, to_csr, and dot_many."""

import math

import numpy as np

from quspin_rs._rs import Hamiltonian, PauliOperator, QMatrix, SpinBasis, Static

N = 4


def make_pauli_op() -> PauliOperator:
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_spin_basis_full() -> SpinBasis:
    return SpinBasis.full(N)


class TestHamiltonian:
    def _make_static(self):
        """Static Hamiltonian: XX+ZZ — both cindices are Static."""
        op = make_pauli_op()
        basis = make_spin_basis_full()
        mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
        return Hamiltonian(mat, [Static(), Static()])

    def test_dim(self):
        h = self._make_static()
        assert h.dim == make_spin_basis_full().size

    def test_num_coeff(self):
        h = self._make_static()
        assert h.num_coeff == 2  # XX (cindex 0) + ZZ (cindex 1)

    def test_to_csr_matches_qmatrix_csr(self):
        op = make_pauli_op()
        basis = make_spin_basis_full()
        mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
        ham = Hamiltonian(mat, [Static(), Static()])
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        ip, ii, id_ = mat.to_csr(coeff)
        hp, hi, hd = ham.to_csr(0.0)
        np.testing.assert_array_equal(ip, hp)
        np.testing.assert_array_equal(ii, hi)
        np.testing.assert_allclose(id_, hd, atol=1e-14)

    def test_dot_matches_qmatrix_dot_many(self):
        op = make_pauli_op()
        basis = make_spin_basis_full()
        mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
        ham = Hamiltonian(mat, [Static(), Static()])
        n = mat.dim
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(7).standard_normal((n, 2)) + 0j
        out_q = np.zeros((n, 2), dtype=np.complex128)
        out_h = np.zeros((n, 2), dtype=np.complex128)
        mat.dot_many(coeff, inp, out_q, True)
        ham.dot_many(0.0, inp, out_h, True)
        np.testing.assert_allclose(out_q, out_h, atol=1e-14)

    def test_time_dependent_coeff(self):
        """Scaling coupling by cos(t) should give cos(t) * static result."""
        op = PauliOperator([("ZZ", [[1.0, 0, 1]])], [("ZZ", [[1.0, 0, 1]])])
        basis = SpinBasis.full(2)
        mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
        ham = Hamiltonian(mat, [Static(), lambda t: math.cos(t) + 0j])
        n = mat.dim
        inp = np.random.default_rng(8).standard_normal((n, 1)) + 0j
        out_t0 = np.zeros((n, 1), dtype=np.complex128)
        out_pi4 = np.zeros((n, 1), dtype=np.complex128)
        ham.dot_many(0.0, inp, out_t0, True)  # cos(0)=1, both cindices weight 1
        ham.dot_many(math.pi / 4, inp, out_pi4, True)  # cindex 1 weight = cos(pi/4)
        # At t=0: result = (1 + 1) * ZZ·v = 2 * ZZ·v
        # At t=pi/4: result = (1 + cos(pi/4)) * ZZ·v
        ratio = np.linalg.norm(out_pi4) / np.linalg.norm(out_t0)
        expected_ratio = (1.0 + math.cos(math.pi / 4)) / 2.0
        assert abs(ratio - expected_ratio) < 1e-12
