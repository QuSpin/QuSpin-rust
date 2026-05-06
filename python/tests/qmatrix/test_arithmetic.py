"""Tests for QMatrix arithmetic (+, scalar mul, etc.)."""

import numpy as np

from quspin_rs._rs import PauliOperator, QMatrix, SpinBasis

N = 4


def make_pauli_op() -> PauliOperator:
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_spin_basis_full() -> SpinBasis:
    return SpinBasis.full(N)


class TestQMatrixArithmetic:
    def _make(self):
        return QMatrix.build_pauli(
            make_pauli_op(), make_spin_basis_full(), np.dtype("complex128")
        )

    def test_add_nnz_at_least_original(self):
        mat = self._make()
        assert (mat + mat).nnz >= mat.nnz

    def test_add_dot_is_double(self):
        mat = self._make()
        doubled = mat + mat
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(3).standard_normal((mat.dim, 1)) + 0j
        out1 = np.zeros((mat.dim, 1), dtype=np.complex128)
        out2 = np.zeros((mat.dim, 1), dtype=np.complex128)
        mat.dot_many(coeff, inp, out1, True)
        doubled.dot_many(coeff, inp, out2, True)
        np.testing.assert_allclose(out2, 2 * out1, atol=1e-12)
