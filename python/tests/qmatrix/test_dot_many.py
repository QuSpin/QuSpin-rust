"""Tests for QMatrix.dot_many / dot_transpose_many."""

import numpy as np

from quspin_rs._rs import PauliOperator, QMatrix, SpinBasis

N = 4


def make_pauli_op() -> PauliOperator:
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_spin_basis_full() -> SpinBasis:
    return SpinBasis.full(N)


class TestQMatrixDotMany:
    def _make(self):
        return QMatrix.build_pauli(
            make_pauli_op(), make_spin_basis_full(), np.dtype("complex128")
        )

    def test_dot_many_overwrite(self):
        mat = self._make()
        n, k = mat.dim, 3
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(0).standard_normal((n, k)) + 0j
        out = np.zeros((n, k), dtype=np.complex128)
        mat.dot_many(coeff, inp, out, True)
        assert not np.allclose(out, 0.0)

    def test_dot_many_accumulate(self):
        mat = self._make()
        n, k = mat.dim, 2
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(1).standard_normal((n, k)) + 0j
        out1 = np.zeros((n, k), dtype=np.complex128)
        out2 = np.zeros((n, k), dtype=np.complex128)
        mat.dot_many(coeff, inp, out1, True)
        mat.dot_many(coeff, inp, out2, True)
        mat.dot_many(coeff, inp, out2, False)
        np.testing.assert_allclose(out2, 2 * out1, atol=1e-12)

    def test_dot_many_equals_transpose_for_hermitian(self):
        """For a Hermitian matrix, H·v == H^†·v == H^T·conj(v)."""
        mat = self._make()
        n, k = mat.dim, 2
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(2).standard_normal((n, k)) + 0j
        out_fwd = np.zeros((n, k), dtype=np.complex128)
        out_trans = np.zeros((n, k), dtype=np.complex128)
        mat.dot_many(coeff, inp, out_fwd, True)
        mat.dot_transpose_many(coeff, inp.conj(), out_trans, True)
        np.testing.assert_allclose(out_fwd, out_trans.conj(), atol=1e-12)
