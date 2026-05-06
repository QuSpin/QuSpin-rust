"""Tests for QMatrix.to_csr."""

import numpy as np

from quspin_rs._rs import PauliOperator, QMatrix, SpinBasis

N = 4


def make_pauli_op() -> PauliOperator:
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_spin_basis_full() -> SpinBasis:
    return SpinBasis.full(N)


class TestQMatrixCsr:
    def test_to_csr_shapes(self):
        mat = QMatrix.build_pauli(
            make_pauli_op(), make_spin_basis_full(), np.dtype("complex128")
        )
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        indptr, indices, data = mat.to_csr(coeff)
        n = mat.dim
        assert indptr.shape == (n + 1,)
        assert indices.shape == data.shape
        assert indptr[-1] == len(data)

    def test_to_csr_symmetry(self):
        """XX + ZZ is Hermitian: H[i,j] == conj(H[j,i])."""
        mat = QMatrix.build_pauli(
            make_pauli_op(), make_spin_basis_full(), np.dtype("complex128")
        )
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        indptr, indices, data = mat.to_csr(coeff)
        n = mat.dim
        # Build dense matrix from CSR manually
        dense = np.zeros((n, n), dtype=np.complex128)
        for row in range(n):
            for pos in range(indptr[row], indptr[row + 1]):
                dense[row, indices[pos]] += data[pos]
        diff = dense - dense.conj().T
        assert np.max(np.abs(diff)) < 1e-12
