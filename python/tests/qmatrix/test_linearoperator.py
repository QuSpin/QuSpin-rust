"""Tests for the SciPy ``LinearOperator`` duck-typed interface on
``QMatrixLinearOperator``."""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from quspin_rs._rs import PauliOperator, QMatrix, SpinBasis

N = 4


def make_pauli_op() -> PauliOperator:
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_qmatrix() -> QMatrix:
    return QMatrix.build_pauli(
        make_pauli_op(), SpinBasis.full(N), np.dtype("complex128")
    )


def dense_from(mat: QMatrix, coeff: np.ndarray) -> np.ndarray:
    indptr, indices, data = mat.to_csr(coeff)
    n = mat.dim
    dense = np.zeros((n, n), dtype=np.complex128)
    for r in range(n):
        for k in range(indptr[r], indptr[r + 1]):
            dense[r, indices[k]] += data[k]
    return dense


class TestQMatrixLinearOperator:
    def _make(self) -> tuple[QMatrix, np.ndarray, np.ndarray]:
        mat = make_qmatrix()
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        return mat, coeff, dense_from(mat, coeff)

    def test_shape_and_dtype(self):
        mat, coeff, _ = self._make()
        op = mat.as_linearoperator(coeff)
        n = mat.dim
        assert op.shape == (n, n)
        assert op.dtype == np.dtype("complex128")

    def test_matvec_matches_dense(self):
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        x = np.random.default_rng(0).standard_normal(mat.dim) + 0j
        y = op.matvec(x)
        np.testing.assert_allclose(y, dense @ x, atol=1e-12)
        assert y.dtype == np.complex128
        assert y.shape == (mat.dim,)

    def test_matmat_matches_dense(self):
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        rng = np.random.default_rng(1)
        k = 3
        X = rng.standard_normal((mat.dim, k)) + 1j * rng.standard_normal((mat.dim, k))
        Y = op.matmat(X)
        np.testing.assert_allclose(Y, dense @ X, atol=1e-12)
        assert Y.shape == (mat.dim, k)
        assert Y.dtype == np.complex128

    def test_rmatvec_matches_hermitian_adjoint(self):
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        x = np.random.default_rng(2).standard_normal(
            mat.dim
        ) + 1j * np.random.default_rng(3).standard_normal(mat.dim)
        y = op.rmatvec(x)
        np.testing.assert_allclose(y, dense.conj().T @ x, atol=1e-12)

    def test_rmatmat_matches_hermitian_adjoint(self):
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        rng = np.random.default_rng(4)
        k = 2
        X = rng.standard_normal((mat.dim, k)) + 1j * rng.standard_normal((mat.dim, k))
        Y = op.rmatmat(X)
        np.testing.assert_allclose(Y, dense.conj().T @ X, atol=1e-12)

    def test_matmul_dunder_1d(self):
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        x = np.random.default_rng(5).standard_normal(mat.dim) + 0j
        np.testing.assert_allclose(op @ x, dense @ x, atol=1e-12)

    def test_matmul_dunder_2d(self):
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        rng = np.random.default_rng(6)
        X = rng.standard_normal((mat.dim, 4)) + 1j * rng.standard_normal((mat.dim, 4))
        np.testing.assert_allclose(op @ X, dense @ X, atol=1e-12)

    def test_rmatmul_dunder_1d(self):
        """``x @ A`` for 1-D ``x`` returns ``A^T @ x`` (plain transpose)."""
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        x = np.random.default_rng(7).standard_normal(
            mat.dim
        ) + 1j * np.random.default_rng(8).standard_normal(mat.dim)
        np.testing.assert_allclose(x @ op, x @ dense, atol=1e-12)

    def test_rmatmul_dunder_2d(self):
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        rng = np.random.default_rng(9)
        m = 3
        X = rng.standard_normal((m, mat.dim)) + 1j * rng.standard_normal((m, mat.dim))
        np.testing.assert_allclose(X @ op, X @ dense, atol=1e-12)

    def test_matvec_wrong_shape_raises(self):
        mat, coeff, _ = self._make()
        op = mat.as_linearoperator(coeff)
        with pytest.raises(ValueError):
            op.matvec(np.zeros(mat.dim + 1, dtype=np.complex128))

    def test_matmat_wrong_shape_raises(self):
        mat, coeff, _ = self._make()
        op = mat.as_linearoperator(coeff)
        with pytest.raises(ValueError):
            op.matmat(np.zeros((mat.dim + 1, 2), dtype=np.complex128))

    def test_scipy_eigsh(self):
        """``scipy.sparse.linalg.eigsh`` works directly on the operator."""
        mat, coeff, dense = self._make()
        op = mat.as_linearoperator(coeff)
        eigvals_dense = np.sort(np.linalg.eigvalsh(dense))
        eigvals_lo = np.sort(spla.eigsh(op, k=3, which="SA", return_eigenvectors=False))
        np.testing.assert_allclose(eigvals_lo, eigvals_dense[:3], atol=1e-8)
