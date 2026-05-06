"""Tests for QMatrix.build_pauli — basic properties."""

import numpy as np

from quspin_rs._rs import PauliOperator, QMatrix, SpinBasis

N = 4


def make_pauli_op() -> PauliOperator:
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_spin_basis_full() -> SpinBasis:
    return SpinBasis.full(N)


class TestQMatrixBuild:
    def _make(self, dtype: str = "float64"):
        op = make_pauli_op()
        basis = make_spin_basis_full()
        return QMatrix.build_pauli(op, basis, np.dtype(dtype))

    def test_build_returns_qmatrix(self):
        assert isinstance(self._make(), QMatrix)

    def test_dim_matches_basis(self):
        basis = make_spin_basis_full()
        mat = QMatrix.build_pauli(make_pauli_op(), basis, np.dtype("float64"))
        assert mat.dim == basis.size

    def test_nnz_positive(self):
        assert self._make().nnz > 0

    def test_all_supported_dtypes(self):
        for dt in ["int8", "int16", "float32", "float64", "complex64", "complex128"]:
            mat = self._make(dt)
            assert mat.dim == make_spin_basis_full().size

    def test_repr(self):
        assert "QMatrix" in repr(self._make())
