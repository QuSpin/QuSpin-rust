"""Tests for FermionBasis + FermionOperator."""

import numpy as np

from quspin_rs._rs import FermionBasis, FermionOperator, QMatrix


def make_fermion_op() -> FermionOperator:
    N_F = 4
    bonds = [[1.0, i + 1, i] for i in range(N_F - 1)] + [
        [1.0, i, i + 1] for i in range(N_F - 1)
    ]
    return FermionOperator([("+-", bonds)])


class TestFermionBasisFull:
    def test_size(self):
        assert FermionBasis.full(4).size == 2**4

    def test_lhss(self):
        assert FermionBasis.full(4).lhss == 2


class TestQMatrixFermion:
    def test_build_and_dim(self):
        op = make_fermion_op()
        basis = FermionBasis.full(4)
        mat = QMatrix.build_fermion(op, basis, np.dtype("complex128"))
        assert mat.dim == basis.size
        assert mat.nnz > 0
