"""Tests for BosonBasis + BosonOperator and large-n_sites regression."""

import numpy as np
import pytest

from quspin_rs._rs import BosonBasis, BosonOperator, QMatrix

LHSS_B = 3
N_B = 3


def make_boson_op() -> BosonOperator:
    bonds = [[1.0, i, i + 1] for i in range(N_B - 1)]
    return BosonOperator([("+-", bonds)], [("-+", bonds)], lhss=LHSS_B)


class TestBosonBasisFull:
    def test_size(self):
        assert BosonBasis.full(N_B, LHSS_B).size == LHSS_B**N_B

    def test_n_sites(self):
        assert BosonBasis.full(N_B, LHSS_B).n_sites == N_B

    def test_lhss(self):
        assert BosonBasis.full(N_B, LHSS_B).lhss == LHSS_B


class TestQMatrixBoson:
    def test_build_and_dim(self):
        op = make_boson_op()
        basis = BosonBasis.full(N_B, LHSS_B)
        mat = QMatrix.build_boson(op, basis, np.dtype("float64"))
        assert mat.dim == basis.size
        assert mat.nnz > 0


class TestBosonBasisLargeNSites:
    """Regression test for issue #12: basis construction panics when n_sites >= 64."""

    @staticmethod
    def _make_boson_op(n: int, lhss: int) -> BosonOperator:
        terms = [[1.0, i, i + 1] for i in range(n - 1)] + [
            [1.0, i + 1, i] for i in range(n - 1)
        ]

        return BosonOperator([("+-", terms)], lhss=lhss)

    @pytest.mark.parametrize("N", [32, 63, 64, 65, 100, 128, 200])
    def test_single_particle_basis(self, N: int):
        nb = 1
        seed = ("1" * nb) + ("0" * (N - nb))
        basis = BosonBasis.subspace(N, 2, self._make_boson_op(N, 2), [seed])
        assert basis.size == N

    @pytest.mark.parametrize("N", [32, 63, 64, 65, 100, 128, 200])
    def test_two_particle_basis(self, N: int):
        nb = 2
        seed = ("1" * nb) + ("0" * (N - nb))
        basis = BosonBasis.subspace(N, 2, self._make_boson_op(N, 2), [seed])
        assert basis.size == N * (N - 1) // 2
