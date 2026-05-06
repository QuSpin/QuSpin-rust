"""Tests for SpinBasis (full + subspace)."""

from quspin_rs._rs import PauliOperator, SpinBasis

N = 4


def make_pauli_op() -> PauliOperator:
    """XX + ZZ nearest-neighbour Hamiltonian on N sites."""
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


class TestSpinBasisFull:
    def test_size(self):
        basis = SpinBasis.full(N)
        assert basis.size == 2**N

    def test_n_sites(self):
        assert SpinBasis.full(N).n_sites == N

    def test_lhss(self):
        assert SpinBasis.full(N).lhss == 2

    def test_is_built(self):
        assert SpinBasis.full(N).is_built

    def test_state_at_returns_str(self):
        b = SpinBasis.full(2)
        s = b.state_at(0)
        assert isinstance(s, str)
        assert len(s) == 2

    def test_index_roundtrip(self):
        b = SpinBasis.full(3)
        for i in range(b.size):
            s = b.state_at(i)
            assert b.index(s) == i


class TestSpinBasisSubspace:
    def test_subspace_smaller_or_equal_full(self):
        op = make_pauli_op()
        full = SpinBasis.full(N)
        sub = SpinBasis.subspace(N, op, ["1100"])
        assert sub.size <= full.size

    def test_subspace_at_least_one(self):
        op = make_pauli_op()
        sub = SpinBasis.subspace(N, op, ["1100"])
        assert sub.size >= 1
