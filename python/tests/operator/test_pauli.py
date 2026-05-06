"""Tests for PauliOperator construction and properties."""

from quspin_rs._rs import PauliOperator

N = 4


def make_pauli_op() -> PauliOperator:
    """XX + ZZ nearest-neighbour Hamiltonian on N sites."""
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


class TestPauliOperator:
    def test_num_cindices(self):
        op = make_pauli_op()
        assert op.num_cindices == 2

    def test_max_site(self):
        op = make_pauli_op()
        assert op.max_site == N - 1

    def test_lhss(self):
        assert make_pauli_op().lhss == 2

    def test_repr(self):
        r = repr(make_pauli_op())
        assert "PauliOperator" in r
