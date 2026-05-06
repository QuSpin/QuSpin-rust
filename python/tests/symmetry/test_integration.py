"""Integration tests for symmetric SpinBasis (translation symmetry)."""

import pytest

from quspin_rs import Lattice, SymmetryGroup
from quspin_rs._rs import PauliOperator, SpinBasis


def _make_pbc_hopping_op(L: int) -> PauliOperator:
    """Single-particle XX+YY hopping on L sites with periodic boundary conditions."""
    xx_bonds = [[1.0, i, (i + 1) % L] for i in range(L)]
    yy_bonds = [[1.0, i, (i + 1) % L] for i in range(L)]
    return PauliOperator([("XX", xx_bonds)], [("YY", yy_bonds)])


def _translation_group(L: int, k: int = 0) -> SymmetryGroup:
    """Cyclic translation group on L sites at momentum sector k.

    Builds T = [1, 2, ..., L-1, 0] and adds it as a cyclic generator with
    character exp(-2*pi*i*k/L). The identity is implicit in `SymBasis`.
    """
    group = SymmetryGroup(n_sites=L, lhss=2)
    perm = [(i + 1) % L for i in range(L)]
    group.add_cyclic(Lattice(perm), k=k)
    return group


class TestSymmetricBasisTranslation:
    """For a single particle hopping on an L-site chain with PBC,
    the single-particle subspace has L states. With k=0 translation
    symmetry all L states collapse into a single orbit, so the
    symmetric basis should have size 1."""

    @pytest.mark.parametrize("L", [4, 6, 8, 10, 12, 16])
    def test_single_particle_k0_small(self, L):
        op = _make_pbc_hopping_op(L)
        seed = "1" + "0" * (L - 1)
        group = _translation_group(L)
        basis = SpinBasis.symmetric(group, op, [seed])
        assert basis.size == 1

    @pytest.mark.slow
    @pytest.mark.parametrize("L", [33, 48, 64])
    def test_single_particle_k0_u64(self, L):
        """L > 32 forces u64 bit representation."""
        op = _make_pbc_hopping_op(L)
        seed = "1" + "0" * (L - 1)
        group = _translation_group(L)
        basis = SpinBasis.symmetric(group, op, [seed])
        assert basis.size == 1

    @pytest.mark.slow
    @pytest.mark.parametrize("L", [65, 80, 128])
    def test_single_particle_k0_large_int(self, L):
        """L > 64 forces multi-word integer representation (ruint)."""
        op = _make_pbc_hopping_op(L)
        seed = "1" + "0" * (L - 1)
        group = _translation_group(L)
        basis = SpinBasis.symmetric(group, op, [seed])
        assert basis.size == 1

    @pytest.mark.parametrize(
        "L,k",
        [(4, 1), (4, 2), (4, 3), (6, 1), (6, 3), (8, 1), (8, 4), (12, 5), (16, 7)],
    )
    def test_single_particle_nonzero_k(self, L, k):
        """Every momentum sector has exactly 1 state for a single particle."""
        op = _make_pbc_hopping_op(L)
        seed = "1" + "0" * (L - 1)
        group = _translation_group(L, k)
        basis = SpinBasis.symmetric(group, op, [seed])
        assert basis.size == 1

    @pytest.mark.parametrize("L", [4, 6, 8])
    def test_subspace_without_symmetry_has_L_states(self, L):
        """Sanity check: without symmetry the subspace should have L states."""
        op = _make_pbc_hopping_op(L)
        seed = "1" + "0" * (L - 1)
        basis = SpinBasis.subspace(L, op, [seed])
        assert basis.size == L
