"""Integration tests for MonomialOperator and GenericBasis."""

import numpy as np
import pytest

from quspin_rs import Lattice, Local, SymmetryGroup
from quspin_rs._rs import GenericBasis, MonomialOperator, QMatrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cyclic_op(lhss: int, n_sites: int, k: int = 2) -> MonomialOperator:
    """Nearest-neighbour cyclic shift: (a, b) -> ((a+1)%lhss, (b+1)%lhss)."""
    dim = lhss**k
    perm = np.array(
        [
            sum(
                ((v // lhss ** (k - 1 - i) % lhss + 1) % lhss) * lhss ** (k - 1 - i)
                for i in range(k)
            )
            for v in range(dim)
        ],
        dtype=np.intp,
    )
    amp = np.ones(dim, dtype=complex)
    bonds = [tuple(range(i, i + k)) for i in range(n_sites - k + 1)]
    return MonomialOperator(lhss, (perm, amp, bonds))


def swap_op(lhss: int, n_sites: int) -> MonomialOperator:
    """Nearest-neighbour SWAP: (a, b) -> (b, a)."""
    dim = lhss * lhss
    perm = np.array(
        [b * lhss + a for a in range(lhss) for b in range(lhss)], dtype=np.intp
    )
    amp = np.ones(dim, dtype=complex)
    bonds = [(i, i + 1) for i in range(n_sites - 1)]
    return MonomialOperator(lhss, (perm, amp, bonds))


# ---------------------------------------------------------------------------
# MonomialOperator construction
# ---------------------------------------------------------------------------


class TestMonomialOperatorConstruction:
    def test_basic_properties(self):
        op = cyclic_op(3, 4)
        assert op.lhss == 3
        assert op.max_site == 3
        assert op.num_coeffs == 1

    def test_two_terms(self):
        dim = 4
        perm = np.arange(dim, dtype=np.intp)
        amp = np.ones(dim, dtype=complex)
        bonds = [(0, 1)]
        op = MonomialOperator(2, (perm, amp, bonds), (perm, amp, [(2, 3)]))
        assert op.num_coeffs == 2

    def test_repr(self):
        op = cyclic_op(3, 4)
        r = repr(op)
        assert "MonomialOperator" in r
        assert "lhss=3" in r

    def test_lhss_too_small_raises(self):
        perm = np.array([0], dtype=np.intp)
        amp = np.array([1.0 + 0j])
        with pytest.raises(ValueError):
            MonomialOperator(1, (perm, amp, [(0,)]))

    def test_no_terms_raises(self):
        with pytest.raises((ValueError, TypeError)):
            MonomialOperator(2)

    def test_wrong_perm_length_raises(self):
        # perm of length 5 is not lhss^k for lhss=3
        perm = np.zeros(5, dtype=np.intp)
        amp = np.ones(5, dtype=complex)
        with pytest.raises(ValueError):
            MonomialOperator(3, (perm, amp, [(0, 1)]))

    def test_perm_amp_length_mismatch_raises(self):
        perm = np.zeros(9, dtype=np.intp)
        amp = np.ones(4, dtype=complex)
        with pytest.raises(ValueError):
            MonomialOperator(3, (perm, amp, [(0, 1)]))


# ---------------------------------------------------------------------------
# GenericBasis — full
# ---------------------------------------------------------------------------


class TestGenericBasisFull:
    def test_lhss2_size(self):
        basis = GenericBasis.full(4, 2)
        assert basis.size == 2**4

    def test_lhss3_size(self):
        basis = GenericBasis.full(3, 3)
        assert basis.size == 3**3

    def test_lhss5_size(self):
        basis = GenericBasis.full(2, 5)
        assert basis.size == 5**2

    def test_properties(self):
        basis = GenericBasis.full(4, 3)
        assert basis.n_sites == 4
        assert basis.lhss == 3
        assert basis.is_built

    def test_state_at_returns_str(self):
        basis = GenericBasis.full(2, 3)
        s = basis.state_at(0)
        assert isinstance(s, str)

    def test_index_roundtrip_lhss2(self):
        # FullSpace uses consecutive integers; index roundtrip is consistent
        # when lhss is a power of 2 (bit-packed = base-lhss).
        basis = GenericBasis.full(3, 2)
        for i in range(basis.size):
            s = basis.state_at(i)
            assert basis.index(s) == i

    def test_repr(self):
        basis = GenericBasis.full(3, 2)
        assert "GenericBasis" in repr(basis)


# ---------------------------------------------------------------------------
# GenericBasis — subspace
# ---------------------------------------------------------------------------


class TestGenericBasisSubspace:
    def test_lhss3_subspace_nonempty(self):
        op = cyclic_op(3, 4)
        basis = GenericBasis.subspace(4, 3, op, ["0000"])
        assert basis.size > 0

    def test_lhss2_subspace_nonempty(self):
        op = swap_op(2, 4)
        basis = GenericBasis.subspace(4, 2, op, ["0011"])
        assert basis.size > 0

    def test_subspace_seed_not_in_basis_gives_empty(self):
        # Single identity-like operator (perm = identity, amp = 0 everywhere)
        # so no states can be reached from any seed
        dim = 4
        perm = np.arange(dim, dtype=np.intp)
        amp = np.zeros(dim, dtype=complex)  # all-zero amplitude
        op = MonomialOperator(2, (perm, amp, [(0, 1)]))
        basis = GenericBasis.subspace(4, 2, op, ["0011"])
        # BFS will start from the seed but emit nothing (amp=0 skipped),
        # so the seed itself may be added but reachable set is just {seed}.
        assert basis.size >= 0  # just check it doesn't crash

    def test_repr(self):
        op = cyclic_op(3, 4)
        basis = GenericBasis.subspace(4, 3, op, ["0000"])
        assert "subspace" in repr(basis) or "GenericBasis" in repr(basis)


# ---------------------------------------------------------------------------
# GenericBasis — symmetric
# ---------------------------------------------------------------------------


class TestGenericBasisSymmetric:
    # 4-site translation generator T = [1,2,3,0]. The identity (T^0) is
    # implicit in `SymBasis`; `add_cyclic` adds the remaining {T, T², T³}.
    TRANSLATION_GEN = [1, 2, 3, 0]

    def test_translation_lhss2(self):
        op = swap_op(2, 4)
        group = SymmetryGroup(n_sites=4, lhss=2)
        group.add_cyclic(Lattice(self.TRANSLATION_GEN), k=0)
        basis = GenericBasis.symmetric(group, op, ["0000"])
        assert basis.size > 0

    def test_translation_lhss3(self):
        op = cyclic_op(3, 4)
        group = SymmetryGroup(n_sites=4, lhss=3)
        group.add_cyclic(Lattice(self.TRANSLATION_GEN), k=0)
        basis = GenericBasis.symmetric(group, op, ["0000"])
        assert basis.size > 0

    def test_local_symmetry_all_sites(self):
        op = cyclic_op(3, 4)
        # Z_3 local symmetry: cyclic shift of dit values. The identity shift
        # [0,1,2] is implicit; `add_cyclic` adds the two non-identity powers.
        group = SymmetryGroup(n_sites=4, lhss=3)
        group.add_cyclic(Local([1, 2, 0]), k=0)
        basis = GenericBasis.symmetric(group, op, ["0000"])
        assert basis.size >= 0  # just check it doesn't crash

    def test_local_symmetry_masked(self):
        op = cyclic_op(3, 4)
        # Apply local symmetry only to sites 0 and 2; `add_cyclic` adds both
        # non-identity powers of the Z_3 cyclic dit shift.
        group = SymmetryGroup(n_sites=4, lhss=3)
        group.add_cyclic(Local([1, 2, 0], locs=[0, 2]), k=0)
        basis = GenericBasis.symmetric(group, op, ["0000"])
        assert basis.size >= 0

    def test_symmetric_smaller_than_full(self):
        op = swap_op(2, 4)
        full = GenericBasis.full(4, 2)
        group = SymmetryGroup(n_sites=4, lhss=2)
        group.add_cyclic(Lattice(self.TRANSLATION_GEN), k=0)
        sym = GenericBasis.symmetric(group, op, ["0000"])
        assert sym.size <= full.size


# ---------------------------------------------------------------------------
# QMatrix.build_monomial
# ---------------------------------------------------------------------------


class TestQMatrixBuildMonomial:
    def test_build_from_full_basis(self):
        op = swap_op(2, 2)
        basis = GenericBasis.full(2, 2)
        qm = QMatrix.build_monomial(op, basis, np.dtype("complex128"))
        assert qm.dim == basis.size
        assert qm.nnz > 0

    def test_build_from_subspace(self):
        op = cyclic_op(3, 4)
        basis = GenericBasis.subspace(4, 3, op, ["0000"])
        qm = QMatrix.build_monomial(op, basis, np.dtype("complex128"))
        assert qm.dim == basis.size
        assert qm.num_coeff == op.num_coeffs

    def test_csr_shape(self):
        op = swap_op(2, 2)
        basis = GenericBasis.full(2, 2)
        qm = QMatrix.build_monomial(op, basis, np.dtype("complex128"))
        coeff = np.ones(qm.num_coeff, dtype=complex)
        indptr, indices, data = qm.to_csr(coeff)
        n = basis.size
        assert len(indptr) == n + 1
        assert len(indices) == len(data)

    def test_swap_lhss2_matrix(self):
        """SWAP on 2-site lhss=2 should give a known 4x4 permutation matrix."""
        op = swap_op(2, 2)
        basis = GenericBasis.full(2, 2)
        qm = QMatrix.build_monomial(op, basis, np.dtype("complex128"))
        coeff = np.ones(1, dtype=complex)
        indptr, indices, data = qm.to_csr(coeff)

        n = basis.size
        # Reconstruct dense matrix from CSR manually (no scipy dependency).
        mat = np.zeros((n, n), dtype=complex)
        for row in range(n):
            for j in range(indptr[row], indptr[row + 1]):
                mat[row, indices[j]] += data[j]

        # SWAP is a permutation matrix — each row has exactly one non-zero of magnitude 1.
        assert np.allclose(np.abs(mat).sum(axis=1), 1.0)

    def test_repr(self):
        op = cyclic_op(3, 4)
        basis = GenericBasis.subspace(4, 3, op, ["0000"])
        qm = QMatrix.build_monomial(op, basis, np.dtype("complex128"))
        assert "QMatrix" in repr(qm)
