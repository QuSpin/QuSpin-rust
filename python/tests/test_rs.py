"""Tests for the quspin._rs PyO3 extension."""

import numpy as np
import pytest

from quspin._rs import (
    PyGrpElement,
    PyHardcoreBasis,
    PyLatticeElement,
    PyPauliHamiltonian,
    PyQMatrix,
    PySymmetryGrp,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Simple 4-site XX+ZZ Hamiltonian used in many tests.
# Terms: [coupling_list per cindex]
#   cindex 0 → "xx" on (0,1), (1,2), (2,3)   (nearest-neighbour XX)
#   cindex 1 → "zz" on (0,1), (1,2), (2,3)   (nearest-neighbour ZZ)
N = 4

XX_ZZ_TERMS = [
    [("xx", [(1.0, 0, 1), (1.0, 1, 2), (1.0, 2, 3)])],  # cindex 0
    [("zz", [(1.0, 0, 1), (1.0, 1, 2), (1.0, 2, 3)])],  # cindex 1
]


def make_ham() -> PyPauliHamiltonian:
    return PyPauliHamiltonian(XX_ZZ_TERMS)


def make_full_basis() -> PyHardcoreBasis:
    return PyHardcoreBasis.full(N)


# ---------------------------------------------------------------------------
# PyPauliHamiltonian
# ---------------------------------------------------------------------------


class TestPauliHamiltonian:
    def test_n_sites(self):
        h = make_ham()
        assert h.n_sites == N

    def test_num_cindices(self):
        h = make_ham()
        assert h.num_cindices == 2

    def test_single_cindex(self):
        h = PyPauliHamiltonian([[("z", [(1.0, 0), (1.0, 1)])]])
        assert h.n_sites == 2
        assert h.num_cindices == 1

    def test_bad_op_char(self):
        with pytest.raises(Exception):
            PyPauliHamiltonian([[("q", [(1.0, 0)])]])

    def test_wrong_site_count(self):
        # op_str "xx" expects 2 sites but coupling has only 1
        with pytest.raises(Exception):
            PyPauliHamiltonian([[("xx", [(1.0, 0)])]])


# ---------------------------------------------------------------------------
# PyHardcoreBasis — full
# ---------------------------------------------------------------------------


class TestHardcoreBasisFull:
    def test_full_size(self):
        basis = PyHardcoreBasis.full(N)
        assert basis.size == 2**N

    def test_full_size_small(self):
        for n in range(1, 8):
            assert PyHardcoreBasis.full(n).size == 2**n

    def test_full_too_large(self):
        with pytest.raises(Exception):
            PyHardcoreBasis.full(65)


# ---------------------------------------------------------------------------
# PyHardcoreBasis — subspace
# ---------------------------------------------------------------------------


class TestHardcoreBasisSubspace:
    def test_half_filling_sector(self):
        # XX is NOT particle-number conserving (it flips pairs of bits), so the
        # subspace reachable from any seed under XX+ZZ spans multiple sectors.
        seeds = ["1100"]  # sites 0,1 occupied
        h = make_ham()
        basis = PyHardcoreBasis.subspace(seeds, h)
        assert basis.size >= 1
        assert basis.size <= 2**N

    def test_single_state_trivial_ham(self):
        # A Hamiltonian with only Z operators doesn't mix states, so seed = 1 state.
        h = PyPauliHamiltonian([[("z", [(1.0, 0)])]])
        basis = PyHardcoreBasis.subspace(["0"], h)
        assert basis.size == 1

    def test_list_seed_equivalent_to_str_seed(self):
        h = make_ham()
        b_str = PyHardcoreBasis.subspace(["1100"], h)
        b_list = PyHardcoreBasis.subspace([[1, 1, 0, 0]], h)
        assert b_str.size == b_list.size

    def test_multiple_seeds_give_more_states(self):
        h = make_ham()
        b1 = PyHardcoreBasis.subspace(["1000"], h)
        b2 = PyHardcoreBasis.subspace(["1000", "0100"], h)
        assert b2.size >= b1.size

    def test_invalid_seed_str_char(self):
        h = make_ham()
        with pytest.raises(Exception):
            PyHardcoreBasis.subspace(["2100"], h)

    def test_invalid_seed_list_value(self):
        h = make_ham()
        with pytest.raises(Exception):
            PyHardcoreBasis.subspace([[2, 1, 0, 0]], h)


# ---------------------------------------------------------------------------
# PyQMatrix — build and properties
# ---------------------------------------------------------------------------


class TestQMatrixBuild:
    def test_build_returns_qmatrix(self):
        h = make_ham()
        basis = make_full_basis()
        mat = PyQMatrix.build(h, basis, np.dtype("float64"))
        assert isinstance(mat, PyQMatrix)

    def test_dim_matches_basis(self):
        h = make_ham()
        basis = make_full_basis()
        mat = PyQMatrix.build(h, basis, np.dtype("float64"))
        assert mat.dim == basis.size

    def test_nnz_positive(self):
        h = make_ham()
        basis = make_full_basis()
        mat = PyQMatrix.build(h, basis, np.dtype("float64"))
        assert mat.nnz > 0

    def test_all_supported_dtypes(self):
        h = make_ham()
        basis = make_full_basis()
        for dt in ["int8", "int16", "float32", "float64", "complex64", "complex128"]:
            mat = PyQMatrix.build(h, basis, np.dtype(dt))
            assert mat.dim == basis.size

    def test_unsupported_dtype_raises(self):
        h = make_ham()
        basis = make_full_basis()
        with pytest.raises(Exception):
            PyQMatrix.build(h, basis, np.dtype("int32"))


# ---------------------------------------------------------------------------
# PyQMatrix — dot / dot_transpose
# ---------------------------------------------------------------------------


class TestQMatrixDot:
    def _build(self, dtype=np.float64):
        h = make_ham()
        basis = make_full_basis()
        return PyQMatrix.build(h, basis, np.dtype(dtype))

    def test_dot_overwrite(self):
        mat = self._build()
        dim = mat.dim
        coeff = np.ones(mat.nnz, dtype=np.float64)  # wrong shape — use 1 per cindex
        # coeff has shape (num_cindices,)
        coeff = np.array([1.0, 1.0], dtype=np.float64)
        v = np.random.default_rng(0).standard_normal(dim)
        out = np.zeros(dim, dtype=np.float64)
        mat.dot(coeff, v, out, True)
        assert not np.allclose(out, 0.0)

    def test_dot_accumulate(self):
        mat = self._build()
        coeff = np.array([1.0, 1.0], dtype=np.float64)
        v = np.random.default_rng(1).standard_normal(mat.dim)
        out1 = np.zeros(mat.dim, dtype=np.float64)
        out2 = np.zeros(mat.dim, dtype=np.float64)
        mat.dot(coeff, v, out1, overwrite=True)
        mat.dot(coeff, v, out2, overwrite=True)
        mat.dot(coeff, v, out2, overwrite=False)  # accumulates: out2 = 2 * out1
        np.testing.assert_allclose(out2, 2 * out1)

    def test_dot_transpose_shape(self):
        mat = self._build()
        coeff = np.array([1.0, 1.0], dtype=np.float64)
        v = np.random.default_rng(2).standard_normal(mat.dim)
        out = np.zeros(mat.dim, dtype=np.float64)
        mat.dot_transpose(coeff, v, out, True)
        assert out.shape == (mat.dim,)

    def test_dot_wrong_dtype_raises(self):
        mat = self._build(np.float64)
        coeff = np.array([1.0, 1.0], dtype=np.float32)  # wrong dtype
        v = np.zeros(mat.dim, dtype=np.float32)
        out = np.zeros(mat.dim, dtype=np.float32)
        with pytest.raises(TypeError):
            mat.dot(coeff, v, out, True)

    def test_hermitian_dot_equals_dot_transpose_for_real_symmetric(self):
        """For a real symmetric Hamiltonian, H·v == H^T·v."""
        mat = self._build()
        coeff = np.array([1.0, 1.0], dtype=np.float64)
        v = np.random.default_rng(3).standard_normal(mat.dim)
        out_fwd = np.zeros(mat.dim, dtype=np.float64)
        out_trans = np.zeros(mat.dim, dtype=np.float64)
        mat.dot(coeff, v, out_fwd, True)
        mat.dot_transpose(coeff, v, out_trans, True)
        np.testing.assert_allclose(out_fwd, out_trans, atol=1e-12)


# ---------------------------------------------------------------------------
# PyQMatrix — arithmetic
# ---------------------------------------------------------------------------


class TestQMatrixArithmetic:
    def _build(self):
        h = make_ham()
        basis = make_full_basis()
        return PyQMatrix.build(h, basis, np.dtype("float64"))

    def test_add_same_nnz_or_more(self):
        mat = self._build()
        result = mat + mat
        assert result.nnz >= mat.nnz

    def test_add_dot_is_double(self):
        mat = self._build()
        result = mat + mat
        coeff = np.array([1.0, 1.0], dtype=np.float64)
        v = np.random.default_rng(4).standard_normal(mat.dim)
        out1 = np.zeros(mat.dim, dtype=np.float64)
        out2 = np.zeros(mat.dim, dtype=np.float64)
        mat.dot(coeff, v, out1, True)
        result.dot(coeff, v, out2, True)
        np.testing.assert_allclose(out2, 2 * out1, atol=1e-12)

    def test_sub_self_gives_zero_action(self):
        mat = self._build()
        result = mat - mat
        # mat - mat cancels all entries → nnz == 0 and num_coeff == 0.
        assert result.nnz == 0
        coeff = np.array([], dtype=np.float64)
        v = np.random.default_rng(5).standard_normal(mat.dim)
        out = np.zeros(mat.dim, dtype=np.float64)
        result.dot(coeff, v, out, True)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# PySymmetryGrp / symmetric basis
# ---------------------------------------------------------------------------


class TestSymmetricBasis:
    def test_translation_symmetry_reduces_dim(self):
        # 4-site chain: translation by 1 site, momentum sector k=0 (char=1).
        T = PyLatticeElement(grp_char=1.0 + 0j, perm=[1, 2, 3, 0], lhss=2)
        grp = PySymmetryGrp([T], [])
        h = make_ham()
        seeds = ["1100"]  # sites 0,1 occupied (0b0011)
        basis = PyHardcoreBasis.symmetric(seeds, h, grp)
        full_basis = PyHardcoreBasis.subspace(seeds, h)
        assert basis.size <= full_basis.size

    def test_bitflip_symmetry(self):
        P = PyGrpElement.bitflip(grp_char=1.0 + 0j, sites=[0, 1, 2, 3])
        grp = PySymmetryGrp([], [P])
        h = make_ham()
        # Use the canonical representative: the larger of each (state, partner) pair.
        # 0b1110 = 14 is the partner of 0b0001 = 1; 14 > 1 so 0b1110 is canonical.
        # "0111" = site0=0, site1=1, site2=1, site3=1 (0b1110).
        seeds = ["0111"]
        basis = PyHardcoreBasis.symmetric(seeds, h, grp)
        subspace = PyHardcoreBasis.subspace(seeds, h)
        assert basis.size <= subspace.size

    def test_symmetric_build_qmatrix(self):
        T = PyLatticeElement(grp_char=1.0 + 0j, perm=[1, 2, 3, 0], lhss=2)
        grp = PySymmetryGrp([T], [])
        h = make_ham()
        seeds = ["1100"]  # sites 0,1 occupied (0b0011)
        basis = PyHardcoreBasis.symmetric(seeds, h, grp)
        mat = PyQMatrix.build(h, basis, np.dtype("complex128"))
        assert mat.dim == basis.size
        assert mat.nnz >= 0
