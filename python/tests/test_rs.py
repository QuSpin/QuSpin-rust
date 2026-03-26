"""Tests for the quspin_rs._rs PyO3 extension."""

import numpy as np
import pytest

from quspin_rs._rs import (
    PyBosonHamiltonian,
    PyDitBasis,
    PyHardcoreBasis,
    PyHardcoreHamiltonian,
    PyQMatrix,
    PySpinSymGrp,
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


def make_ham() -> PyHardcoreHamiltonian:
    return PyHardcoreHamiltonian(XX_ZZ_TERMS)


def make_full_basis() -> PyHardcoreBasis:
    return PyHardcoreBasis.full(N)


# ---------------------------------------------------------------------------
# PyHardcoreHamiltonian
# ---------------------------------------------------------------------------


class TestPauliHamiltonian:
    def test_max_site(self):
        h = make_ham()
        assert h.max_site == N - 1

    def test_num_cindices(self):
        h = make_ham()
        assert h.num_cindices == 2

    def test_single_cindex(self):
        h = PyHardcoreHamiltonian([[("z", [(1.0, 0), (1.0, 1)])]])
        assert h.max_site == 1
        assert h.num_cindices == 1

    def test_bad_op_char(self):
        with pytest.raises(Exception):
            PyHardcoreHamiltonian([[("q", [(1.0, 0)])]])

    def test_wrong_site_count(self):
        # op_str "xx" expects 2 sites but coupling has only 1
        with pytest.raises(Exception):
            PyHardcoreHamiltonian([[("xx", [(1.0, 0)])]])


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
        h = PyHardcoreHamiltonian([[("z", [(1.0, 0)])]])
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
        mat = PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype("float64"))
        assert isinstance(mat, PyQMatrix)

    def test_dim_matches_basis(self):
        h = make_ham()
        basis = make_full_basis()
        mat = PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype("float64"))
        assert mat.dim == basis.size

    def test_nnz_positive(self):
        h = make_ham()
        basis = make_full_basis()
        mat = PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype("float64"))
        assert mat.nnz > 0

    def test_all_supported_dtypes(self):
        h = make_ham()
        basis = make_full_basis()
        for dt in ["int8", "int16", "float32", "float64", "complex64", "complex128"]:
            mat = PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype(dt))
            assert mat.dim == basis.size

    def test_unsupported_dtype_raises(self):
        h = make_ham()
        basis = make_full_basis()
        with pytest.raises(Exception):
            PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype("int32"))


# ---------------------------------------------------------------------------
# PyQMatrix — dot / dot_transpose
# ---------------------------------------------------------------------------


class TestQMatrixDot:
    def _build(self, dtype=np.float64):
        h = make_ham()
        basis = make_full_basis()
        return PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype(dtype))

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
        return PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype("float64"))

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
# PySpinSymGrp / symmetric basis
# ---------------------------------------------------------------------------


class TestSymmetricBasis:
    def test_translation_symmetry_reduces_dim(self):
        # 4-site chain: translation by 1 site, momentum sector k=0 (char=1).
        grp = PySpinSymGrp(lhss=2, n_sites=N)
        grp.add_lattice(grp_char=1.0 + 0j, perm=[1, 2, 3, 0])
        h = make_ham()
        seeds = ["1100"]
        basis = PyHardcoreBasis.symmetric(seeds, h, grp)
        full_basis = PyHardcoreBasis.subspace(seeds, h)
        assert basis.size <= full_basis.size

    def test_bitflip_symmetry(self):
        grp = PySpinSymGrp(lhss=2, n_sites=N)
        grp.add_inverse(grp_char=1.0 + 0j, locs=list(range(N)))
        assert grp.n_sites == N
        h = make_ham()
        # 0b1110 = 14 is the partner of 0b0001 = 1; 14 > 1 so 0b1110 is canonical.
        seeds = ["0111"]
        basis = PyHardcoreBasis.symmetric(seeds, h, grp)
        subspace = PyHardcoreBasis.subspace(seeds, h)
        assert basis.size <= subspace.size

    def test_symmetric_build_qmatrix(self):
        grp = PySpinSymGrp(lhss=2, n_sites=N)
        grp.add_lattice(grp_char=1.0 + 0j, perm=[1, 2, 3, 0])
        h = make_ham()
        seeds = ["1100"]
        basis = PyHardcoreBasis.symmetric(seeds, h, grp)
        mat = PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype("complex128"))
        assert mat.dim == basis.size
        assert mat.nnz >= 0

    def test_basis_n_sites(self):
        basis = make_full_basis()
        assert basis.n_sites == N

    def test_grp_n_sites(self):
        grp = PySpinSymGrp(lhss=2, n_sites=N)
        assert grp.n_sites == N

    def test_grp_lhss(self):
        grp = PySpinSymGrp(lhss=2, n_sites=N)
        assert grp.lhss == 2

    def test_bitflip_explicit_locs_same_as_all(self):
        # Flipping all sites explicitly should give same basis as flipping all.
        grp_all = PySpinSymGrp(lhss=2, n_sites=N)
        grp_all.add_inverse(grp_char=1.0 + 0j, locs=list(range(N)))
        grp_explicit = PySpinSymGrp(lhss=2, n_sites=N)
        grp_explicit.add_inverse(grp_char=1.0 + 0j, locs=[0, 1, 2, 3])
        h = make_ham()
        seeds = ["0111"]
        b_all = PyHardcoreBasis.symmetric(seeds, h, grp_all)
        b_explicit = PyHardcoreBasis.symmetric(seeds, h, grp_explicit)
        assert b_all.size == b_explicit.size

    def test_n_sites_mismatch_symmetric_raises(self):
        # Group built for 3 sites, ham for 4 sites → error.
        grp = PySpinSymGrp(lhss=2, n_sites=3)
        grp.add_lattice(grp_char=1.0 + 0j, perm=[1, 2, 0])
        h = make_ham()  # 4 sites
        with pytest.raises(Exception):
            PyHardcoreBasis.symmetric(["110"], h, grp)

    def test_n_sites_mismatch_build_raises(self):
        # Basis for 2 sites, ham for 4 sites → error.
        basis = PyHardcoreBasis.full(2)
        h = make_ham()  # 4 sites
        with pytest.raises(Exception):
            PyQMatrix.build_hardcore_hamiltonian(h, basis, np.dtype("float64"))

    def test_lhss2_n_sites_too_large_raises(self):
        with pytest.raises(Exception):
            PySpinSymGrp(lhss=2, n_sites=8193)


# ---------------------------------------------------------------------------
# PyBosonHamiltonian
# ---------------------------------------------------------------------------

LHSS = 3
N_BOSON = 3  # 3-site chain


def make_boson_ham() -> PyBosonHamiltonian:
    """Bose-Hubbard hopping: H = Σ_i (a†_i a_{i+1} + h.c.)"""
    return PyBosonHamiltonian(
        lhss=LHSS,
        terms=[
            [("+-", [(1.0, 0, 1), (1.0, 1, 2)])],  # a†_i a_j, cindex 0
            [("-+", [(1.0, 0, 1), (1.0, 1, 2)])],  # a_i a†_j, cindex 1
        ],
    )


class TestBosonHamiltonian:
    def test_lhss(self):
        h = make_boson_ham()
        assert h.lhss == LHSS

    def test_max_site(self):
        h = make_boson_ham()
        assert h.max_site == N_BOSON - 1

    def test_num_cindices(self):
        h = make_boson_ham()
        assert h.num_cindices == 2

    def test_number_op(self):
        h = PyBosonHamiltonian(lhss=4, terms=[[("n", [(1.0, 0), (1.0, 1)])]])
        assert h.lhss == 4
        assert h.num_cindices == 1

    def test_bad_op_char_raises(self):
        with pytest.raises(Exception):
            PyBosonHamiltonian(lhss=3, terms=[[("x", [(1.0, 0)])]])

    def test_lhss_too_small_raises(self):
        with pytest.raises(Exception):
            PyBosonHamiltonian(lhss=1, terms=[[("n", [(1.0, 0)])]])


# ---------------------------------------------------------------------------
# PyDitBasis — full
# ---------------------------------------------------------------------------


class TestDitBasisFull:
    def test_full_size(self):
        basis = PyDitBasis.full(n_sites=N_BOSON, lhss=LHSS)
        assert basis.size == LHSS**N_BOSON

    def test_full_size_various_lhss(self):
        for lhss in [2, 3, 4, 5]:
            for n in [1, 2, 3]:
                b = PyDitBasis.full(n_sites=n, lhss=lhss)
                assert b.size == lhss**n

    def test_n_sites(self):
        basis = PyDitBasis.full(n_sites=N_BOSON, lhss=LHSS)
        assert basis.n_sites == N_BOSON

    def test_lhss_property(self):
        basis = PyDitBasis.full(n_sites=N_BOSON, lhss=LHSS)
        assert basis.lhss == LHSS

    def test_state_at_returns_string(self):
        basis = PyDitBasis.full(n_sites=2, lhss=3)
        s = basis.state_at(0)
        assert isinstance(s, str)
        assert len(s) == 2

    def test_full_too_many_bits_raises(self):
        # lhss=3 needs 2 bits/site; 33 sites → 66 bits > 64
        with pytest.raises(Exception):
            PyDitBasis.full(n_sites=33, lhss=3)


# ---------------------------------------------------------------------------
# PyDitBasis — subspace
# ---------------------------------------------------------------------------


class TestDitBasisSubspace:
    def test_subspace_size_at_least_1(self):
        h = make_boson_ham()
        basis = PyDitBasis.subspace(["100"], h)
        assert basis.size >= 1

    def test_subspace_size_at_most_full(self):
        h = make_boson_ham()
        full = PyDitBasis.full(n_sites=N_BOSON, lhss=LHSS)
        sub = PyDitBasis.subspace(["100"], h)
        assert sub.size <= full.size

    def test_str_and_list_seeds_equivalent(self):
        h = make_boson_ham()
        b_str = PyDitBasis.subspace(["100"], h)
        b_list = PyDitBasis.subspace([[1, 0, 0]], h)
        assert b_str.size == b_list.size

    def test_number_conserving_subspace_size(self):
        # n̂ only Hamiltonian doesn't connect states → each seed gives size 1.
        h = PyBosonHamiltonian(lhss=3, terms=[[("n", [(1.0, 0)])]])
        basis = PyDitBasis.subspace(["010"], h)
        assert basis.size == 1

    def test_invalid_seed_char_raises(self):
        h = make_boson_ham()
        with pytest.raises(Exception):
            PyDitBasis.subspace(["9xx"], h)  # invalid for lhss=3

    def test_invalid_seed_value_raises(self):
        h = make_boson_ham()
        with pytest.raises(Exception):
            PyDitBasis.subspace([[3, 0, 0]], h)  # 3 >= lhss=3


# ---------------------------------------------------------------------------
# PyQMatrix — boson
# ---------------------------------------------------------------------------


class TestQMatrixBoson:
    def _make(self, dtype=np.float64):
        h = make_boson_ham()
        basis = PyDitBasis.full(n_sites=N_BOSON, lhss=LHSS)
        return PyQMatrix.build_boson_hamiltonian(h, basis, np.dtype(dtype))

    def test_build_returns_qmatrix(self):
        mat = self._make()
        assert isinstance(mat, PyQMatrix)

    def test_dim_matches_basis(self):
        h = make_boson_ham()
        basis = PyDitBasis.full(n_sites=N_BOSON, lhss=LHSS)
        mat = PyQMatrix.build_boson_hamiltonian(h, basis, np.dtype("float64"))
        assert mat.dim == basis.size

    def test_nnz_positive(self):
        mat = self._make()
        assert mat.nnz > 0

    def test_dot_overwrite(self):
        mat = self._make()
        coeff = np.array([1.0, 1.0], dtype=np.float64)
        v = np.random.default_rng(10).standard_normal(mat.dim)
        out = np.zeros(mat.dim, dtype=np.float64)
        mat.dot(coeff, v, out, True)
        assert not np.allclose(out, 0.0)

    def test_lhss_mismatch_raises(self):
        h = PyBosonHamiltonian(lhss=3, terms=[[("n", [(1.0, 0)])]])
        basis = PyDitBasis.full(n_sites=2, lhss=4)  # lhss mismatch
        with pytest.raises(Exception):
            PyQMatrix.build_boson_hamiltonian(h, basis, np.dtype("float64"))

    def test_number_op_diagonal(self):
        """n̂ on a single site gives a diagonal matrix with occupation eigenvalues."""
        h = PyBosonHamiltonian(lhss=3, terms=[[("n", [(1.0, 0)])]])
        basis = PyDitBasis.full(n_sites=1, lhss=3)
        mat = PyQMatrix.build_boson_hamiltonian(h, basis, np.dtype("float64"))
        coeff = np.array([1.0], dtype=np.float64)
        v = np.ones(mat.dim, dtype=np.float64)
        out = np.zeros(mat.dim, dtype=np.float64)
        mat.dot(coeff, v, out, True)
        # full 1-site lhss=3 basis: states [2,1,0] (descending), so out = [2,1,0]
        np.testing.assert_allclose(sorted(out), [0.0, 1.0, 2.0], atol=1e-12)
