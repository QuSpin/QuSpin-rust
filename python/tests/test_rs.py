"""Tests for the quspin_rs._rs PyO3 extension (new API)."""

import cmath
import math

import numpy as np
import pytest

from quspin_rs import Lattice, SymmetryGroup
from quspin_rs._rs import (
    BosonBasis,
    BosonOperator,
    ExpmOp,
    ExpmWorker,
    ExpmWorker2,
    FermionBasis,
    FermionOperator,
    Hamiltonian,
    PauliOperator,
    QMatrix,
    QMatrixLinearOperator,
    SchrodingerEq,
    SpinBasis,
    Static,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 4  # number of sites


def make_pauli_op() -> PauliOperator:
    """XX + ZZ nearest-neighbour Hamiltonian on N sites."""
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_spin_basis_full() -> SpinBasis:
    return SpinBasis.full(N)


# ---------------------------------------------------------------------------
# SpinBasis
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PauliOperator
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# QMatrix — build and properties
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# QMatrix — to_csr
# ---------------------------------------------------------------------------


class TestQMatrixCsr:
    def test_to_csr_shapes(self):
        mat = QMatrix.build_pauli(
            make_pauli_op(), make_spin_basis_full(), np.dtype("complex128")
        )
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        indptr, indices, data = mat.to_csr(coeff)
        n = mat.dim
        assert indptr.shape == (n + 1,)
        assert indices.shape == data.shape
        assert indptr[-1] == len(data)

    def test_to_csr_symmetry(self):
        """XX + ZZ is Hermitian: H[i,j] == conj(H[j,i])."""
        mat = QMatrix.build_pauli(
            make_pauli_op(), make_spin_basis_full(), np.dtype("complex128")
        )
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        indptr, indices, data = mat.to_csr(coeff)
        n = mat.dim
        # Build dense matrix from CSR manually
        dense = np.zeros((n, n), dtype=np.complex128)
        for row in range(n):
            for pos in range(indptr[row], indptr[row + 1]):
                dense[row, indices[pos]] += data[pos]
        diff = dense - dense.conj().T
        assert np.max(np.abs(diff)) < 1e-12


# ---------------------------------------------------------------------------
# QMatrix — dot_many
# ---------------------------------------------------------------------------


class TestQMatrixDotMany:
    def _make(self):
        return QMatrix.build_pauli(
            make_pauli_op(), make_spin_basis_full(), np.dtype("complex128")
        )

    def test_dot_many_overwrite(self):
        mat = self._make()
        n, k = mat.dim, 3
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(0).standard_normal((n, k)) + 0j
        out = np.zeros((n, k), dtype=np.complex128)
        mat.dot_many(coeff, inp, out, True)
        assert not np.allclose(out, 0.0)

    def test_dot_many_accumulate(self):
        mat = self._make()
        n, k = mat.dim, 2
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(1).standard_normal((n, k)) + 0j
        out1 = np.zeros((n, k), dtype=np.complex128)
        out2 = np.zeros((n, k), dtype=np.complex128)
        mat.dot_many(coeff, inp, out1, True)
        mat.dot_many(coeff, inp, out2, True)
        mat.dot_many(coeff, inp, out2, False)
        np.testing.assert_allclose(out2, 2 * out1, atol=1e-12)

    def test_dot_many_equals_transpose_for_hermitian(self):
        """For a Hermitian matrix, H·v == H^†·v == H^T·conj(v)."""
        mat = self._make()
        n, k = mat.dim, 2
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(2).standard_normal((n, k)) + 0j
        out_fwd = np.zeros((n, k), dtype=np.complex128)
        out_trans = np.zeros((n, k), dtype=np.complex128)
        mat.dot_many(coeff, inp, out_fwd, True)
        mat.dot_transpose_many(coeff, inp.conj(), out_trans, True)
        np.testing.assert_allclose(out_fwd, out_trans.conj(), atol=1e-12)


# ---------------------------------------------------------------------------
# QMatrix — arithmetic
# ---------------------------------------------------------------------------


class TestQMatrixArithmetic:
    def _make(self):
        return QMatrix.build_pauli(
            make_pauli_op(), make_spin_basis_full(), np.dtype("complex128")
        )

    def test_add_nnz_at_least_original(self):
        mat = self._make()
        assert (mat + mat).nnz >= mat.nnz

    def test_add_dot_is_double(self):
        mat = self._make()
        doubled = mat + mat
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(3).standard_normal((mat.dim, 1)) + 0j
        out1 = np.zeros((mat.dim, 1), dtype=np.complex128)
        out2 = np.zeros((mat.dim, 1), dtype=np.complex128)
        mat.dot_many(coeff, inp, out1, True)
        doubled.dot_many(coeff, inp, out2, True)
        np.testing.assert_allclose(out2, 2 * out1, atol=1e-12)


# ---------------------------------------------------------------------------
# BosonBasis / BosonOperator
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FermionBasis / FermionOperator
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------


class TestHamiltonian:
    def _make_static(self):
        """Static Hamiltonian: XX+ZZ — both cindices are Static."""
        op = make_pauli_op()
        basis = make_spin_basis_full()
        mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
        return Hamiltonian(mat, [Static(), Static()])

    def test_dim(self):
        h = self._make_static()
        assert h.dim == make_spin_basis_full().size

    def test_num_coeff(self):
        h = self._make_static()
        assert h.num_coeff == 2  # XX (cindex 0) + ZZ (cindex 1)

    def test_to_csr_matches_qmatrix_csr(self):
        op = make_pauli_op()
        basis = make_spin_basis_full()
        mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
        ham = Hamiltonian(mat, [Static(), Static()])
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        ip, ii, id_ = mat.to_csr(coeff)
        hp, hi, hd = ham.to_csr(0.0)
        np.testing.assert_array_equal(ip, hp)
        np.testing.assert_array_equal(ii, hi)
        np.testing.assert_allclose(id_, hd, atol=1e-14)

    def test_dot_matches_qmatrix_dot_many(self):
        op = make_pauli_op()
        basis = make_spin_basis_full()
        mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
        ham = Hamiltonian(mat, [Static(), Static()])
        n = mat.dim
        coeff = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
        inp = np.random.default_rng(7).standard_normal((n, 2)) + 0j
        out_q = np.zeros((n, 2), dtype=np.complex128)
        out_h = np.zeros((n, 2), dtype=np.complex128)
        mat.dot_many(coeff, inp, out_q, True)
        ham.dot_many(0.0, inp, out_h, True)
        np.testing.assert_allclose(out_q, out_h, atol=1e-14)

    def test_time_dependent_coeff(self):
        """Scaling coupling by cos(t) should give cos(t) * static result."""
        op = PauliOperator([("ZZ", [[1.0, 0, 1]])], [("ZZ", [[1.0, 0, 1]])])
        basis = SpinBasis.full(2)
        mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
        ham = Hamiltonian(mat, [Static(), lambda t: math.cos(t) + 0j])
        n = mat.dim
        inp = np.random.default_rng(8).standard_normal((n, 1)) + 0j
        out_t0 = np.zeros((n, 1), dtype=np.complex128)
        out_pi4 = np.zeros((n, 1), dtype=np.complex128)
        ham.dot_many(0.0, inp, out_t0, True)  # cos(0)=1, both cindices weight 1
        ham.dot_many(math.pi / 4, inp, out_pi4, True)  # cindex 1 weight = cos(pi/4)
        # At t=0: result = (1 + 1) * ZZ·v = 2 * ZZ·v
        # At t=pi/4: result = (1 + cos(pi/4)) * ZZ·v
        ratio = np.linalg.norm(out_pi4) / np.linalg.norm(out_t0)
        expected_ratio = (1.0 + math.cos(math.pi / 4)) / 2.0
        assert abs(ratio - expected_ratio) < 1e-12


# ---------------------------------------------------------------------------
# SchrodingerEq — Pauli X on 1 site
# ---------------------------------------------------------------------------


class TestSchrodingerEq:
    def _pauli_x_eq(self) -> SchrodingerEq:
        """H = X on a 1-site spin-1/2 basis."""
        op = PauliOperator([("X", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        ham = Hamiltonian(mat, [Static()])
        return SchrodingerEq(ham)

    def test_dim(self):
        assert self._pauli_x_eq().dim == 2

    def test_integrate_returns_correct_shape(self):
        eq = self._pauli_x_eq()
        y0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        yf = eq.integrate(0.0, math.pi / 2, y0)
        assert yf.shape == (2,)

    def test_integrate_pauli_x_half_pi(self):
        """Under H=X, |0> → cos(t)|0> - i sin(t)|1>.
        At t=pi/2: yf ≈ [0, -i] = [0, -i*1]."""
        eq = self._pauli_x_eq()
        y0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        yf = eq.integrate(0.0, math.pi / 2, y0, rtol=1e-10, atol=1e-12)
        assert abs(yf[0]) < 1e-6, f"expected yf[0]≈0, got {yf[0]}"
        assert abs(yf[1] - (-1j)) < 1e-6, f"expected yf[1]≈-i, got {yf[1]}"

    def test_integrate_norm_conserved(self):
        eq = self._pauli_x_eq()
        y0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        yf = eq.integrate(0.0, 1.0, y0)
        assert abs(np.linalg.norm(yf) - 1.0) < 1e-8

    def test_integrate_dense_shapes(self):
        eq = self._pauli_x_eq()
        y0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        times, states = eq.integrate_dense(0.0, 1.0, y0)
        assert times.ndim == 1
        assert states.ndim == 2
        assert states.shape[1] == 2
        assert states.shape[0] == len(times)


# ---------------------------------------------------------------------------
# Symmetric basis — translation symmetry integration tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Operator apply / apply_and_project_to
# ---------------------------------------------------------------------------


class TestOperatorApply:
    """Tests for the matrix-free apply and apply_and_project_to methods."""

    @staticmethod
    def _xx_op():
        return PauliOperator([("xx", [[1.0, 0, 1]])])

    @staticmethod
    def _zz_op():
        return PauliOperator([("zz", [[1.0, 0, 1]])])

    @staticmethod
    def _sp_op():
        """S+ (creation) on site 0."""
        return PauliOperator([("+", [[1.0, 0]])])

    @staticmethod
    def _sz0_basis(xx_op):
        """1-particle subspace of 2 sites: {|01⟩, |10⟩}."""
        return SpinBasis.subspace(2, xx_op, ["01"])

    @staticmethod
    def _sz1_basis(zz_op):
        """2-particle subspace of 2 sites: {|11⟩}."""
        return SpinBasis.subspace(2, zz_op, ["11"])

    def test_apply_same_space_matches_hamiltonian_dot(self):
        """apply() should agree with Hamiltonian.dot for the same operator."""
        xx = self._xx_op()
        sz0 = self._sz0_basis(xx)

        mat = QMatrix.build_pauli(xx, sz0, np.dtype("complex128"))
        ham = Hamiltonian(mat, [Static()])

        psi = np.array([1.0 + 0j, 0.0 + 0j])
        coeffs = np.array([1.0 + 0j])

        # Hamiltonian.dot
        out_ham = np.zeros(2, dtype=complex)
        ham.dot(0.0, psi, out_ham, True)

        # Operator.apply
        out_apply = np.zeros(2, dtype=complex)
        xx.apply(sz0, coeffs, psi, out_apply)

        np.testing.assert_allclose(out_apply, out_ham, atol=1e-12)

    def test_apply_and_project_to_cross_sector(self):
        """S+ maps |01⟩ from Sz=0 to |11⟩ in Sz=1."""
        xx = self._xx_op()
        zz = self._zz_op()
        sp = self._sp_op()

        sz0 = self._sz0_basis(xx)
        sz1 = self._sz1_basis(zz)

        # sz0 states sorted ascending: state_at(0)="10", state_at(1)="01"
        # S+_0 on "01" (site 0 = 0) → "11"; S+_0 on "10" (site 0 = 1) → 0
        # psi = [0, 1] selects state_at(1) = "01"
        psi = np.array([0.0 + 0j, 1.0 + 0j])
        coeffs = np.array([1.0 + 0j])
        out = np.zeros(1, dtype=complex)

        sp.apply_and_project_to(sz0, sz1, coeffs, psi, out)
        np.testing.assert_allclose(out, [1.0 + 0j], atol=1e-12)

    def test_apply_and_project_to_annihilation_gives_zero(self):
        """S+_0 on state_at(0) = '10' (site 0 already occupied) gives 0."""
        xx = self._xx_op()
        zz = self._zz_op()
        sp = self._sp_op()

        sz0 = self._sz0_basis(xx)
        sz1 = self._sz1_basis(zz)

        psi = np.array([1.0 + 0j, 0.0 + 0j])
        coeffs = np.array([1.0 + 0j])
        out = np.zeros(1, dtype=complex)

        sp.apply_and_project_to(sz0, sz1, coeffs, psi, out)
        np.testing.assert_allclose(out, [0.0 + 0j], atol=1e-12)

    def test_apply_and_project_to_full_space(self):
        """Project XX result from subspace into full space."""
        xx = self._xx_op()
        sz0 = self._sz0_basis(xx)
        full = SpinBasis.full(2)

        # psi = state_at(0) in sz0
        psi = np.array([1.0 + 0j, 0.0 + 0j])
        coeffs = np.array([1.0 + 0j])
        out = np.zeros(full.size, dtype=complex)

        xx.apply_and_project_to(sz0, full, coeffs, psi, out)

        # Exactly one entry should be nonzero
        assert np.count_nonzero(np.abs(out) > 1e-12) == 1

    def test_apply_overwrite_false_accumulates(self):
        """overwrite=False should add to existing output."""
        xx = self._xx_op()
        sz0 = self._sz0_basis(xx)

        psi = np.array([1.0 + 0j, 0.0 + 0j])
        coeffs = np.array([1.0 + 0j])
        out = np.array([5.0 + 0j, 5.0 + 0j])

        xx.apply(sz0, coeffs, psi, out, overwrite=False)

        # XX swaps the two states, so out[1] += 1 → 6
        assert abs(out[1] - 6.0) < 1e-12

    def test_apply_coeffs_scale(self):
        """coeffs should scale the result."""
        xx = self._xx_op()
        sz0 = self._sz0_basis(xx)

        psi = np.array([1.0 + 0j, 0.0 + 0j])
        coeffs = np.array([3.0 + 1j])
        out = np.zeros(2, dtype=complex)

        xx.apply(sz0, coeffs, psi, out)
        np.testing.assert_allclose(out[1], 3.0 + 1j, atol=1e-12)


# ---------------------------------------------------------------------------
# QMatrix.as_linearoperator / Hamiltonian.as_linearoperator + ExpmOp
# ---------------------------------------------------------------------------


class TestExpmOpAndWorker:
    """Tests for the new linear-operator + cached expm path."""

    def _diagonal_ham(self) -> Hamiltonian:
        """H = diag(1, -1) via Z on a 1-site spin-1/2 basis."""
        op = PauliOperator([("Z", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        return Hamiltonian(mat, [Static()])

    def _xx_ham(self) -> Hamiltonian:
        """H = X_0 X_1 on 2 spin-1/2 sites."""
        op = PauliOperator([("XX", [[1.0, 0, 1]])])
        basis = SpinBasis.full(2)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        return Hamiltonian(mat, [Static()])

    @staticmethod
    def _ref_expm_z(a: complex, f: np.ndarray) -> np.ndarray:
        scale = np.array([cmath.exp(a), cmath.exp(-a)], dtype=np.complex128)
        if f.ndim == 1:
            return scale * f
        return scale[:, None] * f

    # -- as_linearoperator ----------------------------------------------------

    def test_qmatrix_as_linearoperator_returns_correct_type(self):
        op = PauliOperator([("Z", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        coeffs = np.array([1.0 + 0j], dtype=np.complex128)
        qop = mat.as_linearoperator(coeffs)
        assert isinstance(qop, QMatrixLinearOperator)
        assert qop.dim == 2
        assert qop.num_coeff == 1

    def test_qmatrix_as_linearoperator_wrong_coeffs_size(self):
        op = PauliOperator([("Z", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        bad = np.zeros(5, dtype=np.complex128)
        with pytest.raises(Exception):
            mat.as_linearoperator(bad)

    def test_hamiltonian_as_linearoperator_evaluates_at_time(self):
        """Time-dependent coefficient: H(t) = sin(t) * Z."""

        def f(t: float) -> complex:
            return complex(np.sin(t))

        op = PauliOperator([("Z", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        ham = Hamiltonian(mat, [f])
        qop_t0 = ham.as_linearoperator(0.0)
        qop_t1 = ham.as_linearoperator(np.pi / 2)
        assert qop_t0.dim == 2 == qop_t1.dim
        # The coefficient is internal — verify indirectly through ExpmOp:
        expm_op_t1 = ExpmOp(qop_t1, complex(-0.5, 0.0))
        worker = expm_op_t1.worker(1)
        psi = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        worker.apply(psi)
        # H(π/2) = Z, exp(-0.5·Z) on |0⟩ = exp(-0.5)·|0⟩
        np.testing.assert_allclose(psi[0], cmath.exp(-0.5), atol=1e-10)
        np.testing.assert_allclose(psi[1], 0.0, atol=1e-10)

    # -- ExpmOp construction --------------------------------------------------

    def test_expm_op_dim_and_a(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -1.0))
        assert eo.dim == 2
        assert eo.a == complex(0.0, -1.0)

    # -- worker(n_vec) dispatches to the right type ---------------------------

    def test_worker_n_vec_1_returns_1d_worker(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -math.pi / 4))
        w = eo.worker(1)
        assert isinstance(w, ExpmWorker)

    def test_worker_n_vec_gt1_returns_2d_worker(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -math.pi / 4))
        w = eo.worker(3)
        assert isinstance(w, ExpmWorker2)
        assert w.n_vec == 3

    def test_worker_n_vec_zero_raises(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -1.0))
        with pytest.raises(Exception):
            eo.worker(0)

    # -- 1-D apply correctness ------------------------------------------------

    def test_worker_apply_diagonal_matches_analytic(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        a = -1j * math.pi / 4
        eo = ExpmOp(qop, a)
        f = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128) / math.sqrt(2)
        expected = self._ref_expm_z(a, f.copy())
        eo.worker(1).apply(f)
        np.testing.assert_allclose(f, expected, atol=1e-10)

    def test_worker_apply_xx_matches_eigendecomposition(self):
        """Cross-check: ExpmOp.worker(1).apply == exp(a·H) via eigendecomposition."""
        ham = self._xx_ham()
        a = -1j * math.pi / 3
        H = ham.to_dense(0.0)
        # H is Hermitian; reconstruct exp(a·H) from its eigendecomposition.
        vals, vecs = np.linalg.eigh(H)
        ref_expm = (vecs * np.exp(a * vals)) @ vecs.conj().T

        rng = np.random.default_rng(7)
        f = (rng.standard_normal(4) + 1j * rng.standard_normal(4)).astype(np.complex128)
        expected = ref_expm @ f

        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        eo.worker(1).apply(f)

        np.testing.assert_allclose(f, expected, atol=1e-10)

    def test_worker_reuse_same_result(self):
        """Calling apply twice with a reset input should give the same answer."""
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        worker = eo.worker(1)
        f = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        worker.apply(f)
        first = f.copy()
        f[:] = [1.0 + 0j, 0.0 + 0j]
        worker.apply(f)
        np.testing.assert_allclose(f, first, atol=1e-12)

    def test_worker_apply_wrong_shape_raises(self):
        ham = self._diagonal_ham()
        eo = ExpmOp(ham.as_linearoperator(0.0), -1j)
        worker = eo.worker(1)
        f = np.zeros(5, dtype=np.complex128)
        with pytest.raises(Exception):
            worker.apply(f)

    # -- 2-D apply correctness ------------------------------------------------

    def test_worker2_apply_diagonal_matches_analytic(self):
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        rng = np.random.default_rng(42)
        F = (rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))).astype(
            np.complex128
        )
        expected = self._ref_expm_z(a, F.copy())
        eo.worker(3).apply(F)
        np.testing.assert_allclose(F, expected, atol=1e-10)

    def test_worker2_apply_with_fewer_columns_than_capacity(self):
        """A worker with n_vec=4 should accept input with k=2 columns."""
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        rng = np.random.default_rng(0)
        F = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))).astype(
            np.complex128
        )
        expected = self._ref_expm_z(a, F.copy())
        eo.worker(4).apply(F)
        np.testing.assert_allclose(F, expected, atol=1e-10)

    def test_worker2_apply_wrong_dim_raises(self):
        ham = self._diagonal_ham()
        eo = ExpmOp(ham.as_linearoperator(0.0), -1j)
        worker = eo.worker(2)
        F = np.zeros((5, 2), dtype=np.complex128)
        with pytest.raises(Exception):
            worker.apply(F)

    def test_worker2_too_many_columns_raises(self):
        ham = self._diagonal_ham()
        eo = ExpmOp(ham.as_linearoperator(0.0), -1j)
        worker = eo.worker(2)  # capacity = 2
        F = np.zeros((2, 3), dtype=np.complex128)
        with pytest.raises(Exception):
            worker.apply(F)

    # -- user-supplied work buffer --------------------------------------------

    def test_worker_with_user_work_1d(self):
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        work = np.zeros(2 * eo.dim, dtype=np.complex128)  # length 4
        worker = eo.worker(1, work=work)
        f = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        worker.apply(f)
        # Ground truth via plain worker
        f_ref = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        eo.worker(1).apply(f_ref)
        np.testing.assert_allclose(f, f_ref, atol=1e-12)

    def test_worker_with_user_work_2d(self):
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        work = np.zeros((2 * eo.dim, 3), dtype=np.complex128)
        worker = eo.worker(3, work=work)
        F = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.complex128)
        F_ref = F.copy()
        worker.apply(F)
        eo.worker(3).apply(F_ref)
        np.testing.assert_allclose(F, F_ref, atol=1e-12)

    def test_worker_user_work_too_small_raises(self):
        ham = self._diagonal_ham()
        eo = ExpmOp(ham.as_linearoperator(0.0), -1j)
        work = np.zeros(1, dtype=np.complex128)  # too small (need 4)
        with pytest.raises(Exception):
            eo.worker(1, work=work)
