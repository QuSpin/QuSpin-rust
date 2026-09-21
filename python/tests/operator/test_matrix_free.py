"""Tests for the matrix-free (operator, basis) LinearOperator — issue #107.

``op.as_linearoperator(basis, coeffs)`` builds a SciPy-compatible operator
that never assembles a ``QMatrix``.  These tests pin it against the
matrix-free ``apply`` path (which defines ``A``), against an assembled
``QMatrix`` where the two conventions agree, and against SciPy.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as sla

from quspin_rs import Lattice, SymmetryGroup
from quspin_rs._rs import (
    BosonBasis,
    BosonOperator,
    ExpmOp,
    FermionBasis,
    FermionOperator,
    PauliOperator,
    QMatrix,
    SpinBasis,
    SpinOperator,
)

N = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dense_from_matvec(lo) -> np.ndarray:
    """Dense matrix of a linear operator, column by column."""
    n = lo.shape[0]
    return np.column_stack(
        [lo.matvec(np.eye(n, dtype=np.complex128)[:, j].copy()) for j in range(n)]
    )


def dense_from_apply(op, basis, coeffs) -> np.ndarray:
    """Dense matrix from the operator's own ``apply`` — the reference for A."""
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    n = basis.size
    out = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = 1.0
        col = np.zeros(n, dtype=np.complex128)
        op.apply(basis, coeffs, e, col, True)
        out[:, j] = col
    return out


def as_scipy_operator(shape, matvec, dtype, rmatvec=None):
    """Wrap callables in a ``scipy.sparse.linalg.LinearOperator``.

    Goes through ``**kwargs`` because SciPy's type stubs do not model the
    callable-argument constructor form, only the subclass one.
    """
    kwargs = {"shape": shape, "matvec": matvec, "dtype": dtype}
    if rmatvec is not None:
        kwargs["rmatvec"] = rmatvec
    return sla.LinearOperator(**kwargs)  # pyright: ignore[reportCallIssue]


def heisenberg_pauli() -> PauliOperator:
    """XX + YY + ZZ nearest-neighbour chain — Hermitian and real-symmetric."""
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("xx", bonds)], [("yy", bonds)], [("zz", bonds)])


def spin_one_heisenberg() -> SpinOperator:
    bonds = [[1.0, i, i + 1] for i in range(3 - 1)]
    half = [[0.5, i, i + 1] for i in range(3 - 1)]
    return SpinOperator([("zz", bonds)], [("+-", half), ("-+", half)], lhss=3)


# ---------------------------------------------------------------------------
# Basic surface
# ---------------------------------------------------------------------------


class TestMatrixFreeSurface:
    def _lo(self):
        basis = SpinBasis.full(N)
        return heisenberg_pauli().as_linearoperator(
            basis, np.ones(3, dtype=np.complex128)
        )

    def test_shape(self):
        assert self._lo().shape == (2**N, 2**N)

    def test_dim(self):
        assert self._lo().dim == 2**N

    def test_dtype_is_complex128(self):
        assert self._lo().dtype == np.dtype("complex128")

    def test_repr_mentions_matrix_free(self):
        assert "matrix_free=True" in repr(self._lo())

    def test_rejects_wrong_coeff_count(self):
        basis = SpinBasis.full(N)
        with pytest.raises(ValueError):
            heisenberg_pauli().as_linearoperator(basis, np.ones(2, dtype=np.complex128))

    def test_rejects_bad_basis_type(self):
        with pytest.raises(TypeError):
            heisenberg_pauli().as_linearoperator(
                "not a basis",  # type: ignore[arg-type]
                np.ones(3, dtype=np.complex128),
            )

    def test_matvec_rejects_wrong_length(self):
        lo = self._lo()
        with pytest.raises(ValueError):
            lo.matvec(np.ones(lo.shape[0] + 1, dtype=np.complex128))


# ---------------------------------------------------------------------------
# Agreement with apply() and with an assembled QMatrix
# ---------------------------------------------------------------------------


class TestMatrixFreeAgreement:
    def test_matvec_matches_apply(self):
        basis = SpinBasis.full(N)
        op = heisenberg_pauli()
        coeffs = np.array([1.0, 0.5, -0.25], dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)

        got = dense_from_matvec(lo)
        want = dense_from_apply(op, basis, coeffs)
        assert np.max(np.abs(got - want)) < 1e-12

    def test_matvec_matches_assembled_qmatrix_for_hermitian(self):
        """Heisenberg is real-symmetric, so the #121 transpose is invisible."""
        basis = SpinBasis.full(N)
        op = heisenberg_pauli()
        coeffs = np.ones(3, dtype=np.complex128)

        free = dense_from_matvec(op.as_linearoperator(basis, coeffs))
        assembled = dense_from_matvec(
            QMatrix.build_pauli(op, basis, np.dtype("complex128")).as_linearoperator(
                coeffs
            )
        )
        assert np.max(np.abs(free - assembled)) < 1e-12

    def test_matmat_matches_column_wise_matvec(self):
        basis = SpinBasis.full(N)
        lo = heisenberg_pauli().as_linearoperator(
            basis, np.ones(3, dtype=np.complex128)
        )
        n = lo.shape[0]
        rng = np.random.default_rng(1)
        x = (rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))).astype(
            np.complex128
        )
        got = lo.matmat(x)
        for j in range(3):
            assert np.max(np.abs(got[:, j] - lo.matvec(x[:, j].copy()))) < 1e-12

    def test_rmatvec_is_hermitian_adjoint(self):
        basis = SpinBasis.full(3, 3)
        op = spin_one_heisenberg()
        coeffs = np.array([1.0, 0.5 + 0.25j], dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)

        a = dense_from_apply(op, basis, coeffs)
        n = a.shape[0]
        got = np.column_stack(
            [lo.rmatvec(np.eye(n, dtype=np.complex128)[:, j].copy()) for j in range(n)]
        )
        assert np.max(np.abs(got - a.conj().T)) < 1e-12

    def test_matmul_operators(self):
        basis = SpinBasis.full(N)
        lo = heisenberg_pauli().as_linearoperator(
            basis, np.ones(3, dtype=np.complex128)
        )
        n = lo.shape[0]
        rng = np.random.default_rng(2)
        x = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)

        assert np.max(np.abs((lo @ x) - lo.matvec(x))) < 1e-12
        # x @ A is the plain transpose product
        a = dense_from_matvec(lo)
        assert np.max(np.abs((x @ lo) - a.T @ x)) < 1e-12

    def test_rmatmat_matches_dense_adjoint(self):
        basis = SpinBasis.full(3, 3)
        op = spin_one_heisenberg()
        coeffs = np.array([1.0, 0.5 + 0.25j], dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)

        a = dense_from_apply(op, basis, coeffs)
        n = a.shape[0]
        rng = np.random.default_rng(3)
        x = (rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))).astype(
            np.complex128
        )
        assert np.max(np.abs(lo.rmatmat(x) - a.conj().T @ x)) < 1e-12


# ---------------------------------------------------------------------------
# Spectral metadata
# ---------------------------------------------------------------------------


class TestMatrixFreeSpectral:
    @pytest.mark.parametrize("lhss", [2, 3])
    def test_trace_matches_dense(self, lhss: int):
        if lhss == 2:
            basis, op, coeffs = (
                SpinBasis.full(N),
                heisenberg_pauli(),
                np.array([1.0, 0.5, -0.25], dtype=np.complex128),
            )
        else:
            basis, op, coeffs = (
                SpinBasis.full(3, 3),
                spin_one_heisenberg(),
                np.array([1.0, 0.75], dtype=np.complex128),
            )
        lo = op.as_linearoperator(basis, coeffs)
        want = np.trace(dense_from_apply(op, basis, coeffs))
        assert abs(lo.trace() - want) < 1e-10

    @pytest.mark.parametrize("shift", [0.0 + 0j, 1.5 + 0j, 0.25 - 0.75j])
    def test_onenorm_matches_dense_column_norm(self, shift: complex):
        basis = SpinBasis.full(N)
        op = heisenberg_pauli()
        coeffs = np.array([1.0, 0.5, -0.25], dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)

        a = dense_from_apply(op, basis, coeffs)
        want = np.max(np.sum(np.abs(a - shift * np.eye(a.shape[0])), axis=0))
        assert abs(lo.onenorm(shift) - want) < 1e-10

    def test_onenorm_defaults_to_zero_shift(self):
        basis = SpinBasis.full(N)
        lo = heisenberg_pauli().as_linearoperator(
            basis, np.ones(3, dtype=np.complex128)
        )
        assert lo.onenorm() == pytest.approx(lo.onenorm(0.0 + 0j))


# ---------------------------------------------------------------------------
# Every operator / basis family routes correctly
# ---------------------------------------------------------------------------


class TestMatrixFreeCoverage:
    def test_spin_operator_on_spin_basis(self):
        basis = SpinBasis.full(3, 3)
        op = spin_one_heisenberg()
        coeffs = np.array([1.0, 1.0], dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)
        assert (
            np.max(np.abs(dense_from_matvec(lo) - dense_from_apply(op, basis, coeffs)))
            < 1e-12
        )

    def test_boson_operator_on_boson_basis(self):
        n, lhss = 3, 3
        bonds = [[1.0, i, i + 1] for i in range(n - 1)]
        op = BosonOperator([("+-", bonds), ("-+", bonds)], lhss=lhss)
        basis = BosonBasis.full(n, lhss)
        coeffs = np.array([1.0], dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)
        assert (
            np.max(np.abs(dense_from_matvec(lo) - dense_from_apply(op, basis, coeffs)))
            < 1e-12
        )

    def test_fermion_operator_on_fermion_basis(self):
        """FermionBasis takes the BitBasis dispatch root, not GenericBasis."""
        n = 4
        bonds = [[1.0, i, i + 1] for i in range(n - 1)]
        op = FermionOperator([("+-", bonds), ("-+", bonds)])
        basis = FermionBasis.full(n)
        coeffs = np.array([1.0], dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)
        assert lo.shape == (2**n, 2**n)
        assert (
            np.max(np.abs(dense_from_matvec(lo) - dense_from_apply(op, basis, coeffs)))
            < 1e-12
        )

    def test_subspace_basis(self):
        op = heisenberg_pauli()
        basis = SpinBasis.subspace(N, op, ["1100"])
        coeffs = np.ones(3, dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)
        assert lo.shape == (basis.size, basis.size)
        assert (
            np.max(np.abs(dense_from_matvec(lo) - dense_from_apply(op, basis, coeffs)))
            < 1e-12
        )


# ---------------------------------------------------------------------------
# Downstream consumers
# ---------------------------------------------------------------------------


class TestMatrixFreeConsumers:
    def test_expm_matches_qmatrix_backed_expm(self):
        basis = SpinBasis.full(N)
        op = heisenberg_pauli()
        coeffs = np.ones(3, dtype=np.complex128)
        a = -0.37j

        rng = np.random.default_rng(4)
        psi0 = (rng.normal(size=basis.size) + 1j * rng.normal(size=basis.size)).astype(
            np.complex128
        )

        free = psi0.copy()
        ExpmOp(op.as_linearoperator(basis, coeffs), a).worker().apply(free)

        assembled = psi0.copy()
        qop = QMatrix.build_pauli(op, basis, np.dtype("complex128")).as_linearoperator(
            coeffs
        )
        ExpmOp(qop, a).worker().apply(assembled)

        assert np.max(np.abs(free - assembled)) < 1e-10

    def test_expm_matches_scipy_expm_multiply(self):
        basis = SpinBasis.full(N)
        op = heisenberg_pauli()
        coeffs = np.ones(3, dtype=np.complex128)
        a = -0.2j
        lo = op.as_linearoperator(basis, coeffs)

        rng = np.random.default_rng(5)
        psi0 = (rng.normal(size=basis.size) + 1j * rng.normal(size=basis.size)).astype(
            np.complex128
        )

        got = psi0.copy()
        ExpmOp(lo, a).worker().apply(got)
        want = sla.expm_multiply(a * dense_from_matvec(lo), psi0)

        assert np.max(np.abs(got - want)) < 1e-9

    def test_expm_rejects_non_operator(self):
        with pytest.raises(TypeError):
            ExpmOp("not an operator", -1j)  # type: ignore[arg-type]

    def test_scipy_eigsh_accepts_the_operator(self):
        """The whole point: iterative solvers with no assembled matrix."""
        basis = SpinBasis.full(N)
        op = heisenberg_pauli()
        coeffs = np.ones(3, dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)

        wrapped = as_scipy_operator(lo.shape, lo.matvec, lo.dtype, rmatvec=lo.rmatvec)
        got = np.sort(sla.eigsh(wrapped, k=3, which="SA", return_eigenvectors=False))
        want = np.sort(np.linalg.eigvalsh(dense_from_matvec(lo)))[:3]

        assert np.allclose(got, want, atol=1e-8)

    def test_scipy_gmres_solves_against_the_operator(self):
        basis = SpinBasis.full(N)
        op = heisenberg_pauli()
        coeffs = np.ones(3, dtype=np.complex128)
        lo = op.as_linearoperator(basis, coeffs)
        n = lo.shape[0]

        # Shift well away from the spectrum so the system is non-singular.
        a = dense_from_matvec(lo)
        shift = 20.0

        def shifted(v):
            return lo.matvec(np.ascontiguousarray(v, dtype=np.complex128)) + shift * v

        wrapped = as_scipy_operator((n, n), shifted, np.complex128)
        rng = np.random.default_rng(6)
        b = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)

        x, info = sla.gmres(wrapped, b, rtol=1e-10, restart=n)
        assert info == 0
        assert np.max(np.abs((a + shift * np.eye(n)) @ x - b)) < 1e-6


# ---------------------------------------------------------------------------
# Operator/basis compatibility at wrapper-construction time
#
# OperatorOnBasis::new validates up front rather than letting the first
# matvec fail, so a mismatched pair is caught where it is created.
# ---------------------------------------------------------------------------


class TestAsLinearOperatorValidation:
    def test_rejects_lhss_mismatch(self):
        op = SpinOperator([("z", [[1.0, 0]])], lhss=3)
        basis = SpinBasis.full(2, 4)
        with pytest.raises(ValueError, match="lhss"):
            op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))

    def test_rejects_site_past_end_of_lattice(self):
        op = SpinOperator([("z", [[1.0, 7]])], lhss=3)
        basis = SpinBasis.full(2, 3)
        with pytest.raises(ValueError, match="site"):
            op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))

    def test_rejects_pauli_operator_on_a_higher_lhss_basis(self):
        op = PauliOperator([("z", [[1.0, 0]])])
        basis = SpinBasis.full(2, 3)
        with pytest.raises(ValueError, match="lhss"):
            op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))

    @pytest.mark.parametrize("lhss", [2, 3, 4])
    def test_accepts_matching_lhss(self, lhss: int):
        op = SpinOperator([("z", [[1.0, 0]])], lhss=lhss)
        basis = SpinBasis.full(2, lhss)
        lo = op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))
        assert lo.shape == (lhss**2, lhss**2)

    def test_accepts_highest_valid_site(self):
        op = SpinOperator([("z", [[1.0, 1]])], lhss=3)
        basis = SpinBasis.full(2, 3)
        assert op.as_linearoperator(basis, np.ones(1, dtype=np.complex128)).shape == (
            9,
            9,
        )

    def test_matching_lhss_gives_correct_sz_eigenvalues(self):
        """The values the unvalidated path used to get wrong."""
        op = SpinOperator([("z", [[1.0, 0]])], lhss=4)  # spin-3/2
        lo = op.as_linearoperator(SpinBasis.full(1, 4), np.ones(1, dtype=np.complex128))
        diag = np.real(np.diag(dense_from_matvec(lo)))
        assert sorted(np.round(diag, 6)) == [-1.5, -0.5, 0.5, 1.5]


class TestSymmetricBasisComplexCharacters:
    """Matrix-free operators on a momentum sector with complex characters.

    Every other test in this file uses a real-symmetric Hamiltonian on a
    non-symmetric basis, where the group character is identically 1.  A stray
    complex conjugation in the ``SymBasis`` projection is invisible there but
    zeroes out the whole sector for any character other than ``chi = +-1``.
    """

    @staticmethod
    def _sector(L: int, k: int):
        """Single-magnon momentum-k sector of an L-site XX+YY ring."""
        op = PauliOperator(
            [("XX", [[1.0, i, (i + 1) % L] for i in range(L)])],
            [("YY", [[1.0, i, (i + 1) % L] for i in range(L)])],
        )
        group = SymmetryGroup(n_sites=L, lhss=2)
        group.add_cyclic(Lattice([(i + 1) % L for i in range(L)]), k=k)
        basis = SpinBasis.symmetric(group, op, ["1" + "0" * (L - 1)])
        return op, basis

    @pytest.mark.parametrize("k", range(6))
    def test_single_magnon_dispersion(self, k: int):
        """E(k) = 4*cos(2*pi*k/L), exactly — no convention ambiguity.

        XX + YY = 2*(S+S- + S-S+), so the hopping amplitude is 2 per bond and
        a single magnon on a ring disperses as 2*2*cos(k).
        """
        L = 6
        op, basis = self._sector(L, k)
        assert basis.size == 1, f"k={k} sector should hold exactly one state"

        lo = op.as_linearoperator(basis, np.ones(2, dtype=np.complex128))
        got = lo.matvec(np.ones(1, dtype=np.complex128))[0]
        want = 4.0 * np.cos(2.0 * np.pi * k / L)

        assert got == pytest.approx(
            want, abs=1e-10
        ), f"k={k}: matrix-free gave {got}, expected {want}"

    @pytest.mark.parametrize("k", range(6))
    def test_matches_assembled_qmatrix(self, k: int):
        """The matrix-free and assembled paths must agree on the sector.

        The sector is 1x1 here, so the #121 transpose convention is moot and
        the two paths are directly comparable.
        """
        op, basis = self._sector(6, k)
        coeffs = np.ones(2, dtype=np.complex128)

        free = op.as_linearoperator(basis, coeffs).matvec(
            np.ones(1, dtype=np.complex128)
        )[0]
        assembled = (
            QMatrix.build_pauli(op, basis, np.dtype("complex128"))
            .as_linearoperator(coeffs)
            .matvec(np.ones(1, dtype=np.complex128))[0]
        )

        assert free == pytest.approx(assembled, abs=1e-10)

    def test_trace_is_not_silently_zero(self):
        """`trace` runs through the same projection as `matvec`."""
        L, k = 6, 1
        op = PauliOperator([("ZZ", [[1.0, i, (i + 1) % L] for i in range(L)])])
        group = SymmetryGroup(n_sites=L, lhss=2)
        group.add_cyclic(Lattice([(i + 1) % L for i in range(L)]), k=k)
        basis = SpinBasis.symmetric(group, op, ["100000"])

        lo = op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))
        dense = dense_from_matvec(lo)
        assert np.trace(dense) == pytest.approx(lo.trace(), abs=1e-10)
        assert abs(lo.trace()) > 1e-10, "ZZ has a non-zero trace in every sector"
