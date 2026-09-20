"""Tests for SpinOperator (spin-S ladder/Sz algebra) — issue #105.

The dense cross-checks build an independent NumPy reference from explicit
spin-S matrices and compare it, element by element, against the matrix the
Rust ``QMatrix.build_spin`` path produces.
"""

import numpy as np
import pytest

from quspin_rs._rs import QMatrix, SpinBasis, SpinOperator

# ---------------------------------------------------------------------------
# NumPy reference: explicit spin-S matrices in the n = 0..lhss-1 encoding
# where n labels the projection m = S - n (so n = 0 is m = +S).
# ---------------------------------------------------------------------------


def spin_matrices(lhss: int):
    """Return ``(S+, S-, Sz)`` as dense ``(lhss, lhss)`` arrays."""
    s = (lhss - 1) / 2
    sz = np.diag([s - n for n in range(lhss)])
    sp = np.zeros((lhss, lhss))
    sm = np.zeros((lhss, lhss))
    for n in range(lhss):
        m = s - n
        if n > 0:
            sp[n - 1, n] = np.sqrt(s * (s + 1) - m * (m + 1))
        if n + 1 < lhss:
            sm[n + 1, n] = np.sqrt(s * (s + 1) - m * (m - 1))
    return sp, sm, sz


def embed(a: np.ndarray, site: int, n_sites: int, lhss: int) -> np.ndarray:
    """Embed a single-site matrix at ``site``; site 0 is the leading factor."""
    out = np.ones((1, 1))
    for k in range(n_sites):
        out = np.kron(out, a if k == site else np.eye(lhss))
    return out


def term_matrix(op_str: str, sites, n_sites: int, lhss: int) -> np.ndarray:
    """Dense matrix for one operator string.

    ``SpinOpEntry`` applies ops right-to-left (element 0 applied last), which
    is exactly the matrix product in left-to-right string order.
    """
    sp, sm, sz = spin_matrices(lhss)
    table = {"+": sp, "-": sm, "z": sz}
    out = np.eye(lhss**n_sites)
    for ch, site in zip(op_str, sites):
        out = out @ embed(table[ch], site, n_sites, lhss)
    return out


def reference_dense(terms, n_sites: int, lhss: int) -> np.ndarray:
    """Dense reference in *kron* ordering (site 0 = most significant digit)."""
    out = np.zeros((lhss**n_sites, lhss**n_sites), dtype=complex)
    for op_str, bonds in terms:
        for bond in bonds:
            coeff, sites = bond[0], bond[1:]
            out += coeff * term_matrix(op_str, sites, n_sites, lhss)
    return out


def library_dense(op: SpinOperator, basis: SpinBasis, coeffs) -> np.ndarray:
    """Dense matrix of ``op`` in the basis' own ordering, column by column.

    Uses the matrix-free ``apply`` path, whose contract is ``out = A @ input``.
    """
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    n = basis.size
    dense = np.zeros((n, n), dtype=complex)
    for j in range(n):
        e = np.zeros(n, dtype=np.complex128)
        e[j] = 1.0
        out = np.zeros(n, dtype=np.complex128)
        op.apply(basis, coeffs, e, out, True)
        dense[:, j] = out
    return dense


def qmatrix_dense(op: SpinOperator, basis: SpinBasis, coeffs) -> np.ndarray:
    """Dense matrix assembled through ``QMatrix.build_spin`` + ``to_csr``."""
    mat = QMatrix.build_spin(op, basis, np.dtype("complex128"))
    indptr, indices, data = mat.to_csr(np.asarray(coeffs, dtype=np.complex128))
    dense = np.zeros((mat.dim, mat.dim), dtype=complex)
    for row in range(mat.dim):
        for pos in range(indptr[row], indptr[row + 1]):
            dense[row, indices[pos]] += data[pos]
    return dense


def basis_permutation(basis: SpinBasis) -> np.ndarray:
    """Map library basis index -> kron-ordering index.

    ``state_at`` returns one digit per site with site 0 leftmost, so reading
    the string as a base-``lhss`` integer gives the kron-ordering index.
    """
    return np.array(
        [int(basis.state_at(i), basis.lhss) for i in range(basis.size)], dtype=int
    )


# ---------------------------------------------------------------------------
# Hamiltonians used across the tests
# ---------------------------------------------------------------------------

N = 3


def heisenberg_terms(n_sites: int):
    """Open spin-S Heisenberg chain: Sz Sz + (S+ S- + S- S+) / 2."""
    bonds = [[1.0, i, i + 1] for i in range(n_sites - 1)]
    half = [[0.5, i, i + 1] for i in range(n_sites - 1)]
    return [("zz", bonds), ("+-", half), ("-+", half)]


def make_heisenberg(n_sites: int, lhss: int) -> SpinOperator:
    return SpinOperator(*[[t] for t in heisenberg_terms(n_sites)], lhss=lhss)


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


class TestSpinOperatorBasics:
    def test_parses_z(self):
        """Regression: SpinOp::from_char had no 'z' arm (issue #105)."""
        op = SpinOperator([("z", [[1.0, 0]])], lhss=3)
        assert op.lhss == 3

    def test_parses_all_op_chars(self):
        op = SpinOperator([("+-z", [[1.0, 0, 1, 2]])], lhss=3)
        assert op.max_site == 2

    def test_max_site(self):
        assert make_heisenberg(N, 3).max_site == N - 1

    def test_num_cindices(self):
        # heisenberg_terms yields three separately-indexed coupling groups
        assert make_heisenberg(N, 3).num_cindices == 3

    def test_lhss(self):
        assert make_heisenberg(N, 5).lhss == 5

    def test_repr(self):
        assert "SpinOperator" in repr(make_heisenberg(N, 3))

    def test_lhss_below_two_rejected(self):
        with pytest.raises(ValueError):
            SpinOperator([("z", [[1.0, 0]])], lhss=1)

    def test_unknown_op_char_rejected(self):
        with pytest.raises(ValueError):
            SpinOperator([("x", [[1.0, 0]])], lhss=3)

    def test_op_str_site_count_mismatch_rejected(self):
        with pytest.raises(ValueError):
            SpinOperator([("+-", [[1.0, 0]])], lhss=3)


# ---------------------------------------------------------------------------
# Matrix elements vs. an independent NumPy reference
# ---------------------------------------------------------------------------


class TestSpinOperatorMatrixElements:
    @pytest.mark.parametrize("lhss", [2, 3, 4])
    def test_single_site_sz_diagonal(self, lhss: int):
        """Sz on site 0 is diagonal with entries m = S - n."""
        op = SpinOperator([("z", [[1.0, 0]])], lhss=lhss)
        basis = SpinBasis.full(2, lhss)
        dense = library_dense(op, basis, [1.0 + 0j])
        s = (lhss - 1) / 2
        for i in range(basis.size):
            n_site0 = int(basis.state_at(i)[0])
            assert dense[i, i] == pytest.approx(s - n_site0)
        assert np.allclose(dense - np.diag(np.diag(dense)), 0.0)

    @pytest.mark.parametrize("lhss", [2, 3, 4])
    def test_heisenberg_matches_numpy_reference(self, lhss: int):
        terms = heisenberg_terms(N)
        op = make_heisenberg(N, lhss)
        basis = SpinBasis.full(N, lhss)

        got = library_dense(op, basis, np.ones(len(terms), dtype=np.complex128))

        perm = basis_permutation(basis)
        want = reference_dense(terms, N, lhss)[np.ix_(perm, perm)]

        assert np.max(np.abs(got - want)) < 1e-12

    @pytest.mark.parametrize("lhss", [2, 3])
    def test_ladder_pair_matches_numpy_reference(self, lhss: int):
        """S+_0 S-_1 alone — off-diagonal, non-Hermitian, distinct amplitudes."""
        terms = [("+-", [[1.0, 0, 1]])]
        op = SpinOperator(list(terms), lhss=lhss)
        basis = SpinBasis.full(2, lhss)

        got = library_dense(op, basis, [1.0 + 0j])
        perm = basis_permutation(basis)
        want = reference_dense(terms, 2, lhss)[np.ix_(perm, perm)]

        assert np.max(np.abs(got - want)) < 1e-12

    def test_complex_coefficients(self):
        terms = [("+-", [[1.0, 0, 1]]), ("-+", [[1.0, 0, 1]])]
        op = SpinOperator([terms[0]], [terms[1]], lhss=3)
        basis = SpinBasis.full(2, 3)
        coeffs = np.array([0.5 - 0.25j, 0.5 + 0.25j], dtype=np.complex128)

        got = library_dense(op, basis, coeffs)
        perm = basis_permutation(basis)
        want = np.zeros_like(got)
        for c, t in zip(coeffs, terms):
            want += c * reference_dense([t], 2, 3)
        want = want[np.ix_(perm, perm)]

        assert np.max(np.abs(got - want)) < 1e-12

    @pytest.mark.parametrize("lhss", [2, 3, 4])
    def test_heisenberg_spectrum(self, lhss: int):
        """Eigenvalues agree with the NumPy reference (ordering-independent)."""
        terms = heisenberg_terms(N)
        op = make_heisenberg(N, lhss)
        basis = SpinBasis.full(N, lhss)

        got = np.linalg.eigvalsh(library_dense(op, basis, np.ones(len(terms))))
        want = np.linalg.eigvalsh(reference_dense(terms, N, lhss))

        assert np.allclose(np.sort(got), np.sort(want), atol=1e-10)

    def test_heisenberg_is_hermitian(self):
        op = make_heisenberg(N, 3)
        dense = library_dense(op, SpinBasis.full(N, 3), np.ones(3))
        assert np.max(np.abs(dense - dense.conj().T)) < 1e-12


# ---------------------------------------------------------------------------
# QMatrix.build_spin
# ---------------------------------------------------------------------------


class TestQMatrixBuildSpin:
    def test_dim_matches_basis(self):
        basis = SpinBasis.full(N, 3)
        mat = QMatrix.build_spin(make_heisenberg(N, 3), basis, np.dtype("float64"))
        assert mat.dim == basis.size

    def test_nnz_positive(self):
        mat = QMatrix.build_spin(
            make_heisenberg(N, 3), SpinBasis.full(N, 3), np.dtype("float64")
        )
        assert mat.nnz > 0

    def test_num_coeff_matches_operator(self):
        mat = QMatrix.build_spin(
            make_heisenberg(N, 3), SpinBasis.full(N, 3), np.dtype("float64")
        )
        assert mat.num_coeff == 3

    def test_all_supported_dtypes(self):
        basis = SpinBasis.full(N, 3)
        for dt in ["int8", "int16", "float32", "float64", "complex64", "complex128"]:
            mat = QMatrix.build_spin(make_heisenberg(N, 3), basis, np.dtype(dt))
            assert mat.dim == basis.size

    def test_hermitian_hamiltonian_matches_apply(self):
        op = make_heisenberg(N, 3)
        basis = SpinBasis.full(N, 3)
        coeffs = np.ones(3, dtype=np.complex128)
        assert (
            np.max(
                np.abs(
                    qmatrix_dense(op, basis, coeffs) - library_dense(op, basis, coeffs)
                )
            )
            < 1e-12
        )

    @pytest.mark.xfail(
        reason="issue #121 — QMatrix.build_* stores the transpose: to_csr/matvec "
        "give A.T while op.apply gives A. Affects every operator type, not just "
        "spin; invisible for the real-symmetric Hamiltonians the other tests use.",
        strict=True,
    )
    def test_non_hermitian_operator_matches_apply(self):
        op = SpinOperator([("+-", [[1.0, 0, 1]])], lhss=3)
        basis = SpinBasis.full(2, 3)
        coeffs = np.array([1.0 + 0j])
        assert (
            np.max(
                np.abs(
                    qmatrix_dense(op, basis, coeffs) - library_dense(op, basis, coeffs)
                )
            )
            < 1e-12
        )


# ---------------------------------------------------------------------------
# apply() — matrix-free path agrees with the assembled matrix
# ---------------------------------------------------------------------------


class TestSpinOperatorApply:
    @pytest.mark.parametrize("lhss", [2, 3])
    def test_apply_matches_dense(self, lhss: int):
        op = make_heisenberg(N, lhss)
        basis = SpinBasis.full(N, lhss)
        coeffs = np.ones(3, dtype=np.complex128)
        dense = library_dense(op, basis, coeffs)

        rng = np.random.default_rng(0)
        psi = rng.normal(size=basis.size) + 1j * rng.normal(size=basis.size)
        psi = psi.astype(np.complex128)
        out = np.zeros(basis.size, dtype=np.complex128)

        op.apply(basis, coeffs, psi, out, True)
        assert np.max(np.abs(out - dense @ psi)) < 1e-12

    def test_apply_accumulates_when_not_overwriting(self):
        op = make_heisenberg(N, 3)
        basis = SpinBasis.full(N, 3)
        coeffs = np.ones(3, dtype=np.complex128)
        psi = np.ones(basis.size, dtype=np.complex128)

        once = np.zeros(basis.size, dtype=np.complex128)
        op.apply(basis, coeffs, psi, once, True)

        twice = np.zeros(basis.size, dtype=np.complex128)
        op.apply(basis, coeffs, psi, twice, True)
        op.apply(basis, coeffs, psi, twice, False)

        assert np.max(np.abs(twice - 2 * once)) < 1e-12


# ---------------------------------------------------------------------------
# SpinBasis BFS routing
# ---------------------------------------------------------------------------


class TestSpinBasisWithSpinOperator:
    def test_subspace_conserves_total_magnetization(self):
        """Heisenberg hopping conserves sum(n); count the compositions."""
        op = make_heisenberg(N, 3)
        # sum(n) == 1 over 3 sites with n in 0..2  ->  3 states
        assert SpinBasis.subspace(N, op, ["100"], 3).size == 3
        # sum(n) == 2  ->  200 110 020 101 011 002  ->  6 states
        assert SpinBasis.subspace(N, op, ["200"], 3).size == 6

    def test_subspace_highest_weight_is_a_singleton(self):
        op = make_heisenberg(N, 3)
        assert SpinBasis.subspace(N, op, ["000"], 3).size == 1

    def test_subspace_spectrum_is_a_subset_of_full(self):
        op = make_heisenberg(N, 3)
        sub = SpinBasis.subspace(N, op, ["100"], 3)
        full = SpinBasis.full(N, 3)
        coeffs = np.ones(3, dtype=np.complex128)

        sub_eigs = np.linalg.eigvalsh(library_dense(op, sub, coeffs))
        full_eigs = np.linalg.eigvalsh(library_dense(op, full, coeffs))

        for e in sub_eigs:
            assert np.min(np.abs(full_eigs - e)) < 1e-10

    def test_bad_ham_type_rejected(self):
        with pytest.raises(TypeError):
            SpinBasis.subspace(N, "not an operator", ["100"], 3)  # type: ignore[arg-type]
