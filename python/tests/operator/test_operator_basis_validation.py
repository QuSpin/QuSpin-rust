"""Operator/basis compatibility and cindex-hole validation.

Regression tests for the review findings on PR #124:

* An operator whose ``lhss`` or ``max_site`` disagrees with the basis used to
  be accepted, then silently produced wrong matrix elements — a spin-1 ``Sz``
  on a spin-3/2 basis returned ``{1, 0, -1, -2}`` instead of
  ``{±1.5, ±0.5}``, and an out-of-range site folded back in as a phantom
  diagonal term.
* An empty ``*terms`` group left a hole in the cindex sequence while
  ``num_cindices`` counted only the cindices actually present, so
  ``coeffs[cindex]`` ran off the end of the coefficient slice and panicked
  inside ``apply``.
"""

import numpy as np
import pytest

from quspin_rs._rs import (
    BondOperator,
    BosonBasis,
    BosonOperator,
    FermionBasis,
    FermionOperator,
    PauliOperator,
    QMatrix,
    SpinBasis,
    SpinOperator,
)

# ---------------------------------------------------------------------------
# lhss / max_site validation
# ---------------------------------------------------------------------------


class TestLhssValidation:
    def test_as_linearoperator_rejects_lhss_mismatch(self):
        op = SpinOperator([("z", [[1.0, 0]])], lhss=3)
        basis = SpinBasis.full(2, 4)
        with pytest.raises(ValueError, match="lhss"):
            op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))

    def test_build_spin_rejects_lhss_mismatch(self):
        op = SpinOperator([("z", [[1.0, 0]])], lhss=3)
        basis = SpinBasis.full(2, 4)
        with pytest.raises(ValueError, match="lhss"):
            QMatrix.build_spin(op, basis, np.dtype("complex128"))

    def test_build_boson_rejects_lhss_mismatch(self):
        """The same gap existed in every pre-existing build_* method."""
        op = BosonOperator([("n", [[1.0, 0]])], lhss=3)
        basis = BosonBasis.full(2, 4)
        with pytest.raises(ValueError, match="lhss"):
            QMatrix.build_boson(op, basis, np.dtype("complex128"))

    def test_build_bond_rejects_lhss_mismatch(self):
        # lhss is inferred from the matrix: a (9, 9) two-site block is lhss=3
        mat = np.eye(9, dtype=np.complex128)
        op = BondOperator([(mat, [(0, 1)])])
        basis = SpinBasis.full(2, 4)
        with pytest.raises(ValueError, match="lhss"):
            QMatrix.build_bond(op, basis, np.dtype("complex128"))

    @pytest.mark.parametrize("lhss", [2, 3, 4])
    def test_matching_lhss_still_accepted(self, lhss: int):
        op = SpinOperator([("z", [[1.0, 0]])], lhss=lhss)
        basis = SpinBasis.full(2, lhss)
        lo = op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))
        assert lo.shape == (lhss**2, lhss**2)
        assert QMatrix.build_spin(op, basis, np.dtype("complex128")).dim == lhss**2

    def test_matching_lhss_gives_correct_sz_eigenvalues(self):
        """The values the old, unvalidated path got wrong."""
        lhss = 4  # spin-3/2
        op = SpinOperator([("z", [[1.0, 0]])], lhss=lhss)
        basis = SpinBasis.full(1, lhss)
        lo = op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))
        n = lo.shape[0]
        diag = np.diag(
            np.column_stack(
                [
                    lo.matvec(np.eye(n, dtype=np.complex128)[:, j].copy())
                    for j in range(n)
                ]
            )
        ).real
        assert sorted(np.round(diag, 6)) == [-1.5, -0.5, 0.5, 1.5]


class TestMaxSiteValidation:
    def test_as_linearoperator_rejects_site_past_end(self):
        op = SpinOperator([("z", [[1.0, 7]])], lhss=3)
        basis = SpinBasis.full(2, 3)
        with pytest.raises(ValueError, match="site"):
            op.as_linearoperator(basis, np.ones(1, dtype=np.complex128))

    def test_build_spin_rejects_site_past_end(self):
        op = SpinOperator([("z", [[1.0, 7]])], lhss=3)
        basis = SpinBasis.full(2, 3)
        with pytest.raises(ValueError, match="site"):
            QMatrix.build_spin(op, basis, np.dtype("complex128"))

    def test_build_pauli_rejects_site_past_end(self):
        op = PauliOperator([("z", [[1.0, 9]])])
        basis = SpinBasis.full(3)
        with pytest.raises(ValueError, match="site"):
            QMatrix.build_pauli(op, basis, np.dtype("complex128"))

    def test_build_fermion_rejects_site_past_end(self):
        op = FermionOperator([("n", [[1.0, 9]])])
        basis = FermionBasis.full(3)
        with pytest.raises(ValueError, match="site"):
            QMatrix.build_fermion(op, basis, np.dtype("complex128"))

    def test_highest_valid_site_accepted(self):
        op = SpinOperator([("z", [[1.0, 1]])], lhss=3)
        basis = SpinBasis.full(2, 3)
        assert op.as_linearoperator(basis, np.ones(1, dtype=np.complex128)).shape == (
            9,
            9,
        )


# ---------------------------------------------------------------------------
# Empty coefficient groups
# ---------------------------------------------------------------------------


class TestEmptyCoefficientGroups:
    """Previously a PanicException out of apply; now a clean ValueError."""

    def test_spin_operator_rejects_leading_empty_group(self):
        with pytest.raises(ValueError, match="no bonds"):
            SpinOperator([], [("z", [[1.0, 0]])], lhss=3)

    def test_spin_operator_rejects_trailing_empty_group(self):
        with pytest.raises(ValueError, match="no bonds"):
            SpinOperator([("z", [[1.0, 0]])], [], lhss=3)

    def test_spin_operator_rejects_group_with_only_empty_bond_lists(self):
        with pytest.raises(ValueError, match="no bonds"):
            SpinOperator([("z", [])], [("z", [[1.0, 0]])], lhss=3)

    def test_pauli_operator_rejects_empty_group(self):
        with pytest.raises(ValueError, match="no bonds"):
            PauliOperator([], [("z", [[1.0, 0]])])

    def test_boson_operator_rejects_empty_group(self):
        with pytest.raises(ValueError, match="no bonds"):
            BosonOperator([], [("n", [[1.0, 0]])], lhss=3)

    def test_fermion_operator_rejects_empty_group(self):
        with pytest.raises(ValueError, match="no bonds"):
            FermionOperator([], [("n", [[1.0, 0]])])

    def test_bond_operator_rejects_empty_group(self):
        mat = np.eye(4, dtype=np.complex128)
        with pytest.raises(ValueError, match="no bonds"):
            BondOperator([], [(mat, [(0, 1)])])

    def test_non_empty_groups_still_accepted(self):
        op = SpinOperator([("z", [[1.0, 0]])], [("z", [[1.0, 1]])], lhss=3)
        assert op.num_cindices == 2

        basis = SpinBasis.full(2, 3)
        out = np.zeros(basis.size, dtype=np.complex128)
        op.apply(
            basis,
            np.array([1.0, 2.0], dtype=np.complex128),
            np.ones(basis.size, dtype=np.complex128),
            out,
            True,
        )
        assert np.isfinite(out).all()

    def test_zero_coefficient_bond_is_the_placeholder_escape_hatch(self):
        """The error message points here; make sure it actually works."""
        op = SpinOperator([("z", [[0.0, 0]])], [("z", [[1.0, 1]])], lhss=3)
        assert op.num_cindices == 2


# ---------------------------------------------------------------------------
# apply() / apply_and_project_to() — validated in the Rust kernel, so every
# operator wrapper is covered, not just the ones with a Python-side check
# ---------------------------------------------------------------------------


class TestApplyValidation:
    @staticmethod
    def _run(op, basis, n_coeffs=1):
        out = np.zeros(basis.size, dtype=np.complex128)
        op.apply(
            basis,
            np.ones(n_coeffs, dtype=np.complex128),
            np.ones(basis.size, dtype=np.complex128),
            out,
            True,
        )
        return out

    def test_apply_rejects_lhss_mismatch(self):
        op = SpinOperator([("z", [[1.0, 0]])], lhss=3)
        with pytest.raises(ValueError, match="lhss"):
            self._run(op, SpinBasis.full(1, 4))

    def test_apply_rejects_lhss_mismatch_for_pauli(self):
        """Pauli assumes two levels; an lhss=3 basis must not be accepted."""
        op = PauliOperator([("z", [[1.0, 0]])])
        with pytest.raises(ValueError, match="lhss"):
            self._run(op, SpinBasis.full(1, 3))

    def test_apply_rejects_site_past_end(self):
        op = SpinOperator([("z", [[1.0, 9]])], lhss=3)
        with pytest.raises(ValueError, match="site"):
            self._run(op, SpinBasis.full(2, 3))

    def test_apply_still_works_when_compatible(self):
        op = SpinOperator([("z", [[1.0, 0]])], lhss=3)
        out = self._run(op, SpinBasis.full(2, 3))
        assert np.isfinite(out).all()

    def test_apply_and_project_to_rejects_lhss_mismatch(self):
        op = SpinOperator([("z", [[1.0, 0]])], lhss=3)
        basis = SpinBasis.full(1, 4)
        out = np.zeros(basis.size, dtype=np.complex128)
        with pytest.raises(ValueError, match="lhss"):
            op.apply_and_project_to(
                basis,
                basis,
                np.ones(1, dtype=np.complex128),
                np.ones(basis.size, dtype=np.complex128),
                out,
                True,
            )


# ---------------------------------------------------------------------------
# lhss upper bound
# ---------------------------------------------------------------------------


class TestLhssUpperBound:
    """Above 255 the dit encoding asserts; that used to surface as a panic."""

    @pytest.mark.parametrize("lhss", [256, 300, 1000])
    def test_spin_operator_rejects_lhss_above_limit(self, lhss: int):
        with pytest.raises(ValueError, match="2..=255"):
            SpinOperator([("z", [[1.0, 0]])], lhss=lhss)

    @pytest.mark.parametrize("lhss", [256, 1000])
    def test_boson_operator_rejects_lhss_above_limit(self, lhss: int):
        with pytest.raises(ValueError, match="2..=255"):
            BosonOperator([("n", [[1.0, 0]])], lhss=lhss)

    @pytest.mark.parametrize("lhss", [0, 1])
    def test_lhss_below_two_still_rejected(self, lhss: int):
        with pytest.raises(ValueError, match="2..=255"):
            SpinOperator([("z", [[1.0, 0]])], lhss=lhss)

    def test_lhss_at_the_limit_is_accepted(self):
        assert SpinOperator([("z", [[1.0, 0]])], lhss=255).lhss == 255

    def test_matches_the_basis_constructors(self):
        """Operators and bases now reject the same range."""
        with pytest.raises(ValueError, match="2..=255"):
            SpinBasis.full(1, 256)
        with pytest.raises(ValueError, match="2..=255"):
            SpinOperator([("z", [[1.0, 0]])], lhss=256)
