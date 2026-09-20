"""Operator/basis compatibility and cindex-hole validation.

Three classes of silent failure, all reachable from the existing public API:

* An operator whose ``lhss`` or ``max_site`` disagreed with the basis was
  accepted and then produced wrong matrix elements — a spin-1 ``Sz`` on a
  spin-3/2 basis returned ``{1, 0, -1, -2}`` instead of ``{±1.5, ±0.5}``, and
  an out-of-range site folded back in as a phantom diagonal term.
* An empty ``*terms`` group left a hole in the cindex sequence while
  ``num_cindices`` counted only the cindices actually present, so
  ``coeffs[cindex]`` ran off the end of the coefficient slice and panicked.
* ``lhss`` above 255 reached ``DynamicDitManip``, which asserts rather than
  erroring, so it surfaced as a ``PanicException``.
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
)

# ---------------------------------------------------------------------------
# lhss / max_site: the assembled (QMatrix) path
# ---------------------------------------------------------------------------


class TestBuildValidation:
    def test_build_boson_rejects_lhss_mismatch(self):
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

    def test_build_pauli_rejects_lhss_mismatch(self):
        op = PauliOperator([("z", [[1.0, 0]])])  # lhss = 2
        basis = SpinBasis.full(2, 3)
        with pytest.raises(ValueError, match="lhss"):
            QMatrix.build_pauli(op, basis, np.dtype("complex128"))

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

    def test_compatible_pairs_still_build(self):
        bonds = [[1.0, i, i + 1] for i in range(3)]
        op = PauliOperator([("xx", bonds)], [("zz", bonds)])
        basis = SpinBasis.full(4)
        assert QMatrix.build_pauli(op, basis, np.dtype("complex128")).dim == 16


# ---------------------------------------------------------------------------
# lhss / max_site: the matrix-free (apply) path
# ---------------------------------------------------------------------------


class TestApplyValidation:
    """Validated inside the Rust kernel, so every wrapper is covered."""

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
        """Pauli assumes two levels; an lhss=3 basis must not be accepted."""
        op = PauliOperator([("z", [[1.0, 0]])])
        with pytest.raises(ValueError, match="lhss"):
            self._run(op, SpinBasis.full(1, 3))

    def test_apply_rejects_lhss_mismatch_for_boson(self):
        op = BosonOperator([("n", [[1.0, 0]])], lhss=3)
        with pytest.raises(ValueError, match="lhss"):
            self._run(op, BosonBasis.full(1, 4))

    def test_apply_rejects_site_past_end(self):
        op = BosonOperator([("n", [[1.0, 9]])], lhss=3)
        with pytest.raises(ValueError, match="site"):
            self._run(op, BosonBasis.full(2, 3))

    def test_apply_still_works_when_compatible(self):
        op = BosonOperator([("n", [[1.0, 0]])], lhss=3)
        assert np.isfinite(self._run(op, BosonBasis.full(2, 3))).all()

    def test_apply_and_project_to_rejects_lhss_mismatch(self):
        op = PauliOperator([("z", [[1.0, 0]])])
        basis = SpinBasis.full(1, 3)
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
# Empty coefficient groups
# ---------------------------------------------------------------------------


class TestEmptyCoefficientGroups:
    """Previously a PanicException out of apply; now a clean ValueError."""

    def test_pauli_operator_rejects_leading_empty_group(self):
        with pytest.raises(ValueError, match="no bonds"):
            PauliOperator([], [("z", [[1.0, 0]])])

    def test_pauli_operator_rejects_trailing_empty_group(self):
        with pytest.raises(ValueError, match="no bonds"):
            PauliOperator([("z", [[1.0, 0]])], [])

    def test_rejects_group_with_only_empty_bond_lists(self):
        with pytest.raises(ValueError, match="no bonds"):
            PauliOperator([("z", [])], [("z", [[1.0, 0]])])

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
        op = PauliOperator([("z", [[1.0, 0]])], [("z", [[1.0, 1]])])
        assert op.num_cindices == 2

        basis = SpinBasis.full(2)
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
        op = PauliOperator([("z", [[0.0, 0]])], [("z", [[1.0, 1]])])
        assert op.num_cindices == 2


# ---------------------------------------------------------------------------
# lhss upper bound
# ---------------------------------------------------------------------------


class TestLhssUpperBound:
    """Above 255 the dit encoding asserts; that used to surface as a panic."""

    @pytest.mark.parametrize("lhss", [256, 300, 1000])
    def test_boson_operator_rejects_lhss_above_limit(self, lhss: int):
        with pytest.raises(ValueError, match="2..=255"):
            BosonOperator([("n", [[1.0, 0]])], lhss=lhss)

    @pytest.mark.parametrize("lhss", [0, 1])
    def test_lhss_below_two_still_rejected(self, lhss: int):
        with pytest.raises(ValueError, match="2..=255"):
            BosonOperator([("n", [[1.0, 0]])], lhss=lhss)

    def test_lhss_at_the_limit_is_accepted(self):
        assert BosonOperator([("n", [[1.0, 0]])], lhss=255).lhss == 255

    def test_matches_the_basis_constructors(self):
        """Operators and bases now reject the same range."""
        with pytest.raises(ValueError, match="2..=255"):
            BosonBasis.full(1, 256)
        with pytest.raises(ValueError, match="2..=255"):
            BosonOperator([("n", [[1.0, 0]])], lhss=256)
