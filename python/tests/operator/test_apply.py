"""Tests for Operator.apply / apply_and_project_to (matrix-free path)."""

import numpy as np

from quspin_rs._rs import Hamiltonian, PauliOperator, QMatrix, SpinBasis, Static


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
