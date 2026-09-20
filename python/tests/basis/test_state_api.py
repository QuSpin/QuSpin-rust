"""Tests for the shared basis state API.

Covers ``Ns``, ``states``, ``state_to_int``, ``int_to_state``, ``index``,
``index_str``, ``project_to`` and ``project_from`` across all four basis
types.
"""

import numpy as np
import pytest

from quspin_rs._rs import (
    BosonBasis,
    FermionBasis,
    GenericBasis,
    Lattice,
    PauliOperator,
    SpinBasis,
)
from quspin_rs.symmetry import SymmetryGroup


def all_bases():
    """One built basis of each type, paired with its (n_sites, lhss)."""
    return [
        pytest.param(SpinBasis.full(3), 3, 2, id="spin"),
        pytest.param(FermionBasis.full(3), 3, 2, id="fermion"),
        pytest.param(BosonBasis.full(2, 3), 2, 3, id="boson"),
        pytest.param(GenericBasis.full(2, 3), 2, 3, id="generic"),
    ]


BASES = all_bases()


class TestStatesProperty:
    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_states_is_ndarray_of_length_ns(self, basis, n_sites, lhss):
        assert isinstance(basis.states, np.ndarray)
        assert basis.states.shape == (basis.Ns,)

    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_ns_matches_size(self, basis, n_sites, lhss):
        assert basis.Ns == basis.size

    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_states_round_trip_through_index(self, basis, n_sites, lhss):
        for i, s in enumerate(basis.states):
            assert basis.index(s) == i

    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_states_agree_with_state_at(self, basis, n_sites, lhss):
        for i, s in enumerate(basis.states):
            assert basis.state_to_int(basis.state_at(i)) == s


class TestStateIntRoundTrip:
    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_int_to_state_round_trips(self, basis, n_sites, lhss):
        for s in basis.states:
            assert basis.state_to_int(basis.int_to_state(s)) == s

    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_bracket_notation_toggle(self, basis, n_sites, lhss):
        s = basis.states[0]
        with_ket = basis.int_to_state(s, bracket_notation=True)
        without = basis.int_to_state(s, bracket_notation=False)
        assert with_ket == f"|{without}>"
        # both forms parse back to the same integer
        assert basis.state_to_int(with_ket) == basis.state_to_int(without) == s

    def test_accepts_every_string_form(self):
        basis = SpinBasis.full(2)
        forms = ["01", "|01>", "0 1", "|0 1>", "0,1", "|0,1>"]
        assert {basis.state_to_int(f) for f in forms} == {basis.state_to_int("01")}

    def test_multi_digit_occupations_need_tokens(self):
        basis = BosonBasis.full(2, 12)
        s = basis.state_to_int("|0 11>")
        assert basis.int_to_state(s) == "|0 11>"
        assert basis.index(s) is not None

    def test_half_a_ket_is_rejected(self):
        basis = SpinBasis.full(2)
        for bad in ["|01", "01>"]:
            with pytest.raises(ValueError):
                basis.state_to_int(bad)

    def test_wrong_length_is_rejected(self):
        basis = SpinBasis.full(2)
        with pytest.raises(ValueError):
            basis.state_to_int("011")

    def test_out_of_range_site_value_is_rejected(self):
        basis = GenericBasis.full(2, 3)
        with pytest.raises(ValueError):
            basis.state_to_int("03")


class TestIndex:
    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_index_str_matches_index(self, basis, n_sites, lhss):
        for i in range(basis.size):
            state_str = basis.state_at(i)
            assert basis.index_str(state_str) == i
            assert basis.index(basis.state_to_int(state_str)) == i

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8, np.uint64])
    def test_accepts_numpy_integers(self, dtype):
        basis = SpinBasis.full(3)
        for i, s in enumerate(basis.states):
            assert basis.index(dtype(s)) == i

    def test_rejects_integers_wider_than_the_basis(self):
        basis = SpinBasis.full(2)
        for bad in (4, 7, 2**40):
            with pytest.raises(ValueError, match="bits"):
                basis.index(bad)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            SpinBasis.full(2).index(-1)

    def test_rejects_non_integers(self):
        with pytest.raises(TypeError):
            SpinBasis.full(2).index(2.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            SpinBasis.full(2).index("01")  # type: ignore[arg-type]

    def test_rejects_site_value_outside_lhss(self):
        # 0b11 = 3 at site 0, but lhss is 3 so only 0..2 are valid.
        with pytest.raises(ValueError, match="invalid local value"):
            GenericBasis.full(2, 3).index(3)

    def test_index_returns_none_for_state_outside_a_subspace(self):
        op = PauliOperator([("XX", [[1.0, 0, 1]])])
        sub = SpinBasis.subspace(3, op, ["011"])
        outside = [s for s in SpinBasis.full(3).states if sub.index(s) is None]
        assert outside, "subspace should not contain every state"


# ---------------------------------------------------------------------------
# project_to / project_from
# ---------------------------------------------------------------------------


def parity_bases(eta):
    """Two-site basis in the parity sector `eta`, plus the full basis."""
    group = SymmetryGroup(n_sites=2, lhss=2)
    group.add_cyclic(Lattice([1, 0]), eta=eta)
    op = PauliOperator([("XX", [[1.0, 0, 1]]), ("Z", [[0.17, 0], [0.17, 1]])])
    seeds = ["01", "00"]
    return SpinBasis.symmetric(group, op, seeds=seeds), SpinBasis.full(2)


class TestProjection:
    def test_project_from_symmetric_state_carries_the_norm(self):
        sym, full = parity_bases(+1)
        # the orbit {|01>, |10>} is the one non-invariant representative
        idx = sym.index(sym.state_to_int("01"))
        psi = np.zeros(sym.Ns)
        psi[idx] = 1.0

        out = sym.project_from(psi)
        expected = np.zeros(full.Ns)
        expected[full.index(full.state_to_int("01"))] = 1 / np.sqrt(2)
        expected[full.index(full.state_to_int("10"))] = 1 / np.sqrt(2)
        np.testing.assert_allclose(out, expected, atol=1e-14)

    def test_project_to_antisymmetric_sector(self):
        sym, full = parity_bases(-1)
        psi = np.zeros(full.Ns)
        psi[full.index(full.state_to_int("01"))] = 1.0

        out = sym.project_to(psi)
        assert out.shape == (sym.Ns,)
        np.testing.assert_allclose(np.abs(out), [1 / np.sqrt(2)], atol=1e-14)

        # Lifting the normalised projection back gives the singlet. Compare via
        # the overlap magnitude rather than elementwise: that tolerates an
        # overall sign/phase convention but still requires the *relative* minus
        # sign that distinguishes the singlet from the triplet.
        lifted = sym.project_from(out / np.linalg.norm(out))
        singlet = np.zeros(full.Ns)
        singlet[full.index(full.state_to_int("01"))] = 1 / np.sqrt(2)
        singlet[full.index(full.state_to_int("10"))] = -1 / np.sqrt(2)
        triplet = np.abs(singlet)

        np.testing.assert_allclose(np.linalg.norm(lifted), 1.0, atol=1e-14)
        np.testing.assert_allclose(abs(np.vdot(singlet, lifted)), 1.0, atol=1e-14)
        # the symmetric combination must NOT pass
        assert abs(np.vdot(triplet, lifted)) < 1e-14

    def test_round_trip_is_identity_on_the_sector(self):
        sym, _ = parity_bases(+1)
        rng = np.random.default_rng(0)
        psi = rng.normal(size=sym.Ns)
        np.testing.assert_allclose(
            sym.project_to(sym.project_from(psi)), psi, atol=1e-13
        )

    def test_projects_each_column_of_a_matrix(self):
        sym, full = parity_bases(+1)
        mat = np.eye(sym.Ns)
        out = sym.project_from(mat)
        assert out.shape == (full.Ns, sym.Ns)
        for col in range(sym.Ns):
            np.testing.assert_allclose(out[:, col], sym.project_from(mat[:, col]))

    def test_accepts_lists_and_complex_input(self):
        sym, full = parity_bases(+1)
        psi = [0.0] * sym.Ns
        psi[0] = 1.0
        np.testing.assert_allclose(
            sym.project_from(psi), sym.project_from(np.array(psi))
        )

        cplx = np.zeros(sym.Ns, dtype=np.complex128)
        cplx[0] = 1j
        out = sym.project_from(cplx)
        assert np.iscomplexobj(out)

    def test_full_basis_projection_is_identity(self):
        full = SpinBasis.full(2)
        psi = np.arange(full.Ns, dtype=float)
        np.testing.assert_allclose(full.project_to(psi), psi)
        np.testing.assert_allclose(full.project_from(psi), psi)

    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_rejects_wrong_length_input(self, basis, n_sites, lhss):
        with pytest.raises(ValueError):
            basis.project_from(np.zeros(basis.Ns + 3))

    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_rejects_wrong_row_count_even_with_zero_columns(self, basis, n_sites, lhss):
        # No column means the per-column projection never runs, so the row
        # count has to be checked up front.
        with pytest.raises(ValueError):
            basis.project_from(np.zeros((basis.Ns + 1, 0)))

    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_accepts_a_zero_column_matrix_of_the_right_height(
        self, basis, n_sites, lhss
    ):
        out = basis.project_from(np.zeros((basis.Ns, 0)))
        assert out.shape == (basis.Ns, 0)

    @pytest.mark.parametrize("basis,n_sites,lhss", BASES)
    def test_sparse_true_raises_not_implemented(self, basis, n_sites, lhss):
        with pytest.raises(NotImplementedError):
            basis.project_to(np.zeros(basis.Ns), sparse=True)
        with pytest.raises(NotImplementedError):
            basis.project_from(np.zeros(basis.Ns), sparse=True)
