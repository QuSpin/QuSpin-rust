import cmath
from math import pi

import pytest

from quspin_rs import Composite, Lattice, Local, SymElement


class TestSymmetryGroupBasics:
    def test_construct(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        assert g.n_sites == 4
        assert g.lhss == 2
        assert len(g) == 0

    def test_add_and_iter(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        T = Lattice([1, 2, 3, 0])
        g.add(T, 1.0 + 0j)
        assert len(g) == 1
        elems = list(g)
        assert elems[0][0] == T
        assert elems[0][1] == 1.0 + 0j

    def test_repr(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        assert "SymmetryGroup" in repr(g)
        assert "n_sites=4" in repr(g)

    def test_n_sites_lhss_validation(self):
        from quspin_rs import SymmetryGroup

        with pytest.raises(ValueError, match="n_sites"):
            SymmetryGroup(n_sites=0, lhss=2)
        with pytest.raises(ValueError, match="lhss"):
            SymmetryGroup(n_sites=4, lhss=1)


class TestSymElementConstructors:
    def test_lattice_repr_roundtrip(self):
        a = Lattice([1, 2, 0])
        b = Lattice([1, 2, 0])
        assert a == b
        assert hash(a) == hash(b)
        assert "Lattice" in repr(a)

    def test_local_default_locs_is_none(self):
        a = Local([1, 0])
        b = Local([1, 0], locs=None)
        assert a == b

    def test_local_explicit_locs(self):
        a = Local([1, 0], locs=[0, 2])
        assert a != Local([1, 0])  # default locs vs explicit aren't equal
        assert "Local" in repr(a)

    def test_composite_repr(self):
        c = Composite([2, 1, 0], [1, 0])
        assert "Composite" in repr(c)

    def test_lattice_rejects_negative_int_with_hint(self):
        with pytest.raises(ValueError, match="Composite"):
            Lattice([-1, 0, 1])

    def test_isinstance_symelement(self):
        assert isinstance(Lattice([0]), SymElement)
        assert isinstance(Local([1, 0]), SymElement)
        assert isinstance(Composite([0], [1, 0]), SymElement)

    def test_local_repr_python_shaped(self):
        a = Local([1, 0], locs=[0, 2])
        s = repr(a)
        assert "Some(" not in s
        assert "locs=[0, 2]" in s
        b = Local([1, 0])
        sb = repr(b)
        assert "Some(" not in sb
        assert "locs=None" in sb

    def test_lattice_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            Lattice([5, 0, 1, 2])

    def test_local_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            Local([3, 0])  # value 3 exceeds perm_vals.len()=2

    def test_composite_rejects_out_of_range_perm(self):
        with pytest.raises(ValueError, match="out of range"):
            Composite([5, 0, 1, 2], [1, 0])

    def test_composite_rejects_out_of_range_perm_vals(self):
        with pytest.raises(ValueError, match="out of range"):
            Composite([1, 0, 2], [3, 0])


class TestOrder:
    def test_lattice_4cycle(self):
        from quspin_rs._rs import _order

        assert _order(Lattice([1, 2, 3, 0]), n_sites=4, lhss=2) == 4

    def test_lattice_two_2cycles(self):
        from quspin_rs._rs import _order

        assert _order(Lattice([1, 0, 3, 2]), n_sites=4, lhss=2) == 2

    def test_lattice_3cycle_plus_2cycle(self):
        from quspin_rs._rs import _order

        # sites 0->1->2->0 (3-cycle) and 3<->4 (2-cycle): order = lcm(3,2) = 6
        assert _order(Lattice([1, 2, 0, 4, 3]), n_sites=5, lhss=2) == 6

    def test_local_z2_swap(self):
        from quspin_rs._rs import _order

        assert _order(Local([1, 0]), n_sites=4, lhss=2) == 2

    def test_local_z3_cycle(self):
        from quspin_rs._rs import _order

        assert _order(Local([1, 2, 0]), n_sites=4, lhss=3) == 3

    def test_composite_lcm(self):
        from quspin_rs._rs import _order

        # perm has order 4 (4-cycle), perm_vals has order 2 -> composite order 4
        assert _order(Composite([1, 2, 3, 0], [1, 0]), n_sites=4, lhss=2) == 4

    def test_composite_lcm_strict(self):
        from quspin_rs._rs import _order

        # lattice 3-cycle (order 3) on lhss=2 sites; perm_vals = [1, 0] (order 2).
        # lcm(3, 2) = 6, which exceeds both 3 and 2. Distinguishes "LCM" from "max".
        assert _order(Composite([1, 2, 0], [1, 0]), n_sites=3, lhss=2) == 6

    def test_identity_order_is_one(self):
        from quspin_rs._rs import _order

        assert _order(Lattice([0, 1, 2]), n_sites=3, lhss=2) == 1


class TestCompose:
    def test_lattice_lattice_stays_lattice(self):
        from quspin_rs._rs import _compose

        a = Lattice([1, 2, 0])  # 3-cycle
        b = Lattice([1, 2, 0])
        c = _compose(a, b)
        assert c == Lattice([2, 0, 1])  # (a∘b)[s] = a[b[s]]

    def test_local_local_identity_errors(self):
        from quspin_rs._rs import _compose

        # Involution squared yields identity perm_vals → must error per
        # the "no identity in the group" contract.
        with pytest.raises(ValueError, match="produced identity"):
            _compose(Local([1, 0]), Local([1, 0]))

    def test_local_local_non_identity(self):
        from quspin_rs._rs import _compose

        a = Local([1, 2, 0])  # 3-cycle
        b = Local([1, 2, 0])
        c = _compose(a, b)
        assert c == Local([2, 0, 1])  # 3-cycle squared

    def test_lattice_local_promotes_to_composite(self):
        from quspin_rs._rs import _compose

        a = Lattice([1, 2, 0])
        b = Local([1, 0])
        c = _compose(a, b)
        assert isinstance(c, SymElement)
        # repr indicates Composite kind
        assert "Composite" in repr(c)

    def test_local_locs_mismatch_errors(self):
        from quspin_rs._rs import _compose

        a = Local([2, 0, 1], locs=[0])
        b = Local([1, 2, 0], locs=[1])
        with pytest.raises(ValueError, match="locs must match"):
            _compose(a, b)

    def test_local_same_locs_carries_through(self):
        from quspin_rs._rs import _compose

        # Self-compose a 3-cycle: result is the inverse 3-cycle (non-identity)
        # with the explicit locs preserved.
        a = Local([1, 2, 0], locs=[0, 2])
        b = Local([1, 2, 0], locs=[0, 2])
        c = _compose(a, b)
        # composition: [1,2,0] ∘ [1,2,0] -> a[b[0]]=a[1]=2, a[b[1]]=a[2]=0, a[b[2]]=a[0]=1
        assert c == Local([2, 0, 1], locs=[0, 2])

    def test_local_same_locs_identity_errors(self):
        from quspin_rs._rs import _compose

        # Identity result on the perm_vals axis should error even with
        # explicit locs (locs annotate where the local op acts; an
        # identity perm_vals is still the identity element).
        a = Local([2, 0, 1], locs=[0, 2])
        b = Local([1, 2, 0], locs=[0, 2])
        with pytest.raises(ValueError, match="produced identity"):
            _compose(a, b)

    def test_mixed_none_explicit_locs_errors(self):
        from quspin_rs._rs import _compose

        a = Local([1, 0])  # locs=None
        b = Local([1, 0], locs=[0])  # explicit
        with pytest.raises(ValueError, match="None locs"):
            _compose(a, b)

    def test_perm_length_mismatch_errors(self):
        from quspin_rs._rs import _compose

        a = Lattice([1, 0])
        b = Lattice([1, 2, 0])
        with pytest.raises(ValueError, match="perm lengths differ"):
            _compose(a, b)

    def test_compose_identity_result_errors(self):
        from quspin_rs._rs import _compose

        # Lattice involution composed with itself yields identity perm.
        a = Lattice([1, 0])
        b = Lattice([1, 0])
        with pytest.raises(ValueError, match="identity"):
            _compose(a, b)

    def test_composite_composite(self):
        from quspin_rs._rs import _compose

        a = Composite([2, 1, 0], [1, 0])
        b = Composite([1, 0, 2], [1, 0])
        c = _compose(a, b)
        # perm: a∘b = [a[b[0]], a[b[1]], a[b[2]]] = [a[1], a[0], a[2]] = [1, 2, 0]
        # perm_vals: [1,0] ∘ [1,0] = [0, 1] (identity perm_vals)
        # But the resulting element still has a non-trivial perm → Composite.
        assert c == Composite([1, 2, 0], [0, 1])

    def test_composite_full_identity_errors(self):
        from quspin_rs._rs import _compose

        # Both components are involutions; their product is full identity
        # on both axes. Must error.
        a = Composite([1, 0], [1, 0])
        b = Composite([1, 0], [1, 0])
        with pytest.raises(ValueError, match="produced identity"):
            _compose(a, b)


class TestAddCyclic:
    def test_translation_k_equiv_eta_for_z2(self):
        from quspin_rs import SymmetryGroup

        g_k = SymmetryGroup(n_sites=2, lhss=2)
        g_k.add_cyclic(Lattice([1, 0]), k=1)
        g_eta = SymmetryGroup(n_sites=2, lhss=2)
        g_eta.add_cyclic(Lattice([1, 0]), eta=-1)
        assert list(g_k) == list(g_eta)

    def test_translation_k1_z4(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        g.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        assert len(g) == 3
        omega = cmath.exp(-2j * pi / 4)
        for (_, chi), expected in zip(g, [omega, omega**2, omega**3]):
            assert abs(chi - expected) < 1e-12

    def test_eta_only_for_order_2(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        with pytest.raises(ValueError, match="order"):
            g.add_cyclic(Lattice([1, 2, 3, 0]), eta=-1)

    def test_exactly_one_of_k_eta_char(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        with pytest.raises(ValueError):
            g.add_cyclic(Lattice([1, 2, 3, 0]))
        with pytest.raises(ValueError):
            g.add_cyclic(Lattice([1, 2, 3, 0]), k=1, eta=1)

    def test_identity_generator_rejected(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=3, lhss=2)
        with pytest.raises(ValueError):
            g.add_cyclic(Lattice([0, 1, 2]), k=0)

    def test_k_out_of_range(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        with pytest.raises(ValueError):
            g.add_cyclic(Lattice([1, 2, 3, 0]), k=5)

    def test_char_argument(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        # Order-4 generator with explicit char z; expect z, z², z³.
        z = cmath.exp(-1j * pi / 2)  # ω, equivalent to k=1 for order=4
        g.add_cyclic(Lattice([1, 2, 3, 0]), char=z)
        chars = [chi for _, chi in g]
        for got, expected in zip(chars, [z, z**2, z**3]):
            assert abs(got - expected) < 1e-12


class TestClose:
    def test_close_dihedral_d4_trivial_rep(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        T = Lattice([1, 2, 3, 0])
        P = Lattice([3, 2, 1, 0])
        g.close(generators=[T, P], char=lambda elem: 1.0)
        # D_4 has 2*4 = 8 elements; 7 non-identity.
        assert len(g) == 7

    def test_close_just_T_z4(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        T = Lattice([1, 2, 3, 0])
        g.close(generators=[T], char=lambda elem: 1.0)
        assert len(g) == 3  # T, T², T³

    def test_close_empty_generators_no_op(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        g.close(generators=[], char=lambda elem: 1.0)
        assert len(g) == 0

    def test_close_char_called_per_element(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        seen: list[object] = []

        def tracker(elem):
            seen.append(elem)
            return 1.0

        T = Lattice([1, 2, 3, 0])
        g.close(generators=[T], char=tracker)
        assert len(seen) == 3  # called for T, T², T³
        # Each elem in `seen` is a SymElement
        from quspin_rs import SymElement

        for elem in seen:
            assert isinstance(elem, SymElement)

    def test_close_propagates_malformed_generator(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        # Two lattice generators with incompatible perm lengths — should
        # surface as a ValueError mentioning length, NOT be silently
        # swallowed by close's identity-skip.
        a = Lattice([1, 2, 3, 0])  # length 4
        b = Lattice([1, 0])  # length 2
        with pytest.raises(ValueError, match="lengths differ"):
            g.close(generators=[a, b], char=lambda elem: 1.0)


class TestProduct:
    def test_z4_x_z2_size(self):
        from quspin_rs import SymmetryGroup

        T = SymmetryGroup(n_sites=4, lhss=2)
        T.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        Z = SymmetryGroup(n_sites=4, lhss=2)
        Z.add_cyclic(Local([1, 0]), eta=-1)
        G = T.product(Z)
        # 4·2 - 1 = 7 non-identity elements
        assert len(G) == 7

    def test_product_does_not_mutate(self):
        from quspin_rs import SymmetryGroup

        T = SymmetryGroup(n_sites=4, lhss=2)
        T.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        n_before = len(T)
        Z = SymmetryGroup(n_sites=4, lhss=2)
        Z.add_cyclic(Local([1, 0]), eta=-1)
        _ = T.product(Z)
        assert len(T) == n_before  # T unchanged

    def test_product_lhss_mismatch_raises(self):
        from quspin_rs import SymmetryGroup

        a = SymmetryGroup(n_sites=4, lhss=2)
        b = SymmetryGroup(n_sites=4, lhss=3)
        with pytest.raises(ValueError):
            a.product(b)

    def test_product_n_sites_mismatch_raises(self):
        from quspin_rs import SymmetryGroup

        a = SymmetryGroup(n_sites=4, lhss=2)
        b = SymmetryGroup(n_sites=3, lhss=2)
        with pytest.raises(ValueError):
            a.product(b)

    def test_product_returns_new_object(self):
        from quspin_rs import SymmetryGroup

        a = SymmetryGroup(n_sites=4, lhss=2)
        a.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        b = SymmetryGroup(n_sites=4, lhss=2)
        b.add_cyclic(Local([1, 0]), eta=-1)
        c = a.product(b)
        assert c is not a
        assert c is not b


class TestValidate:
    def test_validate_clean_group(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        g.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        g.validate()  # should not raise

    def test_validate_rejects_missing_closure(self):
        from quspin_rs import SymmetryGroup

        g = SymmetryGroup(n_sites=4, lhss=2)
        # Add T but not T^2 / T^3 - closure broken.
        g.add(Lattice([1, 2, 3, 0]), 1.0 + 0j)
        with pytest.raises(ValueError, match="closed|closure"):
            g.validate()

    def test_validate_rejects_inconsistent_chars(self):
        from quspin_rs import SymmetryGroup
        from quspin_rs._rs import _compose

        # Cyclic group with wrong chars (not a 1-D rep).
        g = SymmetryGroup(n_sites=4, lhss=2)
        T = Lattice([1, 2, 3, 0])
        g.add(
            T, -1.0 + 0j
        )  # chi(T) = -1, but T has order 4 so chi should be a 4th root of unity
        T2 = _compose(T, T)
        g.add(T2, 2.0 + 0j)  # chi(T^2) should be (-1)^2 = 1, not 2
        T3 = _compose(T2, T)
        g.add(T3, -1.0 + 0j)  # chi(T^3)
        with pytest.raises(ValueError, match="character"):
            g.validate()

    def test_validate_empty_group_ok(self):
        from quspin_rs import SymmetryGroup

        # Trivial group (only implicit identity) should validate.
        g = SymmetryGroup(n_sites=4, lhss=2)
        g.validate()


class TestBasisSymmetricEndToEnd:
    def test_spin_symmetric_translation(self):
        from quspin_rs import PauliOperator, SpinBasis, SymmetryGroup

        n_sites = 4
        # XX chain with periodic boundary conditions.
        bonds = [[1.0, i, (i + 1) % n_sites] for i in range(n_sites)]
        H = PauliOperator([("XX", bonds)])

        group = SymmetryGroup(n_sites=n_sites, lhss=2)
        group.add_cyclic(Lattice(list(range(1, n_sites)) + [0]), k=0)

        basis = SpinBasis.symmetric(group, H, ["0000"])
        assert basis.is_built
        assert basis.size > 0

    def test_fermion_symmetric_rejects_lhss_neq_2(self):
        from quspin_rs import FermionBasis, FermionOperator, SymmetryGroup

        # Fermions are always lhss=2; a SymmetryGroup with lhss=3 must raise.
        group = SymmetryGroup(n_sites=4, lhss=3)
        H = FermionOperator([("+", [[1.0, 0]])])
        with pytest.raises(TypeError, match="lhss"):
            FermionBasis.symmetric(group, H, ["0000"])

    def test_generic_symmetric_with_local(self):
        import numpy as np

        from quspin_rs import SymmetryGroup
        from quspin_rs._rs import GenericBasis, MonomialOperator

        n_sites = 3
        lhss = 3
        # Identity-like Hamiltonian: perm = identity, amp = 0 everywhere
        # so BFS reaches just the seed.
        dim = lhss * lhss
        perm = np.arange(dim, dtype=np.intp)
        amp = np.zeros(dim, dtype=complex)
        H = MonomialOperator(lhss, (perm, amp, [(0, 1)]))

        group = SymmetryGroup(n_sites=n_sites, lhss=lhss)
        # cyclic Z_3 local symmetry (per-site value-permutation [1, 2, 0]).
        group.add_cyclic(Local([1, 2, 0]), k=0)

        basis = GenericBasis.symmetric(group, H, ["000"])
        assert basis.is_built
