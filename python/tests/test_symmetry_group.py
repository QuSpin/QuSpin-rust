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

    def test_local_local_stays_local(self):
        from quspin_rs._rs import _compose

        a = Local([1, 0])
        b = Local([1, 0])
        c = _compose(a, b)
        assert c == Local([0, 1])  # involution squared = identity perm_vals

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

        a = Local([2, 0, 1], locs=[0, 2])
        b = Local([1, 2, 0], locs=[0, 2])
        c = _compose(a, b)
        # composition: [2,0,1] ∘ [1,2,0] -> a[b[0]]=a[1]=0, a[b[1]]=a[2]=1, a[b[2]]=a[0]=2
        # → identity perm_vals, but with explicit locs preserved.
        assert c == Local([0, 1, 2], locs=[0, 2])

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
        # But the resulting element still has perm + perm_vals → Composite.
        assert c == Composite([1, 2, 0], [0, 1])
