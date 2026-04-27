"""Demo of the SymmetryGroup API for *Basis.symmetric(...).

Walks through the four canonical cases from
``docs/superpowers/specs/2026-04-26-python-symmetry-group-api-design.md``:

1. translation `Z_L` with momentum `k`
2. translation × spin-flip `Z_L × Z_2` (abelian direct product)
3. PZ composite (single non-trivial generator)
4. dihedral `D_L = ⟨T, P⟩` (non-abelian closure)

Plus a compatibility-check failure demo for `FermionBasis`.

Run with: ``uv run python scripts/symmetry_group_demo.py``.
"""

from __future__ import annotations

from quspin_rs import (
    Composite,
    FermionBasis,
    FermionOperator,
    Lattice,
    Local,
    PauliOperator,
    SpinBasis,
    SymmetryGroup,
)


def _xx_chain(n_sites: int) -> PauliOperator:
    """XX nearest-neighbour chain with periodic boundary conditions."""
    bonds = [[1.0, i, (i + 1) % n_sites] for i in range(n_sites)]
    return PauliOperator([("XX", bonds)])


def _print_group(label: str, group: SymmetryGroup) -> None:
    print(f"  {label}")
    print(f"    n_sites = {group.n_sites}, lhss = {group.lhss}, |G| = {1 + len(group)}")
    group.validate()  # closure + 1-D rep check; raises on mismatch
    print("    validate(): OK")


def case_1_translation_only() -> None:
    """Z_L cyclic translation, momentum k=1."""
    print("\n--- Case 1: translation only Z_L, k=1 -----------------------------")
    n_sites = 4
    group = SymmetryGroup(n_sites=n_sites, lhss=2)
    group.add_cyclic(Lattice([(i + 1) % n_sites for i in range(n_sites)]), k=1)
    _print_group(f"Z_{n_sites} with momentum k=1", group)

    basis = SpinBasis.symmetric(group, _xx_chain(n_sites), seeds=["0001"])
    print(f"    SpinBasis.size = {basis.size}")


def case_2_translation_times_spinflip() -> None:
    """Abelian direct product: translation × spin-flip via .product()."""
    print("\n--- Case 2: Z_L × Z_2 translation × spin-flip ---------------------")
    n_sites = 4

    # Build each factor as its own cyclic group, then take the direct
    # product — `product()` enumerates cross-terms (T^a · Z^b) with
    # composed characters χ_T(T^a) · χ_Z(Z^b).
    translation = SymmetryGroup(n_sites=n_sites, lhss=2)
    translation.add_cyclic(Lattice([(i + 1) % n_sites for i in range(n_sites)]), k=0)
    spin_flip = SymmetryGroup(n_sites=n_sites, lhss=2)
    spin_flip.add_cyclic(Local([1, 0]), eta=-1)

    group = translation.product(spin_flip)
    _print_group("Z_4 (k=0) × Z_2 (eta=-1)", group)

    basis = SpinBasis.symmetric(group, _xx_chain(n_sites), seeds=["0011"])
    print(f"    SpinBasis.size = {basis.size}")


def case_3_pz_composite() -> None:
    """PZ as a single composite generator (neither P nor Z is a symmetry alone)."""
    print("\n--- Case 3: PZ composite (single generator) -----------------------")
    n_sites = 4
    pz = Composite(perm=list(range(n_sites - 1, -1, -1)), perm_vals=[1, 0])
    group = SymmetryGroup(n_sites=n_sites, lhss=2)
    group.add_cyclic(pz, eta=-1)
    _print_group("⟨PZ⟩, eta=-1", group)

    basis = SpinBasis.symmetric(group, _xx_chain(n_sites), seeds=["0011"])
    print(f"    SpinBasis.size = {basis.size}")


def case_4_dihedral() -> None:
    """Non-abelian dihedral D_L = ⟨T, P⟩ via .close()."""
    print("\n--- Case 4: dihedral D_L = ⟨T, P⟩ (non-abelian closure) ----------")
    n_sites = 4
    T = Lattice([(i + 1) % n_sites for i in range(n_sites)])
    P = Lattice(list(range(n_sites - 1, -1, -1)))

    group = SymmetryGroup(n_sites=n_sites, lhss=2)
    # trivial rep: χ ≡ 1 on every element
    group.close(generators=[T, P], char=lambda elem: 1.0)
    _print_group(f"D_{n_sites}, trivial rep", group)

    basis = SpinBasis.symmetric(group, _xx_chain(n_sites), seeds=["0011"])
    print(f"    SpinBasis.size = {basis.size}")


def case_5_compatibility_error() -> None:
    """FermionBasis rejects a SymmetryGroup with lhss != 2."""
    print("\n--- Case 5: FermionBasis rejects lhss != 2 ------------------------")
    bad_group = SymmetryGroup(n_sites=4, lhss=3)
    H = FermionOperator([("+", [[1.0, 0]])])
    try:
        FermionBasis.symmetric(bad_group, H, seeds=["0000"])
    except TypeError as exc:
        print(f"    TypeError raised as expected: {exc}")


def main() -> None:
    case_1_translation_only()
    case_2_translation_times_spinflip()
    case_3_pz_composite()
    case_4_dihedral()
    case_5_compatibility_error()


if __name__ == "__main__":
    main()
