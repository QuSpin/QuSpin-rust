"""User-facing SymmetryGroup for *Basis.symmetric(...)."""

from __future__ import annotations

import cmath
from collections.abc import Callable, Iterator
from math import gcd, pi
from typing import TYPE_CHECKING

from quspin_rs._rs import _compose, _order, _validate_group

if TYPE_CHECKING:
    from quspin_rs._rs import SymElement


def _root_of_unity(p: int, q: int) -> complex:
    """Compute exp(2πi · p / q) with bit-exact results for q | 4.

    For q=1 returns ``1``; q=2 returns ``±1``; q=4 returns ``±1`` or ``±i``;
    otherwise falls back to :func:`cmath.exp`. Reducing ``p / q`` by
    ``gcd(p, q)`` avoids spurious rounding for representations like
    ``2/4 == 1/2``.
    """
    g = gcd(p, q) if (p != 0 and q != 0) else max(abs(p), abs(q), 1)
    p //= g
    q //= g
    p %= q
    if q == 1:
        return complex(1)
    if q == 2:
        return complex(-1)
    if q == 4:
        return (complex(1), 1j, complex(-1), -1j)[p]
    return cmath.exp(2j * pi * p / q)


class SymmetryGroup:
    """Collection of (element, character) pairs describing a symmetry group.

    Construct with ``(n_sites, lhss)``, then add elements via :meth:`add`,
    :meth:`add_cyclic`, :meth:`close`, or :meth:`product`. Pass the
    result as the first positional argument to
    ``*Basis.symmetric(group, ham, seeds)``.
    """

    def __init__(self, n_sites: int, lhss: int) -> None:
        if n_sites < 1:
            raise ValueError(f"n_sites must be >= 1, got {n_sites}")
        if lhss < 2:
            raise ValueError(f"lhss must be >= 2, got {lhss}")
        self._n_sites = n_sites
        self._lhss = lhss
        self._elements: list[tuple["SymElement", complex]] = []

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def lhss(self) -> int:
        return self._lhss

    def add(self, element: "SymElement", character: complex) -> None:
        """Add a single non-identity element with its 1-D-rep character."""
        self._elements.append((element, complex(character)))

    def add_cyclic(
        self,
        generator: "SymElement",
        *,
        k: int | None = None,
        eta: int | None = None,
        char: complex | None = None,
    ) -> None:
        """Add g, g², …, g^(N-1) where N is the generator's computed order.

        Exactly one of ``{k, eta, char}`` must be supplied:

        - ``k=int``    χ(g^a) = exp(-2πi · k · a / N), any cyclic
        - ``eta=±1``   χ(g^a) = η^a, requires N == 2
        - ``char=z``   χ(g^a) = z^a (user picks any consistent rep)

        Raises ``ValueError`` if more than one (or none) of ``k`` / ``eta``
        / ``char`` is supplied, if ``eta`` is given on a non-order-2
        generator, or if the generator is the identity (computed order
        less than 2).
        """
        supplied = sum(x is not None for x in (k, eta, char))
        if supplied != 1:
            raise ValueError(
                "add_cyclic requires exactly one of {k, eta, char}, " f"got {supplied}"
            )

        order = _order(generator, self._n_sites, self._lhss)
        if order < 2:
            raise ValueError("add_cyclic generator has order < 2 (identity element)")
        if eta is not None:
            if order != 2:
                raise ValueError(f"eta=±1 requires order 2, got order={order}")
            if eta not in (1, -1):
                raise ValueError(f"eta must be ±1, got {eta}")
            base_char: complex = complex(eta)
        elif k is not None:
            if not (0 <= k < order):
                raise ValueError(f"k must be in [0, {order}), got k={k}")
            base_char = _root_of_unity(-k, order)
        else:
            assert char is not None
            base_char = complex(char)

        # Enumerate g, g², …, g^(N-1) via repeated composition.
        g_pow: "SymElement" = generator
        for a in range(1, order):
            self._elements.append((g_pow, base_char**a))
            if a + 1 < order:
                g_pow = _compose(g_pow, generator)

    def close(
        self,
        generators: list["SymElement"],
        char: Callable[["SymElement"], complex],
    ) -> None:
        """BFS-close the orbit under composition.

        For each generator and each element discovered so far, compose
        them and append any new (i.e. previously-unseen) action with the
        user-supplied character. The user is responsible for supplying a
        self-consistent 1-D representation; :meth:`validate` catches
        inconsistencies.

        A no-op when ``generators`` is empty.

        Compositions that produce the identity element are dropped (the
        identity is implicit in every group).
        """
        if not generators:
            return

        seen: set["SymElement"] = set()
        frontier: list["SymElement"] = []
        for g in generators:
            if g not in seen:
                seen.add(g)
                frontier.append(g)
                self._elements.append((g, complex(char(g))))

        while frontier:
            next_frontier: list["SymElement"] = []
            for x in frontier:
                for g in generators:
                    try:
                        composed = _compose(x, g)
                    except ValueError as exc:
                        # Only skip identity-composition results; re-raise length /
                        # locs / shape errors so malformed generators surface.
                        if "produced identity" not in str(exc):
                            raise
                        continue
                    if composed in seen:
                        continue
                    seen.add(composed)
                    next_frontier.append(composed)
                    self._elements.append((composed, complex(char(composed))))
            frontier = next_frontier

    def product(self, other: "SymmetryGroup") -> "SymmetryGroup":
        """Out-of-place direct product. Both groups must share
        ``(n_sites, lhss)``. Caller asserts the factors commute;
        :meth:`validate` catches non-commuting products at first build.

        The result contains every element of ``self``, every element of
        ``other``, plus every cross product ``_compose(a, b)`` with
        character ``χ_a · χ_b``. Compositions producing the identity are
        dropped (the identity is implicit in every group).

        ``self`` and ``other`` are not mutated.
        """
        if (self._n_sites, self._lhss) != (other._n_sites, other._lhss):
            raise ValueError(
                f"product: factor groups must share (n_sites, lhss); "
                f"got {(self._n_sites, self._lhss)} vs "
                f"{(other._n_sites, other._lhss)}"
            )
        out = SymmetryGroup(self._n_sites, self._lhss)
        # Self-only.
        for elem, chi in self._elements:
            out._elements.append((elem, chi))
        # Other-only.
        for elem, chi in other._elements:
            out._elements.append((elem, chi))
        # Cross terms.
        for a_elem, chi_a in self._elements:
            for b_elem, chi_b in other._elements:
                try:
                    composed = _compose(a_elem, b_elem)
                except ValueError as exc:
                    # Skip only identity-composition results; re-raise
                    # length / locs / shape errors so malformed factors
                    # surface.
                    if "produced identity" not in str(exc):
                        raise
                    continue
                out._elements.append((composed, chi_a * chi_b))
        return out

    def validate(self) -> None:
        """Eagerly run SymBasis::validate_group: closure under
        composition + 1-D-rep character consistency.

        Raises :class:`ValueError` if the supplied elements don't form
        a closed group, the characters violate ``χ(g·h) = χ(g)·χ(h)``,
        or any two elements have the same action.

        ``*Basis.symmetric(group, ...)`` runs the same check
        implicitly on first build, so calling :meth:`validate`
        directly is opt-in early-feedback."""
        elements_for_rust = [
            (elem, (chi.real, chi.imag)) for elem, chi in self._elements
        ]
        _validate_group(elements_for_rust, self._n_sites, self._lhss)

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[tuple["SymElement", complex]]:
        return iter(self._elements)

    def __repr__(self) -> str:
        return (
            f"SymmetryGroup(n_sites={self._n_sites}, lhss={self._lhss}, "
            f"|G|={1 + len(self._elements)})"
        )
