"""User-facing SymmetryGroup for *Basis.symmetric(...)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quspin_rs._rs import SymElement


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

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[tuple["SymElement", complex]]:
        return iter(self._elements)

    def __repr__(self) -> str:
        return (
            f"SymmetryGroup(n_sites={self._n_sites}, lhss={self._lhss}, "
            f"|G|={1 + len(self._elements)})"
        )
