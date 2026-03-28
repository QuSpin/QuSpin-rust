"""Type stubs for the ``quspin_rs._rs`` extension module (QuSpin Rust core)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# PySpinSymGrp
# ---------------------------------------------------------------------------


class PySpinSymGrp:
    """A spin-symmetry group: lattice permutations + spin-inversion / bit-flip ops.

    Mutable builder — construct with ``lhss`` and ``n_sites``, then call
    :meth:`add_lattice` and :meth:`add_inverse` to add symmetry elements.

    For LHSS = 2: local operations are XOR bit-flips (Z₂ symmetry).
    For LHSS > 2: local operations map ``v → lhss − v − 1`` (spin inversion).

    Use :class:`PyDitSymGrp` for local value-permutation symmetries.

    Example:
        >>> grp = PySpinSymGrp(lhss=2, n_sites=4)
        >>> grp.add_lattice(grp_char=1.0 + 0j, perm=[1, 2, 3, 0])
        >>> grp.add_inverse(grp_char=-1.0 + 0j, locs=[0, 1, 2, 3])
        >>> grp.n_sites
        4
    """

    def __init__(self, lhss: int, n_sites: int) -> None:
        """Construct an empty spin-symmetry group.

        Args:
            lhss (int): Local Hilbert-space size (e.g. ``2`` for spin-1/2,
                ``3`` for spin-1).
            n_sites (int): Number of lattice sites.

        Raises:
            ValueError: If ``lhss = 2`` and ``n_sites > 8192``.
        """
        ...

    def add_lattice(self, grp_char: complex, perm: list[int]) -> None:
        """Add a lattice (site-permutation) symmetry element.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator, e.g. ``1+0j`` for even parity, ``-1+0j`` for odd).
            perm (list[int]): Forward site permutation where ``perm[src] = dst``.
                Must have length ``n_sites``.
        """
        ...

    def add_inverse(self, grp_char: complex, locs: list[int]) -> None:
        """Add a spin-inversion / bit-flip symmetry element.

        For LHSS = 2: XOR-flips the bits at the specified site indices.
        For LHSS > 2: maps ``v → lhss − v − 1`` at the specified sites.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator).
            locs (list[int]): Site indices to which the operation is applied.
        """
        ...

    @property
    def n_sites(self) -> int:
        """Number of lattice sites."""
        ...

    @property
    def lhss(self) -> int:
        """Local Hilbert-space size."""
        ...

    def __repr__(self) -> str:
        """Return ``PySpinSymGrp(lhss=..., n_sites=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyDitSymGrp
# ---------------------------------------------------------------------------


class PyDitSymGrp:
    """A dit symmetry group: lattice permutations + local value-permutation ops.

    Mutable builder — construct with ``lhss`` and ``n_sites``, then call
    :meth:`add_lattice` and :meth:`add_local_perm` to add symmetry elements.

    Only supported for LHSS ≥ 3. Use :class:`PySpinSymGrp` for LHSS = 2 or
    for spin-inversion symmetries.

    Example:
        >>> grp = PyDitSymGrp(lhss=3, n_sites=4)
        >>> grp.add_lattice(grp_char=1.0 + 0j, perm=[1, 2, 3, 0])
        >>> grp.add_local_perm(grp_char=1.0 + 0j, perm=[2, 1, 0], locs=[0, 1, 2, 3])
        >>> grp.lhss
        3
    """

    def __init__(self, lhss: int, n_sites: int) -> None:
        """Construct an empty dit symmetry group.

        Args:
            lhss (int): Local Hilbert-space size. Must be ≥ 3.
            n_sites (int): Number of lattice sites.

        Raises:
            ValueError: If ``lhss < 3`` (use :class:`PySpinSymGrp` instead).
        """
        ...

    def add_lattice(self, grp_char: complex, perm: list[int]) -> None:
        """Add a lattice (site-permutation) symmetry element.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator, e.g. ``1+0j`` for even parity, ``-1+0j`` for odd).
            perm (list[int]): Forward site permutation where ``perm[src] = dst``.
                Must have length ``n_sites``.
        """
        ...

    def add_local_perm(
        self, grp_char: complex, perm: list[int], locs: list[int]
    ) -> None:
        """Add an on-site value-permutation symmetry element.

        Maps local occupation ``v → perm[v]`` at each site in ``locs``.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator).
            perm (list[int]): Value permutation of length ``lhss``.
                Entry ``i`` is the image of dit value ``i``.
            locs (list[int]): Site indices to which the permutation is applied.
        """
        ...

    @property
    def n_sites(self) -> int:
        """Number of lattice sites."""
        ...

    @property
    def lhss(self) -> int:
        """Local Hilbert-space size."""
        ...

    def __repr__(self) -> str:
        """Return ``PyDitSymGrp(lhss=..., n_sites=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyFermionicSymGrp
# ---------------------------------------------------------------------------


class PyFermionicSymGrp:
    """A fermionic symmetry group: lattice permutations with Jordan-Wigner sign
    tracking.

    Mutable builder — construct with ``n_sites``, then call :meth:`add_lattice`
    to add symmetry elements.

    All lattice elements automatically include the fermionic permutation sign
    based on the pre-image state, implementing the Jordan-Wigner transformation
    for site-permutation symmetries.

    Use :class:`PySpinSymGrp` for bosonic systems.

    Example:
        >>> grp = PyFermionicSymGrp(n_sites=4)
        >>> grp.add_lattice(grp_char=1.0 + 0j, perm=[1, 2, 3, 0])
        >>> grp.n_sites
        4
    """

    def __init__(self, n_sites: int) -> None:
        """Construct an empty fermionic symmetry group.

        Args:
            n_sites (int): Number of lattice sites. Maximum value is 8192.

        Raises:
            ValueError: If ``n_sites > 8192``.
        """
        ...

    def add_lattice(self, grp_char: complex, perm: list[int]) -> None:
        """Add a lattice (site-permutation) symmetry element with fermionic sign
        tracking.

        The Jordan-Wigner sign of the permutation acting on the pre-image state
        is automatically included in the group character.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator, e.g. ``1+0j`` for even parity, ``-1+0j`` for odd).
            perm (list[int]): Forward site permutation where ``perm[src] = dst``.
                Must have length ``n_sites``.
        """
        ...

    @property
    def n_sites(self) -> int:
        """Number of lattice sites."""
        ...

    def __repr__(self) -> str:
        """Return ``PyFermionicSymGrp(n_sites=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyHardcoreHamiltonian
# ---------------------------------------------------------------------------


class PyHardcoreHamiltonian:
    """A hardcore-boson / spin-1/2 Hamiltonian defined by operator strings.

    Stores operator terms grouped by coefficient index (``cindex``). Each term
    is a product of single-site operators applied to specified sites, scaled by
    a complex coupling coefficient.

    Supported operator characters:

    - ``'x'``: Pauli-X (σˣ)
    - ``'y'``: Pauli-Y (σʸ)
    - ``'z'``: Pauli-Z (σᶻ)
    - ``'+'``: raising operator (σ⁺)
    - ``'-'``: lowering operator (σ⁻)
    - ``'n'``: number operator (n = (1 + σᶻ) / 2)

    The number of sites is inferred from the largest site index encountered.

    Example:
        >>> J, h = 1.0, 0.5
        >>> H = PyHardcoreHamiltonian([
        ...     [("xx", [(J, 0, 1), (J, 1, 2)])],   # cindex 0: hopping
        ...     [("z",  [(h, 0), (h, 1), (h, 2)])],  # cindex 1: on-site field
        ... ])
        >>> H.max_site
        2
        >>> H.num_cindices
        2
    """

    def __init__(
        self,
        terms: list[list[tuple[str, list[tuple[Any, ...]]]]],
    ) -> None:
        """Construct a Hamiltonian from a nested list of operator terms.

        Args:
            terms (list[list[tuple[str, list[tuple]]]]): Outer list indexed by
                ``cindex``. Each element is a list of ``(op_str, coupling_list)``
                pairs where:

                - ``op_str`` (str): Operator string — one character per site
                  acted on (``'x'``, ``'y'``, ``'z'``, ``'+'``, ``'-'``,
                  ``'n'``).
                - ``coupling_list`` (list[tuple]): Each element is
                  ``(coeff, site_0, site_1, ...)`` with exactly one site index
                  per character in ``op_str``. ``coeff`` may be ``complex``,
                  ``float``, or ``int``.

        Raises:
            ValueError: If ``op_str`` contains an unknown operator character,
                if the number of site indices does not match ``len(op_str)``,
                or if the input structure is malformed.
        """
        ...

    @property
    def max_site(self) -> int:
        """Maximum site index across all operator strings."""
        ...

    @property
    def num_cindices(self) -> int:
        """Number of distinct coefficient indices (outer list length)."""
        ...

    def __repr__(self) -> str:
        """Return ``PyHardcoreHamiltonian(max_site=..., num_cindices=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyBondTerm
# ---------------------------------------------------------------------------


class PyBondTerm:
    """A single term in a ``PyBondHamiltonian``: a dense matrix and site-pair bonds.

    Performs shallow validation at construction (2-D ``complex128`` array,
    list of ``(int, int)`` pairs).  Semantic validation (perfect-square
    dimension, ``lhss`` range, site bounds) is deferred to
    ``PyBondHamiltonian``.

    Example:
        >>> import numpy as np
        >>> M = np.eye(4, dtype=complex)
        >>> t = PyBondTerm(M, [(0, 1), (1, 2)])
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.complexfloating],
        bonds: list[tuple[int, int]],
    ) -> None:
        """Create a bond term.

        Args:
            matrix (NDArray): 2-D array with ``dtype=complex128`` and shape
                ``(lhss², lhss²)``.
            bonds (list[tuple[int, int]]): Site pairs ``(si, sj)`` to apply
                the matrix to.

        Raises:
            ValueError: If ``matrix`` is not a 2-D ``complex128`` NumPy array
                or ``bonds`` is not a list of ``(int, int)`` pairs.
        """
        ...

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Shape of the interaction matrix as ``(rows, cols)``."""
        ...

    @property
    def bonds(self) -> list[tuple[int, int]]:
        """Site-pair bonds applied by this term."""
        ...

    def __repr__(self) -> str:
        """Return ``PyBondTerm(matrix_shape=(...), n_bonds=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyBondHamiltonian
# ---------------------------------------------------------------------------


class PyBondHamiltonian:
    """A Hamiltonian built from dense two-site interaction matrices.

    Each term specifies a single ``(lhss² × lhss²)`` matrix applied to a list
    of site pairs.  All terms share the same ``lhss`` (local Hilbert-space
    size), which is inferred from the shape of the first term's matrix.

    ``max_site`` is inferred from the largest site index across all bonds.

    Example:
        >>> import numpy as np
        >>> M = np.zeros((4, 4), dtype=complex)
        >>> M[3, 0] = M[2, 1] = M[1, 2] = M[0, 3] = 1.0  # XX
        >>> t = PyBondTerm(M, [(0, 1), (1, 2), (2, 3)])
        >>> H = PyBondHamiltonian([t])
        >>> H.max_site
        3
        >>> H.lhss
        2
    """

    def __init__(self, terms: list[PyBondTerm]) -> None:
        """Construct a BondHamiltonian from a list of ``PyBondTerm`` objects.

        Each term is assigned a ``cindex`` equal to its position in the list.
        ``max_site`` is inferred as the largest site index across all bonds.

        Args:
            terms (list[PyBondTerm]): One ``PyBondTerm`` per ``cindex``.

        Raises:
            ValueError: If ``terms`` is empty, matrices have inconsistent
                shapes, or ``lhss`` is out of range.
        """
        ...

    @property
    def max_site(self) -> int:
        """Maximum site index across all bonds."""
        ...

    @property
    def num_cindices(self) -> int:
        """Number of distinct coefficient indices (length of the terms list)."""
        ...

    @property
    def lhss(self) -> int:
        """Local Hilbert-space size, inferred from ``sqrt(matrix.shape[0])``."""
        ...

    def __repr__(self) -> str:
        """Return ``PyBondHamiltonian(max_site=..., lhss=..., num_cindices=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyBosonHamiltonian
# ---------------------------------------------------------------------------


class PyBosonHamiltonian:
    """A bosonic Hamiltonian for LHSS ≥ 2 sites (truncated harmonic oscillator).

    Operators are ``'+'`` (a†), ``'-'`` (a), and ``'n'`` (n̂ = a†a).
    The number of sites is inferred from the largest site index encountered.

    Conventions:
    - a†|n⟩ = √(n+1)|n+1⟩  (zero if n = LHSS−1)
    - a|n⟩  = √n|n−1⟩       (zero if n = 0)
    - n̂|n⟩ = n|n⟩

    Example:
        >>> J, mu = 1.0, 0.5
        >>> H = PyBosonHamiltonian(lhss=3, terms=[
        ...     [("+-", [(J, 0, 1), (J, 1, 2)])],   # cindex 0: hopping a†_i a_j
        ...     [("-+", [(J, 0, 1), (J, 1, 2)])],   # cindex 1: hopping a_i a†_j
        ...     [("n",  [(mu, 0), (mu, 1), (mu, 2)])],  # cindex 2: on-site n
        ... ])
        >>> H.lhss
        3
        >>> H.max_site
        2
    """

    def __init__(
        self,
        lhss: int,
        terms: list[list[tuple[str, list[tuple[Any, ...]]]]],
    ) -> None:
        """Construct a bosonic Hamiltonian.

        Args:
            lhss (int): Local Hilbert-space size (number of levels per site).
                Must be ≥ 2.
            terms (list[list[tuple[str, list[tuple]]]]): Outer list indexed by
                ``cindex``. Each element is a list of ``(op_str, coupling_list)``
                pairs where:

                - ``op_str`` (str): Operator string — one character per site
                  acted on (``'+'``, ``'-'``, ``'n'``).
                - ``coupling_list`` (list[tuple]): Each element is
                  ``(coeff, site_0, site_1, ...)`` with exactly one site index
                  per character in ``op_str``.

        Raises:
            ValueError: If ``lhss < 2``, if ``op_str`` contains an unknown
                operator character, or if the input structure is malformed.
        """
        ...

    @property
    def lhss(self) -> int:
        """Local Hilbert-space size (number of levels per site)."""
        ...

    @property
    def max_site(self) -> int:
        """Maximum site index across all operator strings."""
        ...

    @property
    def num_cindices(self) -> int:
        """Number of distinct coefficient indices (outer list length)."""
        ...

    def __repr__(self) -> str:
        """Return ``PyBosonHamiltonian(lhss=..., max_site=..., num_cindices=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyFermionHamiltonian
# ---------------------------------------------------------------------------


class PyFermionHamiltonian:
    """A fermionic Hamiltonian defined by creation/annihilation operator strings.

    Jordan-Wigner signs are accumulated automatically during matrix construction.
    Reuses the hardcore (LHSS=2) basis; orbital labelling convention:
    site ``2*i`` = spin-down orbital ``i``, site ``2*i+1`` = spin-up orbital ``i``.

    Supported operator characters:

    - ``'+'``: creation operator c†
    - ``'-'``: annihilation operator c
    - ``'n'``: number operator n̂

    Example:
        >>> t = 1.0
        >>> H = PyFermionHamiltonian([
        ...     [("+-", [(t, 0, 1), (t, 1, 0)])],  # cindex 0: hopping
        ... ])
        >>> H.max_site
        1
        >>> H.num_cindices
        1
    """

    def __init__(
        self,
        terms: list[list[tuple[str, list[tuple[Any, ...]]]]],
    ) -> None:
        """Construct a fermionic Hamiltonian from a nested list of operator terms.

        Args:
            terms (list[list[tuple[str, list[tuple]]]]): Outer list indexed by
                ``cindex``. Each element is a list of ``(op_str, coupling_list)``
                pairs where:

                - ``op_str`` (str): Operator string — one character per site
                  acted on (``'+'`` for c†, ``'-'`` for c, ``'n'`` for n̂).
                - ``coupling_list`` (list[tuple]): Each element is
                  ``(coeff, site_0, site_1, ...)`` with exactly one site index
                  per character in ``op_str``. ``coeff`` may be ``complex``,
                  ``float``, or ``int``.

        Raises:
            ValueError: If ``op_str`` contains an unknown operator character,
                if the number of site indices does not match ``len(op_str)``,
                or if the input structure is malformed.
        """
        ...

    @property
    def max_site(self) -> int:
        """Maximum site index across all operator strings."""
        ...

    @property
    def num_cindices(self) -> int:
        """Number of distinct coefficient indices (outer list length)."""
        ...

    def __repr__(self) -> str:
        """Return ``PyFermionHamiltonian(max_site=..., num_cindices=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyHardcoreBasis
# ---------------------------------------------------------------------------

# A seed state: a bit-string ("0110") or a list of 0/1 ints ([0, 1, 1, 0]).
# Position i gives the occupation of site i.
_Seed = str | list[int]


class PyHardcoreBasis:
    """A basis for a hardcore-boson / spin-1/2 Hilbert space.

    Represents one of three basis types, built via static factory methods:

    - **Full**: all :math:`2^{n_{\\mathrm{sites}}}` computational basis states.
    - **Subspace**: the sector reachable from given seed states under a
      Hamiltonian (e.g., fixed particle number).
    - **Symmetric**: a symmetry-reduced sector using a :class:`PySpinSymGrp`.

    Example:
        >>> basis = PyHardcoreBasis.full(4)
        >>> basis.size
        16

        >>> basis = PyHardcoreBasis.subspace(["1100", "0110"], ham)
        >>> basis = PyHardcoreBasis.symmetric(["1100"], ham, grp)
    """

    @staticmethod
    def full(n_sites: int) -> PyHardcoreBasis:
        """Build the full Hilbert space of ``n_sites`` spin-1/2 sites.

        Contains all :math:`2^{n_{\\mathrm{sites}}}` computational basis states.
        Only supported for ``n_sites ≤ 64``.

        Args:
            n_sites (int): Number of lattice sites. Maximum value is 64.

        Returns:
            PyHardcoreBasis: Full-space basis with
            :math:`2^{n_{\\mathrm{sites}}}` states.

        Raises:
            ValueError: If ``n_sites > 64``.
        """
        ...

    @staticmethod
    def subspace(
        seeds: Iterable[_Seed],
        ham: PyHardcoreHamiltonian,
    ) -> PyHardcoreBasis:
        """Build the subspace reachable from seed states under a Hamiltonian.

        Starting from each seed, repeatedly applies the Hamiltonian to discover
        all connected basis states (e.g., a fixed-particle-number sector).

        Args:
            seeds (Iterable[str | list[int]]): Initial states. Each element is
                either:

                - A ``str`` of ``'0'``/``'1'`` characters, e.g. ``"1100"``.
                - A ``list[int]`` of ``0``/``1`` values, e.g. ``[1, 1, 0, 0]``.

                In both cases position ``i`` gives the occupation of site ``i``.
            ham (PyHardcoreHamiltonian): The Hamiltonian whose connectivity
                defines the sector.

        Returns:
            PyHardcoreBasis: Subspace basis containing all states reachable
            from any seed.

        Raises:
            ValueError: If any seed contains invalid characters or values, or
                if ``n_sites`` exceeds 8192.
        """
        ...

    @staticmethod
    def symmetric(
        seeds: Iterable[_Seed],
        ham: PyHardcoreHamiltonian,
        grp: PySpinSymGrp,
    ) -> PyHardcoreBasis:
        """Build a symmetry-reduced subspace.

        Like :meth:`subspace`, but projects into a symmetry sector defined by
        ``grp``, yielding a smaller basis. Requires a spin-symmetry group with
        LHSS = 2.

        Args:
            seeds (Iterable[str | list[int]]): Initial states (same format as
                :meth:`subspace`).
            ham (PyHardcoreHamiltonian): The Hamiltonian defining connectivity.
            grp (PySpinSymGrp): The symmetry group defining the sector.
                Must have ``lhss = 2``.

        Returns:
            PyHardcoreBasis: Symmetry-reduced basis.

        Raises:
            ValueError: If ``ham.n_sites != grp.n_sites``, if ``grp.lhss != 2``,
                if any seed is malformed, or if ``n_sites`` exceeds 8192.
        """
        ...

    @staticmethod
    def symmetric_fermionic(
        seeds: Iterable[_Seed],
        ham: PyHardcoreHamiltonian,
        grp: PyFermionicSymGrp,
    ) -> PyHardcoreBasis:
        """Build a symmetry-reduced subspace for a fermionic system.

        Like :meth:`symmetric`, but accepts a :class:`PyFermionicSymGrp` whose
        lattice elements include Jordan-Wigner permutation signs.

        Args:
            seeds (Iterable[str | list[int]]): Initial states (same format as
                :meth:`subspace`).
            ham (PyHardcoreHamiltonian): The Hamiltonian defining connectivity.
            grp (PyFermionicSymGrp): The fermionic symmetry group.

        Returns:
            PyHardcoreBasis: Symmetry-reduced fermionic basis.

        Raises:
            ValueError: If ``ham.n_sites != grp.n_sites``, if any seed is
                malformed, or if ``n_sites`` exceeds 8192.
        """
        ...

    def state_at(self, i: int) -> str:
        """Return the ``i``-th basis state as a bit string.

        Character at position ``j`` is ``'1'`` if site ``j`` is occupied,
        ``'0'`` otherwise. The ordering matches the seed convention used in
        :meth:`subspace` and :meth:`symmetric`.

        Args:
            i (int): Row index, ``0 ≤ i < size``.

        Returns:
            str: Bit string of length ``n_sites``.

        Raises:
            IndexError: If ``i`` is out of range.

        Example:
            >>> basis = PyHardcoreBasis.full(2)
            >>> basis.state_at(0)
            '11'
            >>> basis.state_at(3)
            '00'
        """
        ...

    def index(self, state: _Seed) -> int | None:
        """Look up the row index of a basis state.

        Args:
            state (str | list[int]): Basis state in the same format accepted
                by :meth:`subspace` — a ``'0'``/``'1'`` string or a
                ``list[int]`` of ``0``/``1`` values, where position ``j``
                gives the occupation of site ``j``.

        Returns:
            int | None: The row index of ``state`` in the basis, or ``None``
            if the state is not present.

        Raises:
            ValueError: If ``state`` is malformed.

        Example:
            >>> basis = PyHardcoreBasis.full(2)
            >>> basis.index("11")
            0
            >>> basis.index("00")
            3
            >>> basis.index([1, 1])
            0
        """
        ...

    @property
    def n_sites(self) -> int:
        """Number of lattice sites."""
        ...

    @property
    def size(self) -> int:
        """Number of basis states."""
        ...

    def __repr__(self) -> str:
        """Return ``PyHardcoreBasis(kind=..., n_sites=..., size=...)``."""
        ...

    def __str__(self) -> str:
        """Return a human-readable enumeration of basis states.

        Format::

            kind(n_sites=N, size=M, symmetries=[...]):
               0. |01001>
               1. |10011>
              ...
              48. |11000>
              49. |00111>

        If ``size > 50``, only the first 25 and last 25 states are shown,
        separated by a ``...`` line. The index column is right-aligned to the
        width of the largest index. Symmetries is ``[]`` for full and subspace
        bases, and ``[symmetric]`` for symmetry-reduced bases.
        """
        ...


# ---------------------------------------------------------------------------
# PyDitBasis
# ---------------------------------------------------------------------------

# A dit seed: a decimal digit string ("012") or a list of ints ([0, 1, 2]).
# Position i gives the occupation (0 ≤ value < lhss) of site i.
_DitSeed = str | list[int]


class PyDitBasis:
    """A basis for a bosonic (LHSS ≥ 2) Hilbert space.

    States are stored as packed dit integers: each site occupies
    ``ceil(log2(lhss))`` bits.

    Seed strings are decimal digit sequences, e.g. ``"012"`` for a 3-site
    system with lhss=3. Seed lists are ``list[int]`` with values in ``0..lhss``.

    Basis types:

    - **Full**: all ``lhss^n_sites`` computational basis states (total bits ≤ 64).
    - **Subspace**: the sector reachable from given seed states under a
      :class:`PyBosonHamiltonian`.

    Example:
        >>> basis = PyDitBasis.full(n_sites=3, lhss=3)
        >>> basis.size
        27
        >>> basis.state_at(0)
        '222'
    """

    @staticmethod
    def full(n_sites: int, lhss: int) -> PyDitBasis:
        """Build the full bosonic Hilbert space.

        Contains all ``lhss^n_sites`` computational basis states.
        Requires ``n_sites * ceil(log2(lhss)) ≤ 64``.

        Args:
            n_sites (int): Number of lattice sites.
            lhss (int): Local Hilbert-space size. Must be ≥ 2.

        Returns:
            PyDitBasis: Full-space basis with ``lhss^n_sites`` states.

        Raises:
            ValueError: If ``lhss < 2`` or total bits exceed 64.
        """
        ...

    @staticmethod
    def subspace(
        seeds: Iterable[_DitSeed],
        ham: PyBosonHamiltonian,
    ) -> PyDitBasis:
        """Build the subspace reachable from seed states under a Hamiltonian.

        Args:
            seeds (Iterable[str | list[int]]): Initial states. Each element is
                either a decimal digit string (e.g. ``"012"``) or a
                ``list[int]`` with values in ``0..lhss``.
            ham (PyBosonHamiltonian): The Hamiltonian whose connectivity
                defines the sector.

        Returns:
            PyDitBasis: Subspace basis.

        Raises:
            ValueError: If any seed is malformed or total bits exceed 8192.
        """
        ...

    def state_at(self, i: int) -> str:
        """Return the ``i``-th basis state as a decimal digit string.

        Args:
            i (int): Row index, ``0 ≤ i < size``.

        Returns:
            str: Decimal digit string of length ``n_sites``.

        Raises:
            IndexError: If ``i`` is out of range.
        """
        ...

    def index(self, state: _DitSeed) -> int | None:
        """Look up the row index of a dit basis state.

        Args:
            state (str | list[int]): Basis state — a decimal digit string or
                a ``list[int]`` with values in ``0..lhss``.

        Returns:
            int | None: Row index, or ``None`` if the state is not present.

        Raises:
            ValueError: If ``state`` is malformed.
        """
        ...

    @property
    def n_sites(self) -> int:
        """Number of lattice sites."""
        ...

    @property
    def size(self) -> int:
        """Number of basis states."""
        ...

    @property
    def lhss(self) -> int:
        """Local Hilbert-space size."""
        ...

    def __repr__(self) -> str:
        """Return ``PyDitBasis(lhss=..., n_sites=..., size=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyQMatrix
# ---------------------------------------------------------------------------


class PyQMatrix:
    """A sparse quantum matrix in a custom CSR-like format.

    Stores a sparse operator matrix built from a :class:`PyHardcoreHamiltonian`
    and a :class:`PyHardcoreBasis`. Supports matrix-vector products and
    element-wise arithmetic.

    The element dtype is fixed at build time and must match all arrays passed
    to :meth:`dot` and :meth:`dot_transpose`.

    Supported dtypes: ``int8``, ``int16``, ``float32``, ``float64``,
    ``complex64``, ``complex128``.

    Example:
        >>> mat = PyQMatrix.build_hardcore_hamiltonian(ham, basis, np.dtype("float64"))
        >>> coeff = np.array([J, h], dtype=np.float64)
        >>> v_in = np.ones(basis.size, dtype=np.float64)
        >>> v_out = np.zeros(basis.size, dtype=np.float64)
        >>> mat.dot(coeff, v_in, v_out, overwrite=True)
    """

    @staticmethod
    def build_hardcore_hamiltonian(
        ham: PyHardcoreHamiltonian,
        basis: PyHardcoreBasis,
        dtype: np.dtype[Any],
    ) -> PyQMatrix:
        """Build a sparse matrix from a PyHardcoreHamiltonian and a basis.

        Args:
            ham (PyHardcoreHamiltonian): The Hamiltonian defining operator
                strings and coupling coefficients.
            basis (PyHardcoreBasis): The Hilbert space basis (full, subspace,
                or symmetric).
            dtype (numpy.dtype): NumPy dtype for matrix element storage.
                Supported values: ``np.dtype("int8")``, ``np.dtype("int16")``,
                ``np.dtype("float32")``, ``np.dtype("float64")``,
                ``np.dtype("complex64")``, ``np.dtype("complex128")``.

        Returns:
            PyQMatrix: Sparse matrix representation of the Hamiltonian in the
            given basis.

        Raises:
            ValueError: If ``ham.max_site >= basis.n_sites``, or if ``dtype``
                is not supported.
        """
        ...

    @staticmethod
    def build_bond_hamiltonian(
        ham: PyBondHamiltonian,
        basis: PyHardcoreBasis,
        dtype: np.dtype[Any],
    ) -> PyQMatrix:
        """Build a sparse matrix from a PyBondHamiltonian and a basis.

        Args:
            ham (PyBondHamiltonian): The Hamiltonian with dense two-site
                interaction matrices.
            basis (PyHardcoreBasis): The Hilbert space basis (full, subspace,
                or symmetric).
            dtype (numpy.dtype): NumPy dtype for matrix element storage.
                Supported values: ``np.dtype("int8")``, ``np.dtype("int16")``,
                ``np.dtype("float32")``, ``np.dtype("float64")``,
                ``np.dtype("complex64")``, ``np.dtype("complex128")``.

        Returns:
            PyQMatrix: Sparse matrix representation of the Hamiltonian in the
            given basis.

        Raises:
            ValueError: If ``ham.max_site >= basis.n_sites``, or if ``dtype``
                is not supported.
        """
        ...

    @staticmethod
    def build_boson_hamiltonian(
        ham: PyBosonHamiltonian,
        basis: PyDitBasis,
        dtype: np.dtype[Any],
    ) -> PyQMatrix:
        """Build a sparse matrix from a PyBosonHamiltonian and a dit basis.

        Args:
            ham (PyBosonHamiltonian): The bosonic Hamiltonian.
            basis (PyDitBasis): The dit Hilbert space basis (full or subspace).
            dtype (numpy.dtype): NumPy dtype for matrix element storage.
                Supported values: ``np.dtype("int8")``, ``np.dtype("int16")``,
                ``np.dtype("float32")``, ``np.dtype("float64")``,
                ``np.dtype("complex64")``, ``np.dtype("complex128")``.

        Returns:
            PyQMatrix: Sparse matrix representation of the Hamiltonian.

        Raises:
            ValueError: If ``ham.max_site >= basis.n_sites``,
                ``ham.lhss != basis.lhss``, or ``dtype`` is not supported.
        """
        ...

    @staticmethod
    def build_fermion_hamiltonian(
        ham: PyFermionHamiltonian,
        basis: PyHardcoreBasis,
        dtype: np.dtype[Any],
    ) -> PyQMatrix:
        """Build a sparse matrix from a PyFermionHamiltonian and a hardcore basis.

        Jordan-Wigner signs are applied automatically during matrix construction.

        Args:
            ham (PyFermionHamiltonian): The fermionic Hamiltonian.
            basis (PyHardcoreBasis): The Hilbert space basis (full, subspace,
                or symmetric). Uses the hardcore (LHSS=2) basis.
            dtype (numpy.dtype): NumPy dtype for matrix element storage.
                Supported values: ``np.dtype("int8")``, ``np.dtype("int16")``,
                ``np.dtype("float32")``, ``np.dtype("float64")``,
                ``np.dtype("complex64")``, ``np.dtype("complex128")``.

        Returns:
            PyQMatrix: Sparse matrix representation of the Hamiltonian.

        Raises:
            ValueError: If ``ham.max_site >= basis.n_sites``, or if ``dtype``
                is not supported.
        """
        ...

    def dot(
        self,
        coeff: npt.NDArray[Any],
        input: npt.NDArray[Any],
        output: npt.NDArray[Any],
        overwrite: bool = True,
    ) -> None:
        """Compute a matrix-vector product, accumulating into ``output``.

        Computes:

        .. code-block:: text

            output[row] = Σ_c coeff[c] * Σ_col M[c, row, col] * input[col]

        where the sum over ``c`` runs over all coefficient indices.

        Args:
            coeff (NDArray): 1-D array of length ``num_cindices``. dtype must
                match the matrix element type.
            input (NDArray): 1-D array of length ``dim``. dtype must match the
                matrix element type.
            output (NDArray): 1-D array of length ``dim``, modified in-place.
                dtype must match the matrix element type.
            overwrite (bool): If ``True`` (default), zero ``output`` before
                accumulating. If ``False``, add to existing values.

        Raises:
            TypeError: If any array dtype does not match the matrix element type.
            ValueError: If any array is not C-contiguous or has the wrong shape.
        """
        ...

    def dot_transpose(
        self,
        coeff: npt.NDArray[Any],
        input: npt.NDArray[Any],
        output: npt.NDArray[Any],
        overwrite: bool = True,
    ) -> None:
        """Compute a transpose matrix-vector product, accumulating into ``output``.

        Computes:

        .. code-block:: text

            output[col] = Σ_c coeff[c] * Σ_row M[c, row, col] * input[row]

        Args:
            coeff (NDArray): 1-D array of length ``num_cindices``. dtype must
                match the matrix element type.
            input (NDArray): 1-D array of length ``dim``. dtype must match the
                matrix element type.
            output (NDArray): 1-D array of length ``dim``, modified in-place.
                dtype must match the matrix element type.
            overwrite (bool): If ``True`` (default), zero ``output`` before
                accumulating. If ``False``, add to existing values.

        Raises:
            TypeError: If any array dtype does not match the matrix element type.
            ValueError: If any array is not C-contiguous or has the wrong shape.
        """
        ...

    def __add__(self, other: PyQMatrix) -> PyQMatrix:
        """Return the element-wise sum of two matrices.

        Args:
            other (PyQMatrix): Matrix to add. Must have the same dtype and
                dimension.

        Returns:
            PyQMatrix: New matrix equal to ``self + other``.

        Raises:
            ValueError: If the matrices have incompatible dtypes or dimensions.
        """
        ...

    def __sub__(self, other: PyQMatrix) -> PyQMatrix:
        """Return the element-wise difference of two matrices.

        Args:
            other (PyQMatrix): Matrix to subtract. Must have the same dtype and
                dimension.

        Returns:
            PyQMatrix: New matrix equal to ``self - other``.

        Raises:
            ValueError: If the matrices have incompatible dtypes or dimensions.
        """
        ...

    @property
    def dim(self) -> int:
        """Matrix dimension (number of rows and columns)."""
        ...

    @property
    def nnz(self) -> int:
        """Number of stored non-zero entries."""
        ...

    def to_dense(
        self,
        coeff: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Materialise the matrix as a dense 2-D NumPy array.

        Returns a C-contiguous ``(dim, dim)`` array where element
        ``[r, col] = Σ_c coeff[c] * M[c, r, col]``.

        Args:
            coeff (NDArray): 1-D array of length ``num_cindices``. dtype must
                match the matrix element type.

        Returns:
            NDArray: 2-D array of shape ``(dim, dim)`` with the same dtype as
            the matrix.

        Raises:
            TypeError: If ``coeff`` dtype does not match the matrix element type.
            ValueError: If ``coeff`` has the wrong length or is not
                C-contiguous.

        Example:
            >>> A = mat.to_dense(coeff)  # shape (dim, dim)
        """
        ...

    def to_csr(
        self,
        coeff: npt.NDArray[Any],
        drop_zeros: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[Any]]:
        """Materialise the matrix as a plain CSR sparse matrix.

        Multiplies stored values by ``coeff`` and sums entries that share the
        same ``(row, col)`` position across different operator strings.

        Args:
            coeff (NDArray): 1-D array of length ``num_cindices``. dtype must
                match the matrix element type.
            drop_zeros (bool): If ``True`` (default), omit entries whose
                accumulated value is exactly zero from the output arrays.

        Returns:
            tuple[NDArray[int64], NDArray[int64], NDArray]: ``(indptr, indices,
            data)`` arrays suitable for constructing a
            ``scipy.sparse.csr_array``:

            .. code-block:: python

                ip, idx, d = mat.to_csr(coeff)
                A = scipy.sparse.csr_array((d, idx, ip), shape=(mat.dim, mat.dim))

            - ``indptr``: ``int64``, length ``dim + 1``
            - ``indices``: ``int64``, length ``nnz``
            - ``data``: same dtype as the matrix, length ``nnz``

        Raises:
            TypeError: If ``coeff`` dtype does not match the matrix element type.
            ValueError: If ``coeff`` has the wrong length or is not
                C-contiguous.

        Example:
            >>> import scipy.sparse
            >>> ip, idx, d = mat.to_csr(coeff)
            >>> A = scipy.sparse.csr_array((d, idx, ip), shape=(mat.dim, mat.dim))
        """
        ...

    def __repr__(self) -> str:
        """Return ``PyQMatrix(dim=..., nnz=..., dtype=...)``."""
        ...
