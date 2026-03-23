"""Type stubs for the ``_rs`` extension module (quspin Rust core)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# PyLatticeElement
# ---------------------------------------------------------------------------


class PyLatticeElement:
    """A lattice symmetry element: a site permutation with a group character.

    Represents a spatial symmetry operation (e.g., translation or reflection)
    that permutes the lattice sites and carries an associated group character.

    Example:
        >>> T = PyLatticeElement(grp_char=1.0 + 0j, perm=[1, 2, 3, 0], lhss=2)
    """

    def __init__(
        self,
        grp_char: complex,
        perm: list[int],
        lhss: int,
    ) -> None:
        """Create a lattice symmetry element.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator, e.g., ``1+0j`` for even parity, ``-1+0j`` for odd).
            perm (list[int]): Forward site permutation where ``perm[src] = dst``.
                The length of ``perm`` determines the number of sites.
            lhss (int): Local Hilbert space size (e.g., ``2`` for spin-1/2,
                ``3`` for spin-1).
        """
        ...

    def __repr__(self) -> str:
        """Return ``PyLatticeElement(grp_char=..., perm=[...], lhss=...)``."""
        ...


# ---------------------------------------------------------------------------
# PyGrpElement
# ---------------------------------------------------------------------------


class PyGrpElement:
    """A local (on-site) symmetry group element with a group character.

    Encodes one of three types of local operation — bit-flip, value permutation,
    or spin inversion — before the concrete basis integer type is resolved.
    Use the static factory methods to construct instances.

    Example:
        >>> P = PyGrpElement.bitflip(grp_char=-1.0 + 0j, n_sites=4)
        >>> Q = PyGrpElement.spin_inversion(
        ...     grp_char=1.0 + 0j, n_sites=4, lhss=2, locs=[0, 1, 2, 3]
        ... )
    """

    @staticmethod
    def bitflip(
        grp_char: complex,
        n_sites: int,
        locs: list[int] | None = None,
    ) -> PyGrpElement:
        """Create a Z₂ bit-flip (XOR-with-mask) symmetry element.

        Applies ``state ^= mask`` where ``mask`` has a 1-bit at each site in
        ``locs``. For spin-1/2 this is a spin-flip symmetry.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator).
            n_sites (int): Total number of sites in the system.
            locs (list[int] | None): Site indices whose bits are flipped.
                If ``None``, all ``n_sites`` bits are flipped.

        Returns:
            PyGrpElement: A new bit-flip symmetry element.
        """
        ...

    @staticmethod
    def local_value(
        grp_char: complex,
        n_sites: int,
        lhss: int,
        perm: list[int],
        locs: list[int],
    ) -> PyGrpElement:
        """Create a local dit-value permutation symmetry element.

        For each site in ``locs``, maps the local occupation value
        ``v → perm[v]``. Requires ``len(perm) == lhss``.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator).
            n_sites (int): Total number of sites in the system.
            lhss (int): Local Hilbert space size (number of distinct dit values).
            perm (list[int]): Value permutation of length ``lhss``.
                Entry ``i`` is the image of dit value ``i``.
            locs (list[int]): Site indices to which the permutation is applied.

        Returns:
            PyGrpElement: A new local-value permutation symmetry element.
        """
        ...

    @staticmethod
    def spin_inversion(
        grp_char: complex,
        n_sites: int,
        lhss: int,
        locs: list[int],
    ) -> PyGrpElement:
        """Create a spin-inversion symmetry element.

        Maps each dit value ``v → lhss - v - 1`` at the specified sites.
        For spin-1/2 (``lhss=2``) this swaps ``0 ↔ 1``; for spin-1
        (``lhss=3``) it maps ``0 ↔ 2`` and fixes ``1``.

        Args:
            grp_char (complex): Group character (eigenvalue of the symmetry
                operator).
            n_sites (int): Total number of sites in the system.
            lhss (int): Local Hilbert space size.
            locs (list[int]): Site indices to which the inversion is applied.

        Returns:
            PyGrpElement: A new spin-inversion symmetry element.
        """
        ...

    def __repr__(self) -> str:
        """Return the factory-call form, e.g.
        ``PyGrpElement.bitflip(grp_char=(1+0j), n_sites=4, locs=None)``."""
        ...


# ---------------------------------------------------------------------------
# PySymmetryGrp
# ---------------------------------------------------------------------------


class PySymmetryGrp:
    """A symmetry group composed of lattice and local symmetry elements.

    Combines spatial (lattice) symmetries with on-site (local) symmetries.
    The concrete basis integer type is selected automatically at construction
    time based on ``n_sites``.

    Example:
        >>> T = PyLatticeElement(grp_char=1.0 + 0j, perm=[1, 2, 3, 0], lhss=2)
        >>> P = PyGrpElement.bitflip(grp_char=-1.0 + 0j, n_sites=4)
        >>> grp = PySymmetryGrp(lattice=[T], local=[P])
        >>> grp.n_sites
        4
    """

    def __init__(
        self,
        lattice: list[PyLatticeElement],
        local: list[PyGrpElement],
    ) -> None:
        """Construct a symmetry group from lattice and local elements.

        Validates that all elements agree on ``n_sites`` and eagerly builds
        the typed symmetry group.

        Args:
            lattice (list[PyLatticeElement]): Spatial symmetry elements (site
                permutations). The permutation length determines ``n_sites``.
            local (list[PyGrpElement]): On-site symmetry elements (bit-flip,
                value permutation, or spin inversion).

        Raises:
            ValueError: If any two elements disagree on the number of sites, or
                if ``n_sites`` exceeds 8192.
        """
        ...

    @property
    def n_sites(self) -> int:
        """Number of sites in the system."""
        ...

    def __repr__(self) -> str:
        """Return ``PySymmetryGrp(n_sites=...)``."""
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
        >>> H.n_sites
        3
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
    def n_sites(self) -> int:
        """Number of sites, inferred from the maximum site index plus one."""
        ...

    @property
    def num_cindices(self) -> int:
        """Number of distinct coefficient indices (outer list length)."""
        ...

    def __repr__(self) -> str:
        """Return ``PyHardcoreHamiltonian(n_sites=..., num_cindices=...)``."""
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
    - **Symmetric**: a symmetry-reduced sector using a :class:`PySymmetryGrp`.

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
        grp: PySymmetryGrp,
    ) -> PyHardcoreBasis:
        """Build a symmetry-reduced subspace.

        Like :meth:`subspace`, but projects into a symmetry sector defined by
        ``grp``, yielding a smaller basis.

        Args:
            seeds (Iterable[str | list[int]]): Initial states (same format as
                :meth:`subspace`).
            ham (PyHardcoreHamiltonian): The Hamiltonian defining connectivity.
            grp (PySymmetryGrp): The symmetry group defining the sector.

        Returns:
            PyHardcoreBasis: Symmetry-reduced basis.

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
        """Return a human-readable enumeration of all basis states.

        Format::

            kind(n_sites=N, size=M, symmetries=[...]):
              0. |01001>
              1. |10011>
              ...

        The index column is right-aligned to the width of the largest index.
        Symmetries is ``[]`` for full and subspace bases, and
        ``[symmetric]`` for symmetry-reduced bases.
        """
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
        """Build a sparse matrix from a Hamiltonian and a basis.

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
            ValueError: If ``ham.n_sites != basis.n_sites``, or if ``dtype``
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

    def __repr__(self) -> str:
        """Return ``PyQMatrix(dim=..., nnz=..., dtype=...)``."""
        ...
