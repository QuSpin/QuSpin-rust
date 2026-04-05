"""Type stubs for the ``quspin_rs._rs`` extension module (QuSpin Rust core)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Basis types
# ---------------------------------------------------------------------------

class SpinBasis:
    """Spin (or hardcore-boson) basis — LHSS = 2 by default.

    Supports full (all states), subspace (reachable from seeds under a
    Hamiltonian), and symmetric (symmetry-projected) construction.
    """

    @classmethod
    def full(cls, n_sites: int, lhss: int = 2) -> SpinBasis:
        """Full Hilbert space (no restriction)."""
        ...

    @classmethod
    def subspace(
        cls,
        n_sites: int,
        ham: PauliOperator | BondOperator,
        seeds: list[str],
        lhss: int = 2,
    ) -> SpinBasis:
        """Krylov subspace spanned by acting ``ham`` on ``seeds``.

        Args:
            seeds: List of bit-strings, e.g. ``["0101", "1010"]``.
        """
        ...

    @classmethod
    def symmetric(
        cls,
        n_sites: int,
        ham: PauliOperator | BondOperator,
        seeds: list[str],
        symmetries: list[tuple[list[int], tuple[float, float]]],
        lhss: int = 2,
    ) -> SpinBasis:
        """Symmetry-projected subspace.

        Args:
            symmetries: List of ``(perm, (re, im))`` tuples where ``perm`` is
                a site-permutation and ``(re, im)`` is the character.
        """
        ...

    @property
    def n_sites(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def is_built(self) -> bool: ...
    def state_at(self, i: int) -> str:
        """Return the i-th basis state as a bit-string."""
        ...

    def index(self, state: str) -> int | None:
        """Return the index of a state given as a bit-string, or ``None`` if absent."""
        ...

    def __repr__(self) -> str: ...

class FermionBasis:
    """Fermionic basis — LHSS = 2."""

    @classmethod
    def full(cls, n_sites: int) -> FermionBasis: ...
    @classmethod
    def subspace(
        cls,
        n_sites: int,
        ham: FermionOperator | BondOperator,
        seeds: list[str],
    ) -> FermionBasis: ...
    @classmethod
    def symmetric(
        cls,
        n_sites: int,
        ham: FermionOperator | BondOperator,
        seeds: list[str],
        symmetries: list[tuple[list[int], tuple[float, float]]],
    ) -> FermionBasis: ...
    @property
    def n_sites(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def is_built(self) -> bool: ...
    def state_at(self, i: int) -> str: ...
    def index(self, state: str) -> int | None: ...
    def __repr__(self) -> str: ...

class BosonBasis:
    """Bosonic basis — LHSS user-supplied (>= 2)."""

    @classmethod
    def full(cls, n_sites: int, lhss: int) -> BosonBasis: ...
    @classmethod
    def subspace(
        cls,
        n_sites: int,
        lhss: int,
        ham: BosonOperator | BondOperator,
        seeds: list[str],
    ) -> BosonBasis: ...
    @classmethod
    def symmetric(
        cls,
        n_sites: int,
        lhss: int,
        ham: BosonOperator | BondOperator,
        seeds: list[str],
        symmetries: list[tuple[list[int], tuple[float, float]]],
    ) -> BosonBasis: ...
    @property
    def n_sites(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def is_built(self) -> bool: ...
    def state_at(self, i: int) -> str: ...
    def index(self, state: str) -> int | None: ...
    def __repr__(self) -> str: ...

class GenericBasis:
    """Generic basis for any on-site Hilbert-space size (LHSS ≥ 2).

    Supports lattice (site-permutation) and local (dit-permutation) symmetries.
    Paired with ``MonomialOperator`` for Hamiltonian construction.
    """

    @classmethod
    def full(cls, n_sites: int, lhss: int) -> GenericBasis:
        """Full Hilbert space (no restriction, no build step required)."""
        ...

    @classmethod
    def subspace(
        cls,
        n_sites: int,
        lhss: int,
        ham: MonomialOperator,
        seeds: list[str],
    ) -> GenericBasis:
        """Subspace built by BFS from seed states.

        Args:
            seeds: List of state strings (one digit per site, base ``lhss``).
        """
        ...

    @classmethod
    def symmetric(
        cls,
        n_sites: int,
        lhss: int,
        ham: MonomialOperator,
        seeds: list[str],
        symmetries: list[tuple[list[int], tuple[float, float]]],
        local_symmetries: list[
            tuple[list[int], tuple[float, float]]
            | tuple[list[int], tuple[float, float], list[int]]
        ] = ...,
    ) -> GenericBasis:
        """Symmetry-reduced subspace.

        Args:
            symmetries:       List of ``(perm, (re, im))`` lattice symmetry tuples.
            local_symmetries: List of 2- or 3-tuples for local (dit-permutation)
                symmetries.  Each entry is either:

                - ``(perm, (re, im))`` — applies to **all** sites, or
                - ``(perm, (re, im), mask)`` — applies only to sites in ``mask``.

                ``perm`` is a list of ``lhss`` integers permuting the local
                Hilbert-space states ``0 … lhss-1``.
        """
        ...

    @property
    def n_sites(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def is_built(self) -> bool: ...
    def state_at(self, i: int) -> str:
        """Return the i-th basis state as a string of site occupations."""
        ...

    def index(self, state: str) -> int | None:
        """Return the index of a state string, or ``None`` if absent."""
        ...

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Operator types
# ---------------------------------------------------------------------------

class PauliOperator:
    """Pauli / hardcore-boson operator.

    Each positional argument is a *term* (one coupling-constant index).
    A term is a list of ``(op_str, bonds)`` pairs that share that cindex.
    Each bond is ``[coeff, site0, site1, ...]``.

    Example::

        bonds = [[1.0, 0, 1], [1.0, 1, 2], [1.0, 2, 3]]
        # Two cindices (XX and ZZ can have independent coefficients):
        op = PauliOperator([("XX", bonds)], [("ZZ", bonds)])
        # One cindex (XX and ZZ always share the same coefficient):
        op = PauliOperator([("XX", bonds), ("ZZ", bonds)])
    """

    def __init__(
        self,
        *terms: list[tuple[str, list[list[Any]]]],
    ) -> None: ...
    @property
    def max_site(self) -> int: ...
    @property
    def num_cindices(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    def __repr__(self) -> str: ...

class BondOperator:
    """Dense two-site bond operator.

    Args:
        terms: List of ``(matrix, bonds, cindex)`` tuples where:
            - ``matrix``: 2-D complex128 array of shape ``(lhss^2, lhss^2)``
            - ``bonds``: list of ``(site_i, site_j)`` pairs
            - ``cindex``: coupling constant index
    """

    def __init__(
        self,
        terms: list[tuple[npt.NDArray[Any], list[tuple[int, int]], int]],
    ) -> None: ...
    @property
    def max_site(self) -> int: ...
    @property
    def num_cindices(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    def __repr__(self) -> str: ...

class BosonOperator:
    """Bosonic operator.

    Same variadic ``*terms`` format as ``PauliOperator``, using boson op
    strings (``+``, ``-``, ``n``).  ``lhss`` is keyword-only.
    """

    def __init__(
        self,
        *terms: list[tuple[str, list[list[Any]]]],
        lhss: int,
    ) -> None: ...
    @property
    def max_site(self) -> int: ...
    @property
    def num_cindices(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    def __repr__(self) -> str: ...

class FermionOperator:
    """Fermionic operator.

    Same variadic ``*terms`` format as ``PauliOperator``, using fermion op
    strings (``+``, ``-``, ``n``).  Jordan-Wigner signs are applied
    automatically.
    """

    def __init__(
        self,
        *terms: list[tuple[str, list[list[Any]]]],
    ) -> None: ...
    @property
    def max_site(self) -> int: ...
    @property
    def num_cindices(self) -> int: ...
    @property
    def lhss(self) -> int: ...
    def __repr__(self) -> str: ...

class MonomialOperator:
    """Generic monomial-matrix operator (one non-zero per row).

    Each positional term argument is a 3-tuple ``(perm, amp, bonds)`` where:

    - ``perm``: 1-D integer array of length ``lhss**k`` — output joint-state
      index for each input joint-state index.
    - ``amp``:  1-D complex128 array of length ``lhss**k`` — complex amplitude
      for each input joint-state.
    - ``bonds``: list of k-tuples of site indices; all bonds in one term must
      have the same number of sites ``k``.

    Cindex (coupling-constant index) is implicit by position: the i-th term
    gets cindex ``i``.

    Example (cyclic shift on nearest-neighbour bonds, lhss=3, 4 sites)::

        import numpy as np
        lhss = 3
        k = 2          # 2-site bond
        dim = lhss**k  # 9
        perm = np.array([
            lhss * ((a + 1) % lhss) + (b + 1) % lhss
            for a in range(lhss) for b in range(lhss)
        ], dtype=np.intp)
        amp = np.ones(dim, dtype=complex)
        bonds = [(0, 1), (1, 2), (2, 3)]
        op = MonomialOperator(lhss, (perm, amp, bonds))
    """

    def __init__(
        self,
        lhss: int,
        *terms: tuple[
            npt.NDArray[np.intp],
            npt.NDArray[np.complexfloating[Any, Any]],
            list[tuple[int, ...]],
        ],
    ) -> None: ...
    @property
    def max_site(self) -> int: ...
    @property
    def num_coeffs(self) -> int:
        """Number of distinct coupling-constant indices (= number of terms)."""
        ...

    @property
    def lhss(self) -> int: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# QMatrix
# ---------------------------------------------------------------------------

class QMatrix:
    """Sparse quantum matrix built from an operator + basis pair.

    Use the ``build_*`` static methods to construct.
    """

    @staticmethod
    def build_pauli(
        op: PauliOperator,
        basis: SpinBasis | FermionBasis,
        dtype: np.dtype[Any],
    ) -> QMatrix:
        """Build from a PauliOperator and a SpinBasis or FermionBasis."""
        ...

    @staticmethod
    def build_bond(
        op: BondOperator,
        basis: SpinBasis | FermionBasis | BosonBasis,
        dtype: np.dtype[Any],
    ) -> QMatrix:
        """Build from a BondOperator and any basis type."""
        ...

    @staticmethod
    def build_boson(
        op: BosonOperator,
        basis: BosonBasis,
        dtype: np.dtype[Any],
    ) -> QMatrix:
        """Build from a BosonOperator and a BosonBasis."""
        ...

    @staticmethod
    def build_fermion(
        op: FermionOperator,
        basis: FermionBasis,
        dtype: np.dtype[Any],
    ) -> QMatrix:
        """Build from a FermionOperator and a FermionBasis."""
        ...

    @staticmethod
    def build_monomial(
        op: MonomialOperator,
        basis: GenericBasis,
        dtype: np.dtype[Any],
    ) -> QMatrix:
        """Build from a MonomialOperator and a GenericBasis."""
        ...

    @property
    def dim(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    @property
    def num_coeff(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __add__(self, rhs: QMatrix) -> QMatrix: ...
    def __sub__(self, rhs: QMatrix) -> QMatrix: ...
    def to_csr(
        self,
        coeff: npt.NDArray[Any],
        drop_zeros: bool = True,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[Any],
    ]:
        """Return ``(indptr, indices, data)`` CSR arrays.

        Args:
            coeff: 1-D complex128 array of length ``num_coeff``.
        """
        ...

    def dot_many(
        self,
        coeff: npt.NDArray[Any],
        input: npt.NDArray[Any],
        output: npt.NDArray[Any],
        overwrite: bool,
    ) -> None:
        """Batch matvec: ``output += coeff * self @ input``.

        Args:
            coeff:  1-D complex128 of length ``num_coeff``.
            input:  2-D complex128 of shape ``(dim, n_vecs)`` (C-contiguous).
            output: 2-D complex128 of shape ``(dim, n_vecs)`` (modified in place).
        """
        ...

    def dot_transpose_many(
        self,
        coeff: npt.NDArray[Any],
        input: npt.NDArray[Any],
        output: npt.NDArray[Any],
        overwrite: bool,
    ) -> None: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Static marker
# ---------------------------------------------------------------------------

class Static:
    """Marker indicating a static (time-independent) coefficient.

    Pass ``Static()`` in the ``coeff_fns`` list to mark a cindex as having
    a constant coefficient of 1.0.
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------

class Hamiltonian:
    """Time-dependent Hamiltonian.

    Wraps a ``QMatrix`` together with coefficient descriptors — one per
    cindex.  Each entry is either ``Static()`` (coefficient 1.0) or a
    callable ``f(t) -> complex``.

    Args:
        qmatrix:   A ``QMatrix`` with ``num_coeff`` operator strings.
        coeff_fns: List of length ``qmatrix.num_coeff``.  Each element is
                   ``Static()`` for a time-independent term or a callable
                   ``f(t: float) -> complex`` for a time-dependent term.
    """

    def __init__(
        self,
        qmatrix: QMatrix,
        coeff_fns: list[Static | Callable[[float], complex]],
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_coeff(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def to_csr(
        self,
        time: float,
        drop_zeros: bool = True,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[Any],
    ]: ...
    def to_dense(
        self,
        time: float,
    ) -> npt.NDArray[np.complexfloating[Any, Any]]:
        """Return the Hamiltonian at ``time`` as a dense ``(dim, dim)`` complex128 matrix."""
        ...
    def dot(
        self,
        time: float,
        input: npt.NDArray[Any],
        output: npt.NDArray[Any],
        overwrite: bool,
    ) -> None: ...
    def dot_many(
        self,
        time: float,
        input: npt.NDArray[Any],
        output: npt.NDArray[Any],
        overwrite: bool,
    ) -> None: ...
    def dot_transpose_many(
        self,
        time: float,
        input: npt.NDArray[Any],
        output: npt.NDArray[Any],
        overwrite: bool,
    ) -> None: ...
    def expm_dot(
        self,
        time: float,
        a: complex,
        f: npt.NDArray[Any],
    ) -> None:
        """Compute ``exp(a · H(time)) · f`` in-place.

        Args:
            time: Evaluation time for the coefficient functions.
            a:    Scalar multiplier on the Hamiltonian exponent.
            f:    1-D complex128 array of shape ``(dim,)`` (modified in place).
        """
        ...

    def expm_dot_many(
        self,
        time: float,
        a: complex,
        f: npt.NDArray[Any],
    ) -> None:
        """Compute ``exp(a · H(time)) · F`` in-place for multiple column vectors.

        Args:
            time: Evaluation time for the coefficient functions.
            a:    Scalar multiplier on the Hamiltonian exponent.
            f:    2-D complex128 array of shape ``(dim, n_vecs)`` (modified in place).
        """
        ...

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# SchrodingerEq
# ---------------------------------------------------------------------------

class SchrodingerEq:
    """Schrödinger equation integrator (Dopri5).

    Args:
        hamiltonian: A ``Hamiltonian`` to integrate.
    """

    def __init__(self, hamiltonian: Hamiltonian) -> None: ...
    @property
    def dim(self) -> int: ...
    def integrate(
        self,
        t0: float,
        t_end: float,
        y0: npt.NDArray[Any],
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> npt.NDArray[Any]:
        """Integrate to ``t_end`` and return the final state.

        Args:
            y0: Initial state, shape ``(dim,)``.

        Returns:
            Final state, shape ``(dim,)``.
        """
        ...

    def integrate_dense(
        self,
        t0: float,
        t_end: float,
        y0: npt.NDArray[Any],
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]:
        """Integrate and return all accepted time-step outputs.

        Returns:
            ``(times, states)`` where ``times`` has shape ``(n_steps,)`` and
            ``states`` has shape ``(n_steps, dim)``.
        """
        ...

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Krylov subspace methods
# ---------------------------------------------------------------------------

class EigSolver:
    """Lanczos eigenvalue solver with full re-orthogonalization.

    Args:
        hamiltonian: A ``Hamiltonian`` whose eigenvalues to compute.
    """

    def __init__(self, hamiltonian: Hamiltonian) -> None: ...
    @property
    def dim(self) -> int: ...
    def solve(
        self,
        v0: npt.NDArray[Any],
        k_krylov: int,
        k_wanted: int = 1,
        which: str = "SA",
        tol: float = 1e-10,
        time: float = 0.0,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any], npt.NDArray[np.float64]]:
        """Compute eigenvalues and eigenvectors.

        Args:
            v0:       Initial vector, shape ``(dim,)``.
            k_krylov: Krylov subspace dimension.
            k_wanted: Maximum number of eigenpairs to return.
            which:    ``"SA"`` (smallest algebraic), ``"LA"`` (largest),
                      or ``"SM"`` (smallest magnitude).
            tol:      Convergence tolerance on residual norms. Only eigenpairs
                      with residual ``≤ tol`` are returned. Pass ``float('inf')``
                      to disable filtering.
            time:     Evaluation time for time-dependent coefficients.

        Returns:
            ``(eigenvalues, eigenvectors, residuals)`` where ``n_eig`` is the
            number of converged eigenpairs (may be less than ``k_wanted``).
            Shapes: ``(n_eig,)``, ``(n_eig, dim)``, ``(n_eig,)``.
        """
        ...

    def __repr__(self) -> str: ...

class FTLM:
    """Finite Temperature Lanczos Method.

    Args:
        hamiltonian: A ``Hamiltonian`` for the system.
    """

    def __init__(self, hamiltonian: Hamiltonian) -> None: ...
    @property
    def dim(self) -> int: ...
    def sample(
        self,
        v0: npt.NDArray[Any],
        k: int,
        observable: Hamiltonian,
        beta: float,
        time: float = 0.0,
        stored: bool = True,
    ) -> tuple[float, complex]:
        """Compute a single FTLM sample.

        Args:
            v0:         Random starting vector, shape ``(dim,)``.
            k:          Number of Lanczos steps.
            observable: ``Hamiltonian`` representing the observable.
            beta:       Inverse temperature.
            time:       Evaluation time for time-dependent coefficients.
            stored:     If ``True``, store all Lanczos vectors (O(k × dim) memory,
                        one Lanczos build). If ``False``, replay the recurrence
                        (O(k + dim) memory, extra matvecs).

        Returns:
            ``(z_r, oz_r)`` — partition function contribution and
            ``⟨O⟩ · Z`` contribution.
        """
        ...

    def __repr__(self) -> str: ...

class LTLM:
    """Low Temperature Lanczos Method.

    Args:
        hamiltonian: A ``Hamiltonian`` for the system.
    """

    def __init__(self, hamiltonian: Hamiltonian) -> None: ...
    @property
    def dim(self) -> int: ...
    def sample(
        self,
        v0: npt.NDArray[Any],
        k: int,
        observable: Hamiltonian,
        beta: float,
        time: float = 0.0,
        stored: bool = True,
    ) -> tuple[float, complex]:
        """Compute a single LTLM sample.

        Args:
            v0:         Random starting vector, shape ``(dim,)``.
            k:          Number of Lanczos steps.
            observable: ``Hamiltonian`` representing the observable.
            beta:       Inverse temperature.
            time:       Evaluation time for time-dependent coefficients.
            stored:     If ``True``, store all Lanczos vectors (O(k × dim) memory,
                        one Lanczos build). If ``False``, replay the recurrence
                        (O(k + dim) memory, extra matvecs).

        Returns:
            ``(z_r, oz_r)`` — partition function contribution and
            ``⟨φ|O|φ⟩`` where ``|φ⟩ = e^{-βH/2}|r⟩``.
        """
        ...

    def __repr__(self) -> str: ...

class FTLMDynamic:
    """FTLM dynamic correlations (spectral function).

    Args:
        hamiltonian: A ``Hamiltonian`` for the system.
    """

    def __init__(self, hamiltonian: Hamiltonian) -> None: ...
    @property
    def dim(self) -> int: ...
    def sample(
        self,
        v0: npt.NDArray[Any],
        k: int,
        operator: Hamiltonian,
        beta: float,
        omegas: npt.NDArray[np.float64],
        eta: float,
        time: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Compute one FTLM dynamic sample for the spectral function.

        Args:
            v0:       Random starting vector, shape ``(dim,)``.
            k:        Number of Lanczos steps for each run.
            operator: ``Hamiltonian`` representing the operator ``A``.
            beta:     Inverse temperature.
            omegas:   Frequency grid, shape ``(n_omega,)``.
            eta:      Lorentzian broadening parameter.
            time:     Evaluation time for time-dependent coefficients.

        Returns:
            Spectral function contribution, shape ``(n_omega,)``.
        """
        ...

    def __repr__(self) -> str: ...
