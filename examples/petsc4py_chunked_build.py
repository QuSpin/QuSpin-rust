"""petsc4py-compatible chunked matrix construction (issue #69).

Run serial:        python examples/petsc4py_chunked_build.py
Run distributed:   mpirun -n 4 python examples/petsc4py_chunked_build.py

Builds the same Hamiltonian two ways and compares them on rank 0:
  (a) Distributed via PauliOperator.csr_slab — each MPI rank computes
      only its locally-owned rows and pushes them into a PETSc Mat.
  (b) Reference via QMatrix.build_pauli + to_csr — the full global matrix.

Memory-wise the point of the slab path: each rank only ever holds its
own (rend - rstart) rows of CSR plus a small per-row scratch buffer,
regardless of the global dimension.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse

try:
    from mpi4py import MPI  # pyright: ignore[reportMissingImports]
    from petsc4py import PETSc  # pyright: ignore[reportMissingImports]
except ImportError as exc:
    raise SystemExit(
        "This example requires petsc4py and mpi4py:\n" "    pip install petsc4py mpi4py"
    ) from exc

from quspin_rs._rs import PauliOperator, QMatrix, SpinBasis

N_SITES = 8


def build_op() -> PauliOperator:
    return PauliOperator(
        [("XX", [[1.0, i, i + 1] for i in range(N_SITES - 1)])],
        [("ZZ", [[1.0, i, i + 1] for i in range(N_SITES - 1)])],
    )


def build_distributed() -> tuple[PETSc.Mat, int]:
    op = build_op()
    basis = SpinBasis.full(N_SITES)

    mat = PETSc.Mat().create(comm=MPI.COMM_WORLD)
    mat.setSizes(((PETSc.DECIDE, basis.size), (basis.size, basis.size)))
    mat.setType(PETSc.Mat.Type.AIJ)
    mat.setUp()
    rstart, rend = mat.getOwnershipRange()

    # XX + ZZ has only real matrix elements, so request float64 directly
    # from `csr_slab` and skip any complex/real conversion.  numpy issues a
    # ComplexWarning at the wrapper boundary because the Rust kernel
    # internally accumulates in complex128 before the cast — silenced here
    # because the imaginary part is provably zero for this Hamiltonian.
    # (Use dtype=np.complex128 for a Hamiltonian with non-trivial imaginary
    # entries, e.g. complex hopping amplitudes — but then PETSc itself must
    # also be built with --with-scalar-type=complex.)
    coeffs = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        indptr, indices, data = op.csr_slab(
            basis, coeffs, rstart, rend, dtype=np.dtype("float64")
        )
    mat.setValuesCSR(
        indptr.astype(PETSc.IntType, copy=False),
        indices.astype(PETSc.IntType, copy=False),
        np.ascontiguousarray(data, dtype=PETSc.ScalarType),
    )
    mat.assemble()
    return mat, basis.size


def build_reference() -> np.ndarray:
    """Rank-0 reference matrix as a dense numpy array."""
    op = build_op()
    basis = SpinBasis.full(N_SITES)
    qm = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
    coeffs = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    indptr, indices, data = qm.to_csr(coeffs)
    return scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(basis.size, basis.size)
    ).toarray()


def gather_distributed_csr(mat: PETSc.Mat, dim: int) -> np.ndarray | None:
    """Gather the distributed AIJ matrix to a dense numpy array on rank 0."""
    rstart, rend = mat.getOwnershipRange()
    indptr_loc, indices_loc, data_loc = mat.getValuesCSR()
    n_local = rend - rstart

    # Always promote to complex128 for the comparison so the same code path
    # works whether PETSc was built with real or complex scalars.
    comm = MPI.COMM_WORLD
    parts = comm.gather(
        (
            np.asarray(indptr_loc, dtype=np.int64),
            np.asarray(indices_loc, dtype=np.int64),
            np.asarray(data_loc, dtype=np.complex128),
            n_local,
            rstart,
        ),
        root=0,
    )
    if comm.Get_rank() != 0:
        return None

    # Stitch into a single global CSR.
    global_indptr = np.zeros(dim + 1, dtype=np.int64)
    indices_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    cum_nnz = 0
    for ip_loc, ii_loc, dd_loc, n_loc, rs in parts:
        for i in range(n_loc):
            global_indptr[rs + i + 1] = cum_nnz + (ip_loc[i + 1] - ip_loc[0])
        indices_parts.append(ii_loc)
        data_parts.append(dd_loc)
        cum_nnz += int(ip_loc[-1] - ip_loc[0])
    indices = (
        np.concatenate(indices_parts) if indices_parts else np.zeros(0, dtype=np.int64)
    )
    data = (
        np.concatenate(data_parts) if data_parts else np.zeros(0, dtype=np.complex128)
    )
    return scipy.sparse.csr_matrix(
        (data, indices, global_indptr), shape=(dim, dim)
    ).toarray()


def main() -> None:
    rank = MPI.COMM_WORLD.Get_rank()

    dist_mat, dim = build_distributed()
    distributed_dense = gather_distributed_csr(dist_mat, dim)

    if rank == 0:
        ref = build_reference()
        np.testing.assert_allclose(distributed_dense, ref, atol=1e-12)
        print(
            f"OK: distributed and reference matrices match "
            f"(dim={dim}, n_ranks={MPI.COMM_WORLD.Get_size()})."
        )


if __name__ == "__main__":
    main()
