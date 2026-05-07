# Examples

Manually-runnable demos that aren't part of CI.

## petsc4py_chunked_build.py

Demonstrates using `PauliOperator.csr_slab` to build a distributed PETSc
matrix without ever materialising the full QuSpin matrix on any single
rank.  Compares the distributed assembly against `QMatrix.build_pauli +
to_csr` on rank 0.

### Requirements

PETSc + an MPI implementation must be installed on the system before
the Python bindings can be built — `brew install petsc` (macOS) or
`apt-get install petsc-dev openmpi-bin` (Debian/Ubuntu) typically
suffice.  On HPC clusters, load the PETSc / MPI modules first.

Then install the optional `petsc4py` dependency group, which bundles
`petsc4py` and `mpi4py`:

```sh
uv sync --group petsc4py
```

(or `pip install petsc4py mpi4py` outside the uv-managed environment).

### Running

```sh
# Single-rank serial check:
uv run python examples/petsc4py_chunked_build.py

# Distributed run (4 ranks):
uv run mpirun -n 4 python examples/petsc4py_chunked_build.py
```

Both should print
`OK: distributed and reference matrices match (dim=256, n_ranks=N).`
