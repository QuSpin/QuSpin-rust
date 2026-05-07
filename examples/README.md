# Examples

Manually-runnable demos that aren't part of CI.

## petsc4py_chunked_build.py

Demonstrates using `PauliOperator.csr_slab` to build a distributed PETSc
matrix without ever materialising the full QuSpin matrix on any single
rank.  Compares the distributed assembly against `QMatrix.build_pauli +
to_csr` on rank 0.

### Requirements

```sh
pip install petsc4py mpi4py
```

petsc4py needs a working PETSc install — `brew install petsc` (macOS) or
`apt-get install petsc-dev` (Debian/Ubuntu) typically suffice. On HPC
clusters, load the PETSc module first.

### Running

```sh
# Single-rank serial check:
python examples/petsc4py_chunked_build.py

# Distributed run (4 ranks):
mpirun -n 4 python examples/petsc4py_chunked_build.py
```

Both should print
`OK: distributed and reference matrices match (dim=256, n_ranks=N).`
