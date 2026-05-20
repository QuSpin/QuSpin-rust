# Examples

Manually-runnable demos that aren't part of CI.

## petsc4py_chunked_build.py

Demonstrates using `PauliOperator.csr_slab` to build a distributed PETSc
matrix without ever materialising the full QuSpin matrix on any single
rank.  Compares the distributed assembly against `QMatrix.build_pauli +
to_csr` on rank 0.

### Requirements

PETSc + an MPI implementation must be installed on the system before
the Python bindings can be built — `brew install petsc open-mpi` (macOS)
or `apt-get install petsc-dev openmpi-bin` (Debian/Ubuntu) typically
suffice.  On HPC clusters, load the PETSc / MPI modules first.

Then install the optional `petsc4py` dependency group:

```sh
uv sync --group petsc4py
```

The `petsc4py` PyPI release usually tracks the latest PETSc; if the
system PETSc is older (e.g. Homebrew sometimes lags by a release), pin
to a matching version to avoid header mismatches:

```sh
PETSC_DIR=$(brew --prefix petsc) \
  uv pip install "petsc4py~=$(brew list --versions petsc | awk '{print $2}' | cut -d. -f1,2).0"
```

### Running

```sh
PETSC_DIR=$(brew --prefix petsc)  # macOS Homebrew install path

# Serial check:
uv run python examples/petsc4py_chunked_build.py

# Distributed run (4 ranks):
mpirun -n 4 uv run python examples/petsc4py_chunked_build.py
```

Both should print
`OK: distributed and reference matrices match (dim=256, n_ranks=N).`

### Notes on PETSc scalar type

PETSc is built with either real (`float64`) or complex (`complex128`)
scalars at compile time.  The demo's Hamiltonian (XX + ZZ) is real, so
it works against either build; the `data` array from `csr_slab` is
cast to whatever `PETSc.ScalarType` reports.  For a Hamiltonian with
non-trivial imaginary entries (e.g. an external magnetic field with a
complex phase) you'll need a complex-scalar PETSc — rebuild with
`--with-scalar-type=complex` (the demo asserts this and errors out
clearly otherwise).

Both should print
`OK: distributed and reference matrices match (dim=256, n_ranks=N).`
