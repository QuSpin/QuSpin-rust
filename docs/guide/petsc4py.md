# Distributed assembly with petsc4py

For a Hamiltonian whose Hilbert space is too large to materialise on a
single MPI rank, the standard pattern is "each rank computes only its
locally-owned rows and pushes them into a [`PETSc.Mat`][PETSc.Mat]".
QuSpin supports this via `op.csr_slab(...)` — a row-range CSR
materialisation that bypasses the global `QMatrix` entirely.

[PETSc.Mat]: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.html

## The API

```python
indptr, indices, data = op.csr_slab(
    basis,                       # SpinBasis or FermionBasis
    coeffs,                      # 1-D complex128, len == op.num_cindices
    row_start: int,              # inclusive, 0-based
    row_end: int,                # exclusive
    dtype=np.complex128,         # output value dtype
    drop_zeros=True,
)
```

Returns three numpy arrays in the layout petsc4py's
[`Mat.setValuesCSR`][setValuesCSR] / [`Mat.createAIJ(csr=…)`][createAIJ]
expect:

| Array     | Shape                           | Contents |
|-----------|---------------------------------|----------|
| `indptr`  | `(row_end - row_start + 1,)` int64 | Row pointer for the local slab; `indptr[0] == 0`, `indptr[-1] == nnz_local`. |
| `indices` | `(nnz_local,)` int64            | **Global** column indices (zero-based). PETSc handles the diag/off-diag split internally. |
| `data`    | `(nnz_local,)` `dtype`          | Accumulated `Σ_c coeffs[c] * stored_value` per entry — same semantics as `QMatrix.to_csr(coeffs)`. |

[setValuesCSR]: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.html#petsc4py.PETSc.Mat.setValuesCSR
[createAIJ]: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.html#petsc4py.PETSc.Mat.createAIJ

Each rank only ever holds its `(rend - rstart)` rows of CSR plus a
small per-row scratch buffer — memory is bounded by the slab,
regardless of the global dimension.

V1 covers `PauliOperator` paired with `SpinBasis` or `FermionBasis`.
Other operator families (`Bond`, `Boson`, `Fermion`, `Monomial`) follow
in mechanical follow-up PRs.

## Minimal petsc4py integration

```python
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from quspin_rs._rs import PauliOperator, SpinBasis

n_sites = 12
op = PauliOperator(
    [("XX", [[1.0, i, i + 1] for i in range(n_sites - 1)])],
    [("ZZ", [[1.0, i, i + 1] for i in range(n_sites - 1)])],
)
basis = SpinBasis.full(n_sites)

# Let PETSc decide the row partition.
mat = PETSc.Mat().create(comm=MPI.COMM_WORLD)
mat.setSizes(((PETSc.DECIDE, basis.size), (basis.size, basis.size)))
mat.setType(PETSc.Mat.Type.AIJ)
mat.setUp()
rstart, rend = mat.getOwnershipRange()

# Each rank materialises just its local rows.
coeffs = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
indptr, indices, data = op.csr_slab(
    basis, coeffs, rstart, rend, dtype=np.dtype("complex128"),
)
mat.setValuesCSR(
    indptr.astype(PETSc.IntType, copy=False),
    indices.astype(PETSc.IntType, copy=False),
    data,
)
mat.assemble()
```

`indptr` and `indices` are emitted as `int64`; the `astype(PETSc.IntType,
copy=False)` is a no-op on 64-bit PETSc builds and a cheap convert on
32-bit builds.

## Choosing the value dtype

PETSc is built with either real (`float64`) or complex (`complex128`)
scalars at compile time, exposed via `PETSc.ScalarType`. Pass that
dtype straight through to `csr_slab` when your Hamiltonian's matrix
elements fit it:

```python
# Real Hamiltonian (XX, ZZ, real Heisenberg, …) on a real PETSc build:
data = op.csr_slab(basis, coeffs, rstart, rend, dtype=np.dtype("float64"))[2]

# Hamiltonian with non-trivial imaginary entries (complex hopping,
# magnetic flux, …) — needs PETSc with --with-scalar-type=complex:
data = op.csr_slab(basis, coeffs, rstart, rend, dtype=np.dtype("complex128"))[2]
```

Internally the kernel always accumulates in `complex128` so the
operator-coefficient machinery stays uniform. Requesting `dtype=float64`
casts on the wrapper boundary; numpy emits a `ComplexWarning` because
the cast can in principle drop imaginary parts. Suppress it at the call
site when you know the matrix is real:

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
    indptr, indices, data = op.csr_slab(
        basis, coeffs, rstart, rend, dtype=np.dtype("float64"),
    )
```

If the matrix has non-trivial imaginary entries and PETSc was built
with real scalars, the cast silently loses them — assert the imaginary
part is small before casting if you're not sure.

## Time-dependent Hamiltonians

`csr_slab` lives on the operator, not on `Hamiltonian` — for a
time-dependent build, construct one PETSc `Mat` per cindex (the operator
terms whose coefficients evolve independently) by calling `csr_slab`
once per term with a coefficient mask that selects only that cindex,
then combine in PETSc:

```python
mats = []
for c in range(op.num_cindices):
    mask = np.zeros(op.num_cindices, dtype=np.complex128)
    mask[c] = 1.0
    indptr, indices, data = op.csr_slab(basis, mask, rstart, rend, dtype=...)
    m = PETSc.Mat().createAIJ(...).setValuesCSR(indptr.astype(...), indices.astype(...), data)
    m.assemble()
    mats.append(m)

# Per timestep:
H_t = mats[0].copy()
H_t.zeroEntries()
for c, m in enumerate(mats):
    H_t.axpy(coeff_c(t), m)
```

This keeps the slab construction one-shot per cindex, with only the
PETSc `axpy` per timestep.

## Runnable demo

A minimal proof of concept lives at
[`examples/petsc4py_chunked_build.py`][demo] in the source tree. It
builds the same Hamiltonian two ways — distributed via `csr_slab` and
reference via `QMatrix.to_csr` on rank 0 — and asserts the assembled
matrices match. Runs both serial and `mpirun -n 4`.

[demo]: https://github.com/QuSpin/QuSpin-rust/blob/main/examples/petsc4py_chunked_build.py

See `examples/README.md` for installing the optional `petsc4py`
dependency group and notes on matching `petsc4py` to your system PETSc
version.
