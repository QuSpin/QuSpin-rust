# petsc4py-compatible CSR slab construction

**Issue:** [#69](https://github.com/QuSpin/QuSpin-rust/issues/69)
**Status:** design approved 2026-05-03
**Scope of v1 (proof of concept):** `PauliOperator` + `SpinBasis` only.  Other
operator/basis families (`Bond`, `Boson`, `Fermion`, `Monomial`) follow in
mechanical follow-up PRs that reuse the same kernel pattern.

## Motivation

Today, materialising a QuSpin matrix for an external sparse-linear-algebra
library means calling `QMatrix.build_pauli(...).to_csr(coeffs)` and copying
the resulting `(indptr, indices, data)` arrays into the consumer.  For a
distributed solver like petsc4py + PETSc, this is wasteful: every MPI rank
ends up holding the full global matrix even though it only needs its
locally-owned rows, and the matrix is built serially on rank 0 (or
duplicated across all ranks) before being scattered.

We want each rank to allocate only its local rows and have QuSpin's
operator+basis machinery compute those rows directly, without ever
constructing a global `QMatrix`.

## Non-goals

- **No QMatrix integration.**  The slab path explicitly bypasses `QMatrix`.
  It must work with operators and bases as primitives.
- **No cindex visible to the caller.**  Multi-cindex operators are
  collapsed via a caller-supplied `coeffs` vector inside the kernel
  (same semantics as `QMatrix.to_csr(coeffs)`).  Per-cindex slabs were
  considered and rejected — external libraries don't have a "term"
  concept and forcing one through the API is friction.
- **No PETSc dependency in QuSpin.**  We produce numpy arrays in the
  exact layout petsc4py expects; the user calls `setValuesCSR` /
  `createAIJ(csr=…)` themselves.

## Target petsc4py integration

Per the [`MatMPIAIJSetPreallocationCSR`](https://petsc.org/release/manualpages/Mat/MatMPIAIJSetPreallocationCSR/)
docs and the [`petsc4py.PETSc.Mat`](https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.html)
reference, the canonical "give me your local rows" entry point is:

```python
mat.setValuesCSR(I, J, V)        # I, J indexed PetscInt; V scalar dtype
# or fused with creation:
mat = PETSc.Mat().createAIJ(size=…, csr=(I, J, V), comm=…)
```

with the conventions:

- `I` (row pointer) length `n_local + 1`, zero-based, references the rank's local rows.
- `J` (column indices) **global** column indices (PETSc handles the diag/off-diag split internally).
- `V` (values) dtype matches the matrix's PETSc scalar type.

The slab API outputs `(indptr, indices, data)` matching this layout exactly.

## API

### Python

```python
indptr, indices, data = op.csr_slab(
    basis,                       # SpinBasis or FermionBasis
    coeffs,                      # 1-D complex128, len == op.num_cindices
    row_start: int,              # inclusive, 0-based
    row_end: int,                # exclusive
    dtype=np.complex128,         # value dtype: any of QuSpin's ValueDType set
    drop_zeros=True,
) -> tuple[
    npt.NDArray[np.int64],       # indptr, shape (row_end - row_start + 1,)
    npt.NDArray[np.int64],       # indices, shape (nnz,), GLOBAL column indices
    npt.NDArray[Any],            # data, shape (nnz,), `dtype`
]
```

Index dtype is fixed at `int64`; the caller does
`indptr.astype(PETSc.IntType, copy=False)` (no-op for 64-bit PETSc builds).

### Rust

A monomorphic kernel generic over output value type, index type, and
operator cindex type, plus two type-erased dispatchers mirroring the
existing `QMatrixInner::build_hardcore` / `build_hardcore_bit` split
(`SpinBasis` → `GenericBasis`; `FermionBasis` → `BitBasis`):

```rust
// quspin-matrix/src/csr_slab.rs
pub fn csr_slab_pauli_generic<V, I, C>(
    op: &HardcoreOperator<C>,
    basis: &GenericBasis,
    coeffs: &[V],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<(Vec<I>, Vec<I>, Vec<V>), QuSpinError>
where V: Primitive, I: Index, C: CIndex;

pub fn csr_slab_pauli_bit<V, I, C>(
    op: &HardcoreOperator<C>,
    basis: &BitBasis,
    /* ... same arguments ... */
) -> Result<(Vec<I>, Vec<I>, Vec<V>), QuSpinError>;

// Type-erased dispatchers (one per basis flavour) for the PyO3 boundary.
// `I` is fixed at `i64` and `V` is selected by `dtype`, mirroring the
// existing QMatrixInner build entrypoints.
pub fn csr_slab_pauli_generic_inner(
    op: &HardcoreOperatorInner,
    basis: &GenericBasis,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    dtype: ValueDType,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError>;

pub fn csr_slab_pauli_bit_inner(/* ... mirrors above with BitBasis ... */);
```

`CsrSlab` is a thin enum wrapping `(Vec<i64>, Vec<i64>, Vec<V>)` for each
`V` in the `ValueDType` set, so the PyO3 layer can convert directly to a
`numpy::PyArray1` of the right dtype.

## Architecture: the row-range kernel

The kernel walks rows `[row_start, row_end)` only, applying the operator
term-by-term to each basis state and projecting back via `basis.index`:

```text
for r in [row_start, row_end):
    state = basis.state_at(r)
    row_entries = []                       # Vec<(col, value)>
    for term in op.terms():
        for bond in term.bonds:
            (sign, flipped) = apply_term(state, op_string, bond)
            col = basis.index(flipped)     # None → skip (state outside basis)
            row_entries.push((col, coeffs[term.cindex] * sign * stored_value))
    sort_and_merge_by_col(row_entries)     # combine same-col entries
    if drop_zeros: drop entries with |acc| ≤ scale * ZERO_TOL
    flush_into(indptr, indices, data)
```

Two properties:

1. **Memory bounded by the slab.**  Each rank holds `(rend - rstart)` rows
   of CSR plus a small per-row scratch buffer, regardless of the global dim.
2. **No coordination required.**  Each MPI rank calls the kernel
   independently; petsc4py orchestrates the row partition.

The per-row sort+merge+drop_zeros logic already exists inside
`QMatrix::to_csr_into` — extract it into a shared free function so both
code paths stay in sync.  This is a small refactor we get "for free".

Coefficients are applied **inside** the kernel — same semantic contract as
`QMatrix.to_csr(coeffs)`.  Output is a single combined CSR slab; cindex is
not visible to callers.

## Components

**1. Rust kernel** — new module `quspin-matrix/src/csr_slab.rs`.

Generic over `V: Primitive` and `I: Index`, shares the per-row
sort/merge/drop_zeros helper with `QMatrix::to_csr_into` (the helper is
extracted as part of this work).

A type-erased dispatcher `csr_slab_pauli_inner` accepts
`HardcoreOperatorInner` × `SpinBasisInner` (with the existing `(M, C)`
matrix dispatch from `QMatrixInner::build_hardcore`).  Returns a `CsrSlab`
enum that mirrors `ValueDType` so the Python wrapper can dispatch on the
output value type.

**2. PyO3 binding** — `csr_slab` method on `PyPauliOperator` in
`quspin-py/src/operator/pauli.rs`:

```rust
fn csr_slab<'py>(
    &self,
    py: Python<'py>,
    basis: &Bound<'py, PyAny>,
    coeffs: PyReadonlyArray1<'py, Complex64>,
    row_start: usize,
    row_end: usize,
    dtype: &Bound<'py, PyArrayDescr>,
    drop_zeros: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyAny>,                   // PyArray1 of dtype
)>;
```

Validation up front under the GIL, then `py.detach` around the kernel
call.  Returns numpy arrays directly with no extra Python-side conversion.

**3. Python demo** — `examples/petsc4py_chunked_build.py`:

```python
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from quspin_rs._rs import PauliOperator, SpinBasis

n_sites = 12
op = PauliOperator(
    [("XX", [[1.0, i, i + 1] for i in range(n_sites - 1)]),
     ("ZZ", [[1.0, i, i + 1] for i in range(n_sites - 1)])],
)
basis = SpinBasis.full(n_sites)

mat = PETSc.Mat().create(comm=MPI.COMM_WORLD)
mat.setSizes(((PETSc.DECIDE, basis.size), (basis.size, basis.size)))
mat.setType(PETSc.Mat.Type.AIJ)
mat.setUp()
rstart, rend = mat.getOwnershipRange()

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

Plus a short `examples/README.md` explaining `pip install petsc4py mpi4py`
and `mpirun -n 4 python examples/petsc4py_chunked_build.py`.

## Error handling

Validated up front in the PyO3 wrapper (cheap, GIL-held):

| Check | Error | Notes |
|---|---|---|
| `0 <= row_start <= row_end <= basis.size` | `ValueError` | `row_start == row_end` is a valid empty slab |
| `basis.is_built` | `ValueError` | otherwise `state_at` and `index` are unsafe |
| `len(coeffs) == op.num_cindices` | `ValueError` | matches `to_csr` semantics |
| basis type compatible with operator | `TypeError` | `PauliOperator` accepts `SpinBasis` or `FermionBasis` |
| `dtype` ∈ `ValueDType` set | `TypeError` | int8/int16/float32/float64/complex64/complex128 |

Validated in the Rust kernel:

- `row_end > basis.size` → `ValueError` (defends direct Rust callers).
- Output buffer sizing: kernel allocates `Vec`s; v1 has no caller-supplied
  buffer mode (a `csr_slab_into` zero-alloc variant could come later if
  profiling motivates it).

Edge cases that are *not* errors:

- **Empty slab** (`row_start == row_end`) — returns
  `(indptr=np.array([0], int64), indices=int64[0], data=dtype[0])`.
- **Row with no entries** — represented by `indptr[r+1] == indptr[r]`.
- **All entries dropped by `drop_zeros`** — same as above.
- **`basis.index(state)` returns `None`** — kernel skips (consistent
  with `QMatrix::build`'s symmetric/subspace handling).

Validation we deliberately *don't* do:

- Whether `row_end - row_start` matches PETSc's chosen ownership range
  (caller's responsibility).
- Whether MPI ranks together cover `[0, basis.size)` without overlap
  (caller's responsibility).

## Testing

Three tiers, increasing integration depth:

### 1. Rust unit tests (`quspin-matrix/src/csr_slab.rs` or `tests/csr_slab.rs`)

The load-bearing invariant is **slab equivalence**: concatenating slabs
from any partition of `[0, dim)` reproduces `QMatrix.to_csr(coeffs)`
byte-for-byte.

- `csr_slab_full_range_matches_to_csr` — `csr_slab(0, dim)` equals
  `QMatrix.build_pauli(...).to_csr(coeffs)`.
- `csr_slab_partition_concat_matches_to_csr` — partition `[0, dim)` into
  `k` chunks (`k = 1, 2, 3, dim`), concatenate via the standard
  CSR-of-row-blocks merge, compare to the full `to_csr`.
- `csr_slab_empty_range` — `csr_slab(r, r)` returns
  `(indptr=[0], indices=[], data=[])`.
- `csr_slab_drop_zeros_matches_to_csr` — same equivalence with
  `drop_zeros=true`.
- `csr_slab_subspace_basis` — equivalence on a Z-symmetric basis where
  some operator products fall outside the basis (exercises
  `basis.index(state) == None` skip).
- `csr_slab_invalid_range` / `csr_slab_wrong_coeffs_len` — error cases.

A 4–6 site Pauli `XX + ZZ` Hamiltonian is fast and exercises multi-cindex
non-trivial structure.

### 2. Python tests (`python/tests/test_rs.py::TestCsrSlab`)

Same equivalence story plus PyO3-boundary checks:

- `test_csr_slab_full_range_matches_to_csr` — round-trip via
  `QMatrix.build_pauli(...).to_csr(coeffs)`.
- `test_csr_slab_partition_round_trip` — assemble a `scipy.sparse` CSR
  from per-slab arrays, compare to scipy CSR built from `to_csr`.
- `test_csr_slab_empty_slab_returns_empty_arrays` — shapes and dtypes.
- `test_csr_slab_indices_dtype_int64`,
  `test_csr_slab_data_dtype_matches_dtype_arg`.
- `test_csr_slab_basis_type_mismatch_raises` — pass `BosonBasis` to
  `PauliOperator.csr_slab`.
- `test_csr_slab_unbuilt_basis_raises` — basis with `is_built == False`.
- `test_csr_slab_wrong_coeffs_size`,
  `test_csr_slab_invalid_row_range`.

### 3. petsc4py demo (`examples/petsc4py_chunked_build.py`)

Build the same Hamiltonian two ways and assert they match:

- (a) Distributed: each rank computes its slab and pushes into PETSc.
- (b) Reference: rank 0 only, `QMatrix.build_pauli(...).to_csr(coeffs)`
  → scipy → numpy dense.

Materialise both as dense and compare on rank 0 with `assert_allclose`.

Demo runs both serial (`python examples/petsc4py_chunked_build.py`) and
distributed (`mpirun -n 4 python …`).

**petsc4py is not added as a CI dev dependency.**  The demo is a
manual-run artifact; the Python tests in tier 2 cover kernel correctness
without needing PETSc.  CI's MPI/PETSc setup is not worth the friction
for a single demo file in v1.

## Future work (out of scope for v1)

- **Other operator/basis families.**  `BondOperator.csr_slab`,
  `BosonOperator.csr_slab`, `FermionOperator.csr_slab`,
  `MonomialOperator.csr_slab` — each follows the same kernel pattern
  with the matching `build_*` dispatch.
- **Time-dependent slabs on `Hamiltonian`.**  Once the per-operator
  pattern lands, a `Hamiltonian.csr_slab(time, rstart, rend)` is a
  thin wrapper that evaluates `coeff_fns(time)` and forwards to the
  underlying operator's slab.  (Or, equivalently, the petsc4py user
  can build per-cindex slabs once and combine in PETSc — see issue #69
  comments for the discussion.)
- **Zero-allocation variant** (`csr_slab_into`) accepting caller-supplied
  numpy buffers, mirroring `to_csr_into` in `quspin-matrix`.  Adds API
  surface; defer until profiling motivates it.
- **Configurable index dtype** (int32 vs int64) at the slab level,
  avoiding the Python-side `astype` for int32 PETSc builds.  Tradeoff:
  duplicates the kernel monomorphisations.  Defer.
