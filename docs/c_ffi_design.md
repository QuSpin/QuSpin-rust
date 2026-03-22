# C FFI design sketch

This document explores what a C-compatible FFI layer (`quspin-c`) would look
like alongside the existing `quspin-py` crate, and identifies what needs to
change in the rest of the codebase to support it.

---

## 1. Proposed workspace structure

```
crates/
  quspin-core/    ← pure Rust algorithms, no FFI concerns (as today)
  quspin-py/      ← PyO3 extension module (as today)
  quspin-c/       ← NEW: cdylib / staticlib exposing a C ABI
```

`quspin-c` depends on `quspin-core` and produces a shared library
(`libquspin.so`) plus a `quspin.h` header. It has no dependency on PyO3 or
numpy.

---

## 2. Memory owned by each object today

Understanding what each type allocates is the foundation of the ownership
discussion.

| Object | Heap allocation |
|--------|----------------|
| `HardcoreHamiltonian<C>` | `Vec<OpEntry<C>>` — the operator terms |
| `FullSpace<B>` | None — purely computed from `dim` |
| `Subspace<B>` | `Vec<B>` (states) + `HashMap<B, usize>` (index map) |
| `SymmetricSubspace<B>` | `Vec<(B, f64)>` + `HashMap<B, usize>` + `SymmetryGrp<B>` |
| `SymmetryGrp<B>` | `Vec<LatticeElement>` + `Vec<GrpElement<B>>` |
| `QMatrix<V, I, C>` | `Vec<I>` (indptr) + `Vec<Entry<V,I,C>>` (data) — the bulk |

Memory consumed **during** a call (not retained):

- **Building any basis**: BFS stack; released after `build()` returns
- **`dot`**: no allocation; reads caller-provided slices, writes in-place
- **Building `QMatrix`**: temporary per-row buffers (merged into `data`)

In the Python binding, NumPy owns `coeff`, `input`, and `output` for `dot`.
The Rust code borrows slices for the duration of the call. No data is copied
into or out of the matrix objects.

---

## 3. The handle / opaque pointer pattern

C has no ownership model, so the standard pattern for heap objects is an
**opaque pointer to a heap-allocated Rust value**:

```c
/* quspin.h */
typedef struct quspin_hamiltonian_s* quspin_hamiltonian_t;
typedef struct quspin_basis_s*       quspin_basis_t;
typedef struct quspin_symmetry_grp_s* quspin_symmetry_grp_t;
typedef struct quspin_qmatrix_s*     quspin_qmatrix_t;
```

In Rust these are thin wrappers around `Box<XxxInner>` cast to raw pointers:

```rust
// quspin-c/src/lib.rs
#[repr(transparent)]
pub struct QuSpinHamiltonian(HardcoreHamiltonianInner);

#[no_mangle]
pub unsafe extern "C" fn quspin_hamiltonian_free(h: *mut QuSpinHamiltonian) {
    if !h.is_null() { drop(Box::from_raw(h)); }
}
```

Every constructor returns a heap-allocated handle; every destructor consumes
it. The caller is solely responsible for lifetime management — there is no
garbage collection.

### Object lifetime rules

```
                  caller                          library
                    │                               │
   quspin_xxx_new() │ ────────────────────────────> │ Box::new(XxxInner)
                    │ <──────── *mut QuSpinXxx ───── │
                    │                               │
   quspin_xxx_do()  │ ───── handle (borrow) ──────> │ &XxxInner (no copy)
                    │                               │
   quspin_xxx_free()│ ───── handle (consume) ──────>│ drop(Box::from_raw(ptr))
                    │                               │
```

All `do` operations borrow the handle immutably; they never take ownership.
`free` is the only function that consumes the handle.

---

## 4. Error handling

Python raises exceptions; C returns error codes or writes to an output
parameter. The cleanest approach for a C FFI is a caller-provided error
struct:

```c
typedef enum {
    QUSPIN_OK               = 0,
    QUSPIN_ERR_VALUE        = 1,   /* invalid argument */
    QUSPIN_ERR_INDEX        = 2,   /* out-of-range index */
    QUSPIN_ERR_RUNTIME      = 3,   /* internal error */
    QUSPIN_ERR_TYPE         = 4,   /* dtype mismatch */
    QUSPIN_ERR_NULL         = 5,   /* null handle */
} quspin_status_t;

typedef struct {
    quspin_status_t code;
    char message[256];             /* null-terminated; empty if code==OK */
} quspin_error_t;
```

Every function that can fail takes a `quspin_error_t* err` as its last
parameter. Passing `NULL` suppresses error details. On success, `err->code` is
`QUSPIN_OK`. On failure, constructors return `NULL` and operation functions
return a non-zero status.

This maps directly from `QuSpinError` variants:

```rust
fn write_error(err: *mut QuSpinError, e: QuSpinError) {
    if err.is_null() { return; }
    // fill code + message from QuSpinError variant
}
```

---

## 5. The C API surface

### 5.1 Dtype enum

The equivalent of Python's `np.dtype(...)`:

```c
typedef enum {
    QUSPIN_DTYPE_INT8       = 0,
    QUSPIN_DTYPE_INT16      = 1,
    QUSPIN_DTYPE_FLOAT32    = 2,
    QUSPIN_DTYPE_FLOAT64    = 3,
    QUSPIN_DTYPE_COMPLEX64  = 4,   /* two f32 components */
    QUSPIN_DTYPE_COMPLEX128 = 5,   /* two f64 components */
} quspin_dtype_t;
```

### 5.2 Symmetry group

```c
/* Lattice element: perm[src] = dst, length = n_sites */
quspin_symmetry_grp_t quspin_symmetry_grp_new(
    size_t             n_lattice,
    const double*      lattice_chars_re,   /* length n_lattice */
    const double*      lattice_chars_im,
    const size_t*      lattice_perms,      /* n_lattice × n_sites, row-major */
    const size_t*      lattice_lhss,       /* length n_lattice */
    size_t             n_local,
    /* local elements encoded as tagged structs — see below */
    const void*        local_ops,          /* opaque; see LocalOpDesc layout */
    size_t             n_sites,
    quspin_error_t*    err
);

void quspin_symmetry_grp_free(quspin_symmetry_grp_t grp);
```

The `local_ops` encoding mirrors `GrpOpDesc` but as a C struct array. A
header-level C struct:

```c
typedef enum { QUSPIN_OP_BITFLIP=0, QUSPIN_OP_LOCAL_VALUE=1,
               QUSPIN_OP_SPIN_INV=2 } quspin_local_op_kind_t;

typedef struct {
    quspin_local_op_kind_t kind;
    double grp_char_re, grp_char_im;
    const size_t* locs;    /* site indices; NULL = all sites (bitflip only) */
    size_t n_locs;
    size_t lhss;           /* ignored for BITFLIP */
    const uint8_t* perm;   /* length lhss; ignored unless LOCAL_VALUE */
} quspin_local_op_t;
```

### 5.3 Hamiltonian

The Python interface uses a nested list structure. The C interface flattens it
into parallel arrays — one element per term:

```c
quspin_hamiltonian_t quspin_hamiltonian_new(
    size_t          n_terms,
    const uint16_t* cindices,        /* one per term */
    const double*   coeffs_re,       /* one per term */
    const double*   coeffs_im,       /* one per term */
    const char*     op_chars,        /* all op strings concatenated: "xxzz+..." */
    const size_t*   op_offsets,      /* op_offsets[i]..op_offsets[i+1] slice
                                        into op_chars for term i;
                                        length = n_terms + 1 */
    const uint32_t* sites,           /* all site arrays concatenated;
                                        same length as op_chars */
    quspin_error_t* err
);

void quspin_hamiltonian_free(quspin_hamiltonian_t ham);

size_t quspin_hamiltonian_n_sites(quspin_hamiltonian_t ham);
size_t quspin_hamiltonian_num_cindices(quspin_hamiltonian_t ham);
```

The flat layout avoids pointer-to-pointer indirection and is cache-friendly for
batch construction. A C++ caller can build a small helper that calls
`add_term()` in a loop and then flushes to this function.

Compare to Python where the structure is:
```
terms[cindex] = [(op_str, [(coeff, site, ...), ...]), ...]
```

The C equivalent expresses the same data with no nested allocation on the
caller's side.

### 5.4 Basis

Seeds in Python are strings of `'0'`/`'1'` or `list[int]`; both are ultimately
`n_sites`-byte occupation vectors. The C interface uses that representation
directly — a flat `uint8_t` array, one byte per site, row-major over seeds:

```c
/* Full Hilbert space */
quspin_basis_t quspin_basis_full(
    size_t n_sites,
    quspin_error_t* err
);

/* Subspace reachable from seeds under ham */
quspin_basis_t quspin_basis_subspace(
    const uint8_t*          seeds,     /* n_seeds × n_sites, row-major */
    size_t                  n_seeds,
    quspin_hamiltonian_t    ham,
    quspin_error_t*         err
);

/* Symmetry-reduced subspace */
quspin_basis_t quspin_basis_symmetric(
    const uint8_t*          seeds,     /* n_seeds × n_sites, row-major */
    size_t                  n_seeds,
    quspin_hamiltonian_t    ham,
    quspin_symmetry_grp_t   grp,
    quspin_error_t*         err
);

void   quspin_basis_free(quspin_basis_t basis);
size_t quspin_basis_n_sites(quspin_basis_t basis);
size_t quspin_basis_size(quspin_basis_t basis);
```

### 5.5 QMatrix

```c
quspin_qmatrix_t quspin_qmatrix_build(
    quspin_hamiltonian_t    ham,
    quspin_basis_t          basis,
    quspin_dtype_t          dtype,
    quspin_error_t*         err
);

void   quspin_qmatrix_free(quspin_qmatrix_t mat);
size_t quspin_qmatrix_dim(quspin_qmatrix_t mat);
size_t quspin_qmatrix_nnz(quspin_qmatrix_t mat);

/* Matrix-vector product.
 * coeff: pointer to num_cindices elements of dtype
 * input:  pointer to dim elements of dtype
 * output: pointer to dim elements of dtype, modified in-place
 * overwrite: non-zero → zero output before accumulating */
quspin_status_t quspin_qmatrix_dot(
    quspin_qmatrix_t    mat,
    const void*         coeff,
    const void*         input,
    void*               output,
    int                 overwrite,
    quspin_error_t*     err
);

quspin_status_t quspin_qmatrix_dot_transpose(
    quspin_qmatrix_t    mat,
    const void*         coeff,
    const void*         input,
    void*               output,
    int                 overwrite,
    quspin_error_t*     err
);

/* Arithmetic — caller owns both inputs and the result */
quspin_qmatrix_t quspin_qmatrix_add(
    quspin_qmatrix_t a,
    quspin_qmatrix_t b,
    quspin_error_t*  err
);
quspin_qmatrix_t quspin_qmatrix_sub(
    quspin_qmatrix_t a,
    quspin_qmatrix_t b,
    quspin_error_t*  err
);
```

The `void*` arrays for `dot` lose static type information. The implementation
recovers them via the same `with_qmatrix!` dispatch that the Python binding
uses — the `QMatrixInner` variant already encodes `V` and `C`, so the dtype
enum is only needed at **build** time, not at `dot` time.

---

## 6. Impact on macro and type placement

This is the central architectural question. Two macros in `quspin-py` are
currently Python-specific but embody logic that belongs to `quspin-core`:

### 6.1 `select_b_for_n_sites!`

**Problem:** The macro hardcodes `return Err(pyo3::exceptions::PyValueError::new_err(...))` in its overflow branch — a PyO3 expression that cannot appear in `quspin-c`.

**Solution:** Move the macro to `quspin-core` and parameterise the overflow
expression:

```rust
// quspin-core/src/basis/hardcore/dispatch.rs  (or a new select.rs)
#[macro_export]
macro_rules! select_b_for_n_sites {
    ($n_sites:expr, $B:ident, $on_overflow:expr, $body:block) => {
        if $n_sites <= 32   { type $B = u32; $body }
        else if $n_sites <= 64   { type $B = u64; $body }
        else if $n_sites <= 128  { type $B = ::ruint::Uint<128,  2>; $body }
        ...
        else if $n_sites <= 8192 { type $B = ::ruint::Uint<8192,128>; $body }
        else { $on_overflow }
    };
}
```

Call sites supply their own overflow expression:

```rust
// quspin-py
select_b_for_n_sites!(n, B,
    return Err(pyo3::exceptions::PyValueError::new_err(format!("n_sites={n} > 8192"))),
    { ... }
);

// quspin-c
select_b_for_n_sites!(n, B,
    return write_error(err, QuSpinError::ValueError("n_sites > 8192".into())),
    { ... }
);
```

### 6.2 `MatrixDType` / `with_value_dtype!`

**Problem:** `MatrixDType` lives in `quspin-py::dtype` and `with_value_dtype!`
references `$crate::dtype::MatrixDType`. Both are currently PyO3-free, but they
live in the wrong crate — `quspin-c` would need to duplicate them.

**Solution:** Move both to `quspin-core`:

```
quspin-core/src/
  dtype.rs        ← NEW: ValueDType enum (renamed from MatrixDType),
                         CIndexDType enum, with_value_dtype!, with_cindex_dtype!
```

`quspin-py::dtype` becomes a thin file that re-exports from quspin-core and
adds only the `from_descr(py, descr)` constructor that converts a NumPy
`PyArrayDescr` into `ValueDType` — the only part that is genuinely PyO3-specific.

`quspin-c::dtype` similarly maps a `quspin_dtype_t` C enum integer to
`ValueDType`.

### 6.3 `with_basis!`, `with_sym_basis!`, `with_sym_grp!`, `with_qmatrix!`

These are already in `quspin-core` and are directly usable by `quspin-c`
without any changes. This was the right call.

### Summary of macro migrations

| Macro / type | Today | After C FFI |
|---|---|---|
| `with_qmatrix!` | quspin-core ✓ | unchanged |
| `with_basis!`, `with_plain_basis!`, `with_sym_basis!`, `with_sym_grp!` | quspin-core ✓ | unchanged |
| `MatrixDType` / `ValueDType` | quspin-py | → quspin-core |
| `CIndexDType` | quspin-py | → quspin-core |
| `with_value_dtype!` | quspin-py | → quspin-core |
| `with_cindex_dtype!` | quspin-py | → quspin-core |
| `select_b_for_n_sites!` | quspin-py | → quspin-core (parameterised overflow) |
| NumPy `from_descr` (dtype parsing) | quspin-py | stays in quspin-py |
| C `quspin_dtype_t` mapping | — | new in quspin-c |

---

## 7. Hamiltonian construction interface

The Python `terms` list is ergonomic but deeply dynamic. A C caller cannot
easily build a `list[list[tuple[str, list[tuple]]]]`. The flat parallel-array
interface from §5.3 is more natural.

However, this changes nothing inside `quspin-core`. `HardcoreHamiltonian<C>`
is already constructed from a flat `Vec<OpEntry<C>>` — the Python binding
parses the nested list into that Vec, and the C binding would parse flat arrays
into the same Vec. The core type is already in the right shape.

The only real question is whether to expose a **one-shot constructor** (pass all
terms at once) or a **builder** (add terms incrementally). For most use cases
the one-shot API is sufficient and simpler to implement. A builder can always
be added later as a convenience wrapper.

---

## 8. Basis construction interface

### Seed representation

| Interface | Python | C |
|---|---|---|
| Small systems (≤64 bits) | `"1100"` or `[1,1,0,0]` | `uint8_t seeds[n_seeds][n_sites]` (one byte per site) |
| Large systems (>64 bits) | same strings | same byte array — uniform, no bit-packing |

The byte-per-site representation is the lowest common denominator. It is what
Python uses internally after parsing the strings (see `seed_to_bits`). A C++
caller can pack bits if they prefer, converting before the call.

### No change to quspin-core

`Subspace::build` and `SymmetricSubspace::build` take a `seed: B` and an `Op`
closure — they are already agnostic to how the seed arrived. The C binding
would convert the byte array to `B` using the same `seed_as::<B>()` helper
that quspin-py uses (or a copy of it in quspin-c).

---

## 9. Memory management for `dot`

In Python, NumPy owns `coeff`, `input`, `output`. Rust borrows slices for the
duration of the call; no data is copied.

In C, the same contract holds but is expressed differently:

```c
// Caller owns all three arrays.
// quspin borrows them for the duration of this call only.
// coeff: num_cindices elements of dtype
// input:  dim elements of dtype
// output: dim elements of dtype, written in-place
quspin_status_t quspin_qmatrix_dot(mat, coeff, input, output, overwrite, err);
```

The `void*` arrays are cast to `*const V` / `*mut V` inside the `with_qmatrix!`
dispatch arm, using `std::slice::from_raw_parts{_mut}`. This is `unsafe` — the
caller guarantees that:
1. The pointers are valid and non-null
2. The arrays have the required lengths (`num_cindices`, `dim`, `dim`)
3. The element type matches the matrix dtype
4. `output` is not aliased by `coeff` or `input`

Condition (3) cannot be checked at compile time (the dtype is a runtime value),
so a runtime check against `mat`'s stored dtype is needed. This is the same
validation that PyO3's `downcast::<PyArray1<V>>()` performs.

---

## 10. Thread safety

`QMatrix::dot` takes `&self` (immutable) and a `&mut [V]` for `output`. In Rust
terms, the matrix is `Send + Sync` (no interior mutability, all fields are plain
data). Multiple threads can call `dot` on the same matrix concurrently, provided
each thread writes to a distinct `output` buffer.

For the C API, this should be documented explicitly in the header:

```c
/* Thread safety:
 * - All handle types are immutable after construction and thread-safe to share.
 * - quspin_qmatrix_dot and quspin_qmatrix_dot_transpose may be called
 *   concurrently on the same matrix handle, provided each call uses a
 *   distinct output buffer.
 * - Constructors and free functions are NOT thread-safe on the same handle. */
```

This is more permissive than the Python binding, where the GIL serialises all
calls.

---

## 11. Open questions

1. **`ValueDType` → quspin-core migration**: should this happen before or after
   starting `quspin-c`? Moving it first keeps the C FFI crate thin but requires
   a quspin-py PR that changes the import paths.

2. **Builder vs one-shot Hamiltonian constructor**: the one-shot flat-array API
   is sufficient for batch construction but awkward for incremental use from C++.
   A thin C++ header providing a `HamiltonianBuilder` RAII type on top of the C
   API may be the right answer without complicating the core C ABI.

3. **Seed packing for large systems**: the byte-per-site representation is
   simple and uniform but wastes 7 bits per site. For very large systems
   (n_sites > 1000), a packed-bit representation would reduce seed memory by 8×.
   This could be offered as an alternative entry-point
   (`quspin_basis_subspace_packed`) without changing the primary API.

4. **`quspin-c` crate type**: `cdylib` (shared library) is the standard choice
   for C interoperability. A `staticlib` target can be added for users who want
   to link statically. Both can be enabled simultaneously in `Cargo.toml` with
   `crate-type = ["cdylib", "staticlib"]`.

5. **Header generation**: `cbindgen` can auto-generate `quspin.h` from the
   `extern "C"` function signatures in `quspin-c/src/lib.rs`. This avoids
   hand-maintaining a header that drifts from the implementation.
