# Rust Rewrite Plan

## Motivation

- Rust's build system has better first-class Python extension support than C++
- More modern, maintainable build toolchain long-term
- No existing downstream consumers — clean slate, no backwards compatibility constraints

## Strategy

- **Complete rewrite** — no incremental C++/Rust FFI bridging
- **Idiomatic Rust** throughout: enums instead of `std::variant`, traits instead of concepts/CRTP, `rayon` instead of OpenMP, macros + PyO3 for the dispatch layer
- **Cargo workspace** with three crates:
  - `bitbasis` — Benes network / bit manipulation primitives (standalone, crates.io candidate)
  - `quspin-core` — core library (no Python dependency)
  - `quspin-py` — PyO3 Python bindings (depends on `quspin-core`)
- **Bottom-up order**: core primitives → bitbasis → scalar/dtype → array → basis → qmatrix → dispatch + PyO3 bindings

## Key Decisions

| Topic | Decision |
|---|---|
| Translation style | Idiomatic Rust, not a literal C++ port |
| Python bindings | PyO3 — the only consumer for now |
| Rewrite order | Complete rewrite, no FFI shim |
| Existing Python consumers | Separate package, will be refactored later — out of scope for now |
| `bitbasis` layer | Rewrite in Rust; use existing C library to **generate test cases** that validate the new Rust implementation |
| Dispatch layer | `macro_rules!` at the PyO3 boundary in `quspin-py`; `quspin-core` is pure generic Rust with no runtime dispatch |
| Multi-word integers | `ruint::Uint<BITS, LIMBS>` for `bits128_t` → `bits16384_t`; `u32`/`u64` for small cases |
| Parallelism | `rayon` throughout; replaces OpenMP and the custom `WorkQueue` thread pool |
| Standalone C++ interface | Deferred — will be implemented separately in idiomatic Rust (e.g. `cbindgen`) if needed |
| `Array` / `Scalar` / `DType` variant types | **Not ported to `quspin-core`**. The C++ variant machinery existed to support runtime dispatch in a statically typed language. PyO3 inspects NumPy dtypes at the boundary and dispatches once; after that, `quspin-core` operates on concrete generic types (`ArrayViewD<T>`, `T`, etc.). Variant types live only in `quspin-py` if needed at all. |

---

## Submodule Walkthrough

Legend: ⬜ not started · 📝 described · 🦀 in progress · ✅ complete

| Submodule | C++ Location | Target Crate | Status |
|---|---|---|---|
| Bit manipulation primitives | `basis/detail/bitbasis/` | `bitbasis` | 📝 |
| Data types | `dtype/` | `quspin-core` | 📝 |
| Shared utilities | `detail/` | `quspin-core` | 📝 |
| Scalar types | `scalar/` | `quspin-core` | 📝 |
| Array types | `array/` | `quspin-core` | 📝 |
| Quantum basis | `basis/` | `quspin-core` | 📝 |
| Quantum sparse matrix | `qmatrix/` | `quspin-core` | 📝 |
| Operators | `operator.hpp` | `quspin-core` | ✅ |
| Dispatch layer + PyO3 | `src/` | `quspin-py` | ✅ |

---

## Submodule Descriptions

### `bitbasis` — Bit Manipulation Primitives
**C++ location:** `include/quspin/basis/detail/bitbasis/`
**Target crate:** `bitbasis`
**Status:** ⬜

#### Description
Low-level bit manipulation primitives for quantum state representation. Basis states are packed into integers (32-bit, 64-bit, or multi-word bitsets for large systems), and this layer provides efficient algorithms for permuting and manipulating those packed representations. The central algorithm is the **Benes network** (Knuth TAOCP Vol. 4), which enables arbitrary bit permutations in O(log n) butterfly stages.

Local degrees of freedom ("dits" — digits with a fixed local Hilbert space size `lhss`) are stored as contiguous bit chunks within a state integer. This layer handles extracting, setting, and permuting those chunks.

#### Files
| File | Purpose |
|---|---|
| `benes.hpp` | Benes network construction and application: `gen_benes()` builds a network config from a permutation; `benes_fwd()` / `benes_bwd()` apply it forward/inverse |
| `cast.hpp` | Safe casts between the supported bit integer types (`bits32_t`, `bits64_t`, multi-word `bitset<N>`) |
| `dit_manip.hpp` | Extract and set dit values within packed state integers; both compile-time (`dit_manip<lhss>`) and runtime (`dynamic_dit_manip`) variants |
| `dit_perm.hpp` | High-level permutation ops: `perm_dit_locations` (permute which sites the dits occupy), `perm_dit_values` (permute the values at given sites), `higher_spin_inv` (invert all dit values) |
| `info.hpp` | Type traits (`bit_info<I>`) recording bit width, log-bits, and byte count for each integer type |
| `utils.hpp` | `bitset<num_bits>` multi-word bitset; `bit_permute_step()` butterfly stage; `bfly()` / `ibfly()` butterfly network primitives |

#### Rust Design Notes
- `bitset<N>` → fixed-size `[u64; N]` array in a newtype; or use the `bitvec` crate
- `bit_info<I>` → associated constants on a sealed `BitInt` trait
- `dit_manip<lhss>` (compile-time lhss) → const generics; `dynamic_dit_manip` → runtime struct
- **`perm_dit_locations` — naive loop first, Benes network deferred:** Rather than porting the Benes network algorithm, implement `perm_dit_locations::apply()` as a plain loop over sites (extract dit at src, insert at dst). This is O(n_sites) vs. O(log n_bits) but is obviously correct and directly testable against the C library's output. The Benes port is deferred to a later optimization pass once correctness is established.
- `perm_dit_values`, `higher_spin_inv`, `dynamic_*` variants — already naive-loop in C++, translate directly
- No heap allocation needed in the hot path; everything should be `#[inline]`

#### Testing Strategy
Use existing C library to generate test vectors that validate the Rust implementation.

---

### `dtype/` — Data Type Abstraction
**C++ location:** `include/quspin/dtype/`, `src/dtype/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
Runtime-polymorphic dtype system over 12 primitive types: `int8` through `int64`, `float32`, `float64`, `complex64`, `complex128`, and unsigned variants. A `DType` is a `std::variant` holding a thin `dtype<T>` tag; `std::visit()` dispatches on the contained type. Provides dtype introspection (`name()`, `is_int()`, `is_float()`, `is_complex()`), common-type resolution for mixed-type operations (`result_dtype()`), and the `PrimativeTypes` concept constraining valid element types.

#### Rust Design Notes
- `DType` / `dtypes` variant machinery is **not ported to `quspin-core`**. Dtype dispatch happens once at the PyO3 boundary in `quspin-py`; core functions are generic over `T: Primitive`.
- `Primitive` sealed trait (replacing `PrimativeTypes` concept) is defined in `quspin-core` and implemented for all 12 supported types — this is the only dtype infrastructure needed in core.
- If `DType` enums are needed in `quspin-py` for inspecting NumPy dtypes or error messages, two enums: `DType` (all 12) and `MatrixDType` (6 qmatrix-valid types: I8, I16, F32, F64, C64, C128), with `From<MatrixDType> for DType` and `TryFrom<DType> for MatrixDType`.
- `result_dtype()` promotion follows Rust's `From` trait hierarchy; mixed signed/unsigned is a type error.
- `typed_object<T>` CRTP base → not needed.

---

### `detail/` — Shared Utilities
**C++ location:** `include/quspin/detail/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
Infrastructure shared across the library. Covers error handling (`ErrorOr<T>` result type and `Error` enum), variant dispatch helpers, threading, casting, math utilities, smart pointers, and type concepts. This is the "stdlib extension" layer that the rest of the library builds on.

#### Files
| File | Purpose |
|---|---|
| `broadcast.hpp` | Array shape/dtype compatibility checks for binary operations |
| `cast.hpp` | Safe cross-type casts; complex→real extracts real part; trait `can_safe_cast<From,To>` prevents lossy conversions |
| `default_containers.hpp` | Type aliases: `svector_t` (small vector), `default_map_t` (unordered_map) |
| `dispatch.hpp` | `dispatch()` / `dispatch_array()` / `dispatch_scalar()` / `dispatch_elementwise()` — variant visitor wrappers that check shape/dtype compatibility before calling the typed kernel |
| `error.hpp` | `Error` with `ErrorType` enum; `ErrorOr<T>` variant result; `ReturnVoidError`; `visit_or_error()` throws on error |
| `math.hpp` | `abs()` and `abs_squared()` overloads for all primitive and complex types |
| `omp.hpp` | OpenMP configuration shims |
| `operators.hpp` | Template binary operators (+, -, *, /) with `std::common_type_t<>` promotion |
| `optional.hpp` | `Optional<T>` wrapper built on variant |
| `pointer.hpp` | `reference_counted_ptr<T>` — manual reference counting smart pointer used by arrays |
| `select.hpp` | Type filtering utilities for selecting a subset of variant alternatives |
| `threading.hpp` | `WorkQueue<Tasks>` thread pool; `async_for_each()` for batched parallel iteration; `ScheduleType` (Interleaved / SequentialBlocks) |
| `type_concepts.hpp` | C++20 concepts: `Floating`, `Integral`, `RealTypes`, `ComplexTypes`, `PrimativeTypes`, `QMatrixTypes`, `QMatrixValueTypes` |
| `variant_container.hpp` | CRTP base class for types that hold a variant and need uniform `visit()` dispatch |

#### Rust Design Notes
- `ErrorOr<T>` / `ReturnVoidError` / `visit_or_error()` → `Result<T, QuSpinError>` with `?`; no special infrastructure needed. `QuSpinError` is a `thiserror`-derived enum with three variants (`RuntimeError`, `ValueError`, `IndexError`) chosen to map directly to Python exception types at the PyO3 boundary via `impl From<QuSpinError> for PyErr`.
- `reference_counted_ptr<T>` → `Arc<T>` or `Rc<T>` (likely `Arc` for `rayon` compatibility)
- `WorkQueue` / `async_for_each()` / `omp.hpp` → replaced entirely by `rayon`. `Interleaved` → `par_iter()` (work-stealing), `SequentialBlocks` → `par_chunks()`, `omp_get_max_threads()` → `rayon::current_num_threads()`. Thread pool size configurable at runtime via `rayon::ThreadPoolBuilder`. Custom thread pool, atomic flag synchronization, and manual batching are not ported.
- `dispatch()` variants → `macro_rules!` match over `DType` tuples. Output and input dtypes are distinct and dispatched as a pair; only lossless `(output, input)` combinations appear as match arms — invalid pairs fall through to `Err(QuSpinError::ValueError(...))`. The kernel carries a `where Output: From<Input>` bound as a compile-time backstop. Dynamic checks (array shapes/sizes) are `if` guards inside the kernel returning `Err(...)`. For 3+ simultaneous type parameters (e.g. `QMatrix`: value × index × cindex), macros are nested rather than flat. The static/dynamic check callable abstraction from C++ (`static_args_check`, `dynamic_args_check`) is not ported — checks are written inline.
- Type concepts → sealed traits; most concept checks become trait bounds
- `optional.hpp` / `select.hpp` / `variant_container.hpp` → idiomatic Rust `Option`, enum pattern matching, and traits make these unnecessary

---

### `scalar/` — Scalar Types
**C++ location:** `include/quspin/scalar/`, `src/scalar/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
Runtime-polymorphic scalar wrapping any of the 12 primitive types. `Scalar` is a `std::variant<scalar<T>...>`; arithmetic operators dispatch via `std::visit()` and promote to the common type. Also includes `Reference`, a mutable reference into an array element that supports the same arithmetic interface (used for array indexing/assignment without copying).

#### Rust Design Notes
- **Not ported to `quspin-core`**. Scalars in core are plain `T` values where `T: Primitive`. No `Scalar` enum needed.
- `Reference` → not needed; dissolves entirely. Mutable array element access goes through the concrete typed array after dispatch.
- If a `Scalar` type is needed in `quspin-py` for Python-facing APIs, it can be a simple enum there.

---

### `array/` — Array Types
**C++ location:** `include/quspin/array/`, `src/array/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
Strided N-dimensional arrays (up to 64 dimensions) with reference-counted memory and runtime type polymorphism. `array<T>` holds a `reference_counted_ptr<T>` plus shape/stride metadata. `Array` is a `std::variant` over all 12 element types. Arrays default to row-major (C-style) strides but can hold arbitrary stride patterns. Copy constructor is deleted; cloning is explicit via `copy()`. Iteration is via `array_iterator<T>` which handles strided traversal.

Specialized type-alias variants (`index_arrays`, `matrix_arrays`, `state_arrays`) narrow the variant to subsets appropriate for each context (e.g., basis state storage vs. operator matrix values).

Kernel operations (`kernels.hpp`) include thread-safe atomic addition for parallel accumulation, norm, and allclose.

#### Rust Design Notes
- **`Array` variant not ported to `quspin-core`**. The C++ variant machinery is replaced by `ndarray`, the standard N-dimensional array library in the Rust scientific computing ecosystem.
- `quspin-core` functions take `ndarray::ArrayViewD<T>` / `ArrayViewMutD<T>` where `T: Primitive`. Static dimensions used where known at compile time: `Array1<T>` (state vectors, coefficients), `Array2<T>` (batched matvec).
- `ndarray::ArrayD<T>` for owned storage; `ArrayViewD<T>` / `ArrayViewMutD<T>` for all function arguments (zero-copy borrows).
- **Zero-copy PyO3 integration:** the `numpy` crate is built on `ndarray` — NumPy arrays convert directly to `ArrayViewD<T>` at the boundary with no allocation. Batched operations (multiple state vectors) arrive as `Array2<T>` naturally.
- `reference_counted_ptr` → ndarray manages its own memory; no custom smart pointer needed.
- `atomic_iadd()` → `ndarray::parallel` + `rayon` reduce patterns.
- Specialized sub-variants (`index_arrays`, `state_arrays`) → typed function signatures (`ArrayView1<i32>`, `ArrayView1<u64>`) rather than newtypes.

---

### `basis/` — Quantum Basis
**C++ location:** `include/quspin/basis/`, `src/basis/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
Manages the enumeration and indexing of quantum Hilbert space states, with optional symmetry reduction. There are three space flavors: `space<bitset_t>` (the full Hilbert space, O(1) lookup via reverse iota), `subspace<bitset_t>` (a filtered set of states stored in a vector + hash map for O(1) average lookup), and `symmetric_subspace<grp, bitset_t, norm_t>` (symmetry-reduced with character / normalization data). The `HardcoreBasis` public API is a variant over all combinations of these with all supported bitset widths (32-bit through 16384-bit).

Basis construction dispatches automatically to the appropriate bitset width based on system size. Symmetry is applied via group elements (lattice permutations, bit-flip, and hardcore-boson combinations) stored in `basis/grp/`.

#### Subcomponents
| Subcomponent | C++ Location | Description |
|---|---|---|
| Group elements | `basis/grp/`, `basis/detail/symmetry/` | `lattice_grp_element` (site permutation), `bitflip_grp_element`, `hardcore_boson_grp` combining both; each applies a symmetry transformation to a state and returns (new_state, character) |
| Space | `basis/detail/space.hpp` | Full Hilbert space; states are contiguous integers [0, 2^N); O(1) index lookup via identity map |
| Types | `basis/detail/types.hpp` | Bitset width type aliases: `bits32_t`, `bits64_t`, `bits128_t`, ..., `bits16384_t` |
| Basis generation | `basis/detail/generate.hpp` | `construct_from()`: builds a subspace by applying an operator to seed states and collecting reachable states |
| Basis iterators | `basis/detail/iterators.hpp` | Iterators over basis states in order |
| Basis operations | `basis/detail/basis_operations.hpp` | `state_at(index)`, `index(state)`, `size()` — uniform interface across space types |
| Hardcore basis | `basis/hardcore.hpp` | Public API: `HardcoreBasis` variant; construction dispatches on system size to pick bitset width |

#### Rust Design Notes
- **Multi-word bitset types** (`bits128_t` → `bits16384_t`) → `ruint::Uint<BITS, LIMBS>`. Covers all required bitwise ops, `Ord`, `Hash`, and `num-traits` integration. Saves implementing a custom multi-word integer. `u32` and `u64` used directly for the small cases.
- **`BasisPrimativeTypes` concept** → sealed `BitInt` trait in `quspin-core` (and `bitbasis`), implemented for `u32`, `u64`, and `ruint::Uint<BITS, LIMBS>`. Required bounds: `BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr`, `Not`, `Ord`, `Hash`, `From<u64>`.
- **Three space types** → three structs implementing a `BasisSpace<B: BitInt>` trait with `size()`, `state_at(usize) -> B`, `index(B) -> Option<usize>`:
  - `FullSpace<B>` — just a `usize` (Ns); `state_at(i) = Ns - i - 1`; no storage; `B` restricted to `u32`/`u64`
  - `Subspace<B>` — `Vec<B>` + `HashMap<B, usize>`; built via DFS from seed states
  - `SymmetricSubspace<B>` — `Vec<(B, f64)>` + `HashMap<B, usize>`; built via parallel BFS
- **`HardcoreBasis` variant** → lives only at the PyO3 boundary in `quspin-py`; `quspin-core` functions are generic over `B: BitInt` and the space type
- **Group elements** → `SymmetryOp<B: BitInt>` trait with `fn apply(&self, state: B) -> (B, Complex<f64>)`; implemented by `LatticeOp<B>` (wraps `perm_dit_locations`), `BitflipOp<B>` (wraps `perm_dit_mask`), and combined types
- **`symmetric_subspace::build()`** (OpenMP tasks) → parallel BFS via `rayon::par_iter()` over the current frontier; new states collected, deduplicated, merged each round
- **Hash map** → `std::collections::HashMap` or `rustc-hash::FxHashMap` for performance on integer keys

---

### `qmatrix/` — Quantum Sparse Matrix
**C++ location:** `include/quspin/qmatrix/`, `src/qmatrix/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
Sparse matrix representation for quantum operators, stored in COO (coordinate) format with separate row/column index vectors and value vectors. Parameterized by three types: element type (`dtype_t`: float, double, cfloat, cdouble), row/column index type (`index_t`: int32 or int64 depending on basis size), and operator string index type (`cindex_t`: uint8 or uint16). Row bounds arrays enable efficient row-wise iteration. The `QMatrix` public API is a variant over all valid type combinations.

The key operation is `dot(overwrite, coeff, input, output)`: computes `output = Σ_k coeff[k] * H[k] * input` for a set of operator strings H[k] — a batched sparse matrix-vector multiply.

#### Rust Design Notes
- `qmatrix<dtype_t, index_t, cindex_t>` → generic struct `QMatrix<V, I, C>` with trait bounds in `quspin-core`. No public enum variant needed — dispatch over (value × index × cindex) happens once at the PyO3 boundary in `quspin-py`.
- **Storage is CSR** (not COO): `indptr: Vec<I>` (row pointers, len = dim+1) and `data: Vec<Entry<V, I, C>>` where `Entry` is a named struct `{ value: V, col: I, cindex: C }`. Each nonzero carries a `cindex` (operator string index) encoding `H = Σ_k coeff[k] * H[k]`; `dot()` computes `output = Σ_k coeff[k] * H[k] * input` in a single pass.
- `(T, I, J)` tuples → named `Entry<V, I, C>` struct with fields `value`, `col`, `cindex`.
- `dot()` → `rayon::par_iter()` over rows; no data races (each row writes to a distinct output element).
- `dot_transpose()` → parallel fold via rayon: each thread accumulates a partial output array, partial arrays are summed at the end. Avoids atomics entirely.
- `operator+` / `operator-` → keep; implemented as sorted row merge via rayon parallel construction (same two-pass pattern as matrix construction).

---

### `operator/` — Operators
**C++ location:** `include/quspin/operator.hpp`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
Pauli operator representations and application to quantum states. A `pauli` namespace defines six operator types (X, Y, Z, creation P, annihilation M, number N). `apply_op(state, op, loc)` applies a single Pauli to a state at a given site, returning `(new_state, amplitude)` — the amplitude encodes the ±1 and ±i factors from Pauli algebra. Operators are composed into strings: `pauli_operator_string<cindex_t>` (variable-length) and `fixed_pauli_operator_string<cindex_t, N>` (fixed-length optimization). Operator strings are applied right-to-left.

#### Rust Design Notes
- `OperatorType` → `enum PauliOp { X, Y, Z, P, M, N }` with `fn apply<B: BitInt>(self, state: B, loc: u32) -> (B, Complex<f64>)`. Keep `apply` branchless (bool arithmetic as in the C++ original); mark `#[inline]`.
- `pauli_operator_string` / `fixed_pauli_operator_string` → **collapsed into a single type**:
  ```rust
  struct OpEntry<C> {
      cindex: C,
      ops: SmallVec<[(PauliOp, u32); 4]>,  // inline for 1–4 body operators; heap fallback for rare longer strings
      coeff: Complex<f64>,
  }
  ```
  The C++ bucketing by length (1/2/3/4/other) existed for compile-time loop unrolling via `fixed_pauli_operator_string<N>`. In Rust, `SmallVec<[_; 4]>` keeps the common case heap-free and LLVM unrolls the short inner loop without needing separate types per length. >4-body strings are rare and the heap fallback is acceptable.
- `pauli_hamiltonian<cindex_t>` → `struct PauliHamiltonian<C>` holding a single `Vec<OpEntry<C>>` sorted by `cindex`. No 5-bucket split.
- `operator()` (apply hamiltonian to a state) → method `fn apply<B: BitInt>(&self, state: B) -> SmallVec<[(Complex<f64>, B, C); 8]>`. Iterates `terms` in order (right-to-left per op within each entry), skips zero-amplitude results. `SmallVec` avoids heap allocation for typical Hamiltonian densities.
- `PauliHamiltonian` variant (`pauli_hamiltonian<uint8_t>` | `pauli_hamiltonian<uint16_t>`) → lives only at the PyO3 boundary in `quspin-py`. Dispatch is **1-way on cindex** (u8 | u16), chosen based on `max(cindex values, site indices)` at construction time.
- `basis_operations.hpp` — **skipped**. The file is incomplete WIP (references free variables `space`, `symmetry`, `space_t` that are not in scope). The basis → QMatrix construction pipeline is implemented from scratch based on the `BasisSpace` trait and CSR `QMatrix` designs.

---

### Dispatch Layer + PyO3
**C++ location:** `src/`
**Target crate:** `quspin-py`
**Status:** ⬜

#### Description
The dispatch layer bridges the runtime-polymorphic public API (variant types, DType, Array) to concrete typed implementations. In C++, this is done via `std::visit()` over variants, with explicit template instantiation in `.cpp` files to control the instantiation space. The Python bindings layer wraps the public API types (`HardcoreBasis`, `QMatrix`, `Array`, `Scalar`) for PyO3, converting NumPy arrays to/from `Array`, and exposing construction and computation functions.

#### Rust Design Notes

##### `#[pyclass]` Types
Three Python-facing wrapper types in `quspin-py`, each a thin `#[pyclass]` struct holding an inner enum:
- `PyPauliHamiltonian` — wraps `PauliHamiltonianInner { Ham8(PauliHamiltonian<u8>), Ham16(PauliHamiltonian<u16>) }`
- `PyHardcoreBasis` — wraps `HardcoreBasisInner` (see below)
- `PyQMatrix` — wraps `QMatrixInner` over all valid (V, I, C) combinations (see below)

No `PyArray` or `PyScalar` — NumPy arrays pass through directly via the `numpy` crate. New Hamiltonian and Basis types will be added as additional `#[pyclass]` types in future; the dispatch macros are designed to accommodate this without restructuring.

##### `PyHardcoreBasis` — `HardcoreBasisInner` Enum
16 variants: `Full32` and `Full64` hardcoded (FullSpace only practical for small systems); all other variants generated from a `for_each_bitset!` registry macro via the `paste` crate for identifier concatenation:

```rust
macro_rules! for_each_bitset {
    ($macro:ident) => {
        $macro! {
            (u32,              32),
            (u64,              64),
            (Uint<128,   2>,   128),
            (Uint<256,   4>,   256),
            (Uint<1024,  16>,  1024),
            (Uint<4096,  64>,  4096),
            (Uint<16384, 256>, 16384),
        }
    }
}

// Generates Sub32, Sym32, Sub64, Sym64, Sub128, Sym128, ...
macro_rules! define_basis_enum {
    ($(($uint:ty, $bits:literal)),*) => {
        enum HardcoreBasisInner {
            Full32(FullSpace<u32>),
            Full64(FullSpace<u64>),
            $(paste::paste! {
                [<Sub $bits>](Subspace<$uint>),
                [<Sym $bits>](SymmetricSubspace<$uint>),
            })*
        }
    }
}
for_each_bitset!(define_basis_enum);
```

The same `for_each_bitset!` registry drives the `with_basis!` dispatch macro, so adding a new bitset size is a one-line change in `for_each_bitset!`.

##### `PyQMatrix` — `QMatrixInner` Enum and Dispatch
`QMatrix<V, I, C>` is parameterized by value type `V`, index type `I`, and cindex type `C`. At the PyO3 boundary:

- **`I` is always `i64`** — fixed for all basis sizes. `I` remains a generic parameter in `quspin-core` for future flexibility, but `quspin-py` only ever instantiates `I=i64`. Rationale: `I` indexes both row/column positions and `indptr` (cumulative non-zero counts); even a basis that fits in i32 can produce a matrix whose non-zero count overflows i32, and this is impossible to know statically. A future heuristic can estimate nnz at construction time and dispatch to i32 when safe — deferred.
- **V**: i8, i16, f32, f64, Complex\<f32\>, Complex\<f64\> (6 types, all valid for all basis sizes)
- **C**: u8, u16 (2 types)
- **Total instantiations**: 6 × 1 × 2 = **12 QMatrix variants**

##### Dispatch Macros — Valid Combinations Only
To keep compile times tractable, dispatch macros only generate code for valid combinations (Option B) — invalid combos fall through to `Err(QuSpinError::ValueError(...))`. Each type axis has its own macro:

```rust
macro_rules! with_value_dtype {
    ($expr:expr, |$V:ident| $body:expr) => {
        match $expr {
            MatrixDType::I8   => { type $V = i8;           $body }
            MatrixDType::I16  => { type $V = i16;          $body }
            MatrixDType::F32  => { type $V = f32;          $body }
            MatrixDType::F64  => { type $V = f64;          $body }
            MatrixDType::C64  => { type $V = Complex<f32>; $body }
            MatrixDType::C128 => { type $V = Complex<f64>; $body }
        }
    }
}
// with_cindex_dtype!: u8 | u16
// with_basis!: generated from for_each_bitset! registry
```

Macros compose via nesting at the call site:
```rust
with_value_dtype!(v_dtype, |V| {
    with_cindex_dtype!(c_dtype, |C| {
        with_basis!(basis.inner, |b| {
            build_qmatrix::<V, i64, C, _>(hamiltonian, b)
        })
    })
})
```

**Construction** (PauliHamiltonian + HardcoreBasis → QMatrix): 6 V × 2 C × 7 bitset sizes = **84 valid combinations**.

**`dot()`**: dispatches on the QMatrix variant (12 combinations); input/output/coeff array dtypes must match V — validated at runtime after dispatch, no additional monomorphizations.

##### Error Propagation and Array Exchange
- `QuSpinError` → `PyErr` via `impl From<QuSpinError> for PyErr` (RuntimeError, ValueError, IndexError)
- NumPy arrays exchanged zero-copy via the `numpy` crate: `PyReadonlyArrayDyn<T>` → `ArrayViewD<T>`
- `quspin-core` has zero PyO3 dependency — pure Rust, no runtime dispatch

---

## Open Questions

- **`bitbasis` Benes network optimization:** `perm_dit_locations` will ship with a naive O(n_sites) loop. Once correctness is validated against the C library test vectors, consider porting the Benes network for large-system performance. Deferred intentionally.
- **`QMatrix` index type heuristic:** `I` is fixed to `i64` at the PyO3 boundary for now. A future optimization can estimate the number of non-zero matrix elements at construction time and dispatch to `i32` when safe, recovering memory. Deferred until there is a concrete performance need.
