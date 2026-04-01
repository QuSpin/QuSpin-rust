# Hamiltonian Design

Time-dependent `Hamiltonian<M, I, C>` wrapping `QMatrix`, plus the prerequisite
`QMatrix` refactor and `operator` module rename.

---

## Pre-implementation: `QMatrix<M, I, C>` refactor

**Motivation**: `Hamiltonian`'s time functions return `Complex<f64>`, but stored
matrix elements are often real (`f64`). Decoupling entry storage type `M` from
computation type `V` avoids storing a complex QMatrix when the physics is real.

**Change**: Rename type parameter `V → M` in `QMatrix`, `Entry`, and `binary_op`.
All methods that produce output become generic over a separate `V: Primitive`:

```rust
// Before: QMatrix<V, I, C> — entry type = output type
// After:  QMatrix<M, I, C> — entry type M; methods generic over output V

pub fn dot<V: Primitive>(
    &self, overwrite: bool, coeff: &[V], input: &[V], output: &mut [V]
) -> Result<(), _>
where M: Primitive   // M already is Primitive, so this is just the renamed bound
```

The arithmetic in method bodies changes from `coeff[c] * entry.value` (both `V`) to:

```rust
let entry_as_v = V::from_complex(entry.value.to_complex()); // M → Complex<f64> → V
let contrib = coeff[c] * entry_as_v * input[col];           // V × V × V → V
```

**Additional method needed for `Hamiltonian::Add`**:

```rust
/// Remap each entry's cindex via `mapping[old] = new`.
/// Re-sorts per row; precondition: the remapping does not create
/// duplicate (col, cindex) keys within a row.
pub fn remap_cindices(self, mapping: &[C]) -> Self
```

**Downstream impact**:
- `with_qmatrix!` macro: inject `type $M = ...` instead of `type $V = ...`. Call
  sites must supply the output type `V` explicitly when calling `dot::<V>(...)` etc.
- `QMatrixInner` variants: unchanged (still 12, keyed by `M × C`)
- `binary_op` in `ops.rs`: type parameter rename `V → M`, no logic changes
- `build.rs`: `build_from_basis<H, B, M, I, C, S>` — same logic, type param rename

---

## Part 1: Module rename

The current `hamiltonian` module holds operator algebra types and a trait for
applying them to basis states. It is renamed to `operator` to free the name for
the new concrete `Hamiltonian` struct.

| Before | After |
|---|---|
| `mod hamiltonian` | `mod operator` |
| `trait Hamiltonian<C>` | `trait Operator<C>` |
| `SpinHamiltonian` / `SpinHamiltonianInner` | `SpinOperator` / `SpinOperatorInner` |
| `FermionHamiltonian` / `FermionHamiltonianInner` | `FermionOperator` / `FermionOperatorInner` |
| `BosonHamiltonian` / `BosonHamiltonianInner` | `BosonOperator` / `BosonOperatorInner` |
| `HardcoreHamiltonian` / `HardcoreHamiltonianInner` | `HardcoreOperator` / `HardcoreOperatorInner` |
| `BondHamiltonian` / `BondHamiltonianInner` | `BondOperator` / `BondOperatorInner` |
| `pub use hamiltonian::{Hamiltonian, ParseOp}` in `lib.rs` | `pub use operator::{Operator, ParseOp}` |

`ParseOp`, `BondTerm`, all `*Op`, and `*OpEntry` types are **unchanged**.

---

## Part 2: New `hamiltonian` module

### Struct

```rust
pub struct Hamiltonian<M: Primitive, I: Index, C: CIndex> {
    matrix: QMatrix<M, I, C>,
    // One Arc per cindex 1..num_coeff; cindex 0 is static (no fn needed)
    coeff_fns: Vec<Arc<dyn Fn(f64) -> Complex<f64> + Send + Sync>>,
}
```

`Arc` rather than `Box` is required for the `Add`/`Sub` merging algorithm.

`cindex = 0` is the **static** part of the Hamiltonian (coefficient always `1.0`).
`cindex k > 0` has time-dependent coefficient `coeff_fns[k-1](t)`.

### Constructor

```rust
pub fn new(
    matrix: QMatrix<M, I, C>,
    coeff_fns: Vec<Arc<dyn Fn(f64) -> Complex<f64> + Send + Sync>>,
) -> Result<Self, QuSpinError>
// Error if coeff_fns.len() != matrix.num_coeff().saturating_sub(1)
```

### Private helper

```rust
fn eval_coeffs(&self, time: f64) -> Vec<Complex<f64>> {
    let mut c = Vec::with_capacity(self.matrix.num_coeff());
    c.push(Complex::new(1.0, 0.0)); // cindex 0: always 1 (static part)
    for f in &self.coeff_fns {
        c.push(f(time));
    }
    c
}
```

The output type of all `Hamiltonian` operations is fixed to `Complex<f64>` —
time-dependent Hamiltonians always live in complex space regardless of `M`.

### Public API

```rust
pub fn dim(&self) -> usize
pub fn num_coeff(&self) -> usize

pub fn to_csr_nnz(&self, time: f64, drop_zeros: bool) -> Result<usize>
pub fn to_csr_into(&self, time: f64, drop_zeros: bool,
                   indptr: &mut [I], indices: &mut [I], data: &mut [Complex<f64>]) -> Result<()>
pub fn to_csr(&self, time: f64, drop_zeros: bool) -> Result<(Vec<I>, Vec<I>, Vec<Complex<f64>>)>
pub fn to_dense_into(&self, time: f64, output: &mut [Complex<f64>]) -> Result<()>
pub fn to_dense(&self, time: f64) -> Result<Vec<Complex<f64>>>
pub fn dot(&self, overwrite: bool, time: f64,
           input: &[Complex<f64>], output: &mut [Complex<f64>]) -> Result<()>
pub fn dot_transpose(&self, overwrite: bool, time: f64,
                     input: &[Complex<f64>], output: &mut [Complex<f64>]) -> Result<()>
```

Each method calls `let coeffs = self.eval_coeffs(time)` then delegates to
`self.matrix.{method}::<Complex<f64>>(&coeffs, ...)`.

### `Add` / `Sub` with cindex merging

Two entries from different Hamiltonians that share the **same `Arc`** (by pointer
equality) share the same time-dependence and fold into one cindex in the result.

```
// 1. Build deduplicated function table by Arc::ptr_eq
let mut merged_fns: Vec<Arc<...>> = vec![];
let lhs_cindex_map: Vec<C>  // mapping[k] = new cindex for lhs's cindex k
let rhs_cindex_map: Vec<C>  // same for rhs

// cindex 0 always maps to 0 in both (static part never remapped)
// For each fn in lhs.coeff_fns:
//   if it ptr_eq matches an existing merged_fns entry → reuse that index
//   else → push to merged_fns, assign next index
// Same for rhs.coeff_fns

// 2. Remap both QMatrices to the merged cindex space
let lhs_mat = lhs.matrix.remap_cindices(&lhs_cindex_map);
let rhs_mat = rhs.matrix.remap_cindices(&rhs_cindex_map);

// 3. binary_op merges entries at the same (col, new_cindex) — exactly right
let result_mat = lhs_mat + rhs_mat;  // or - for Sub

// 4. Return
Hamiltonian { matrix: result_mat, coeff_fns: merged_fns }
```

Example: if `H_A` and `H_B` both contain operators weighted by the same `Arc`
function `f`, the result has a single cindex for `f` and the stored values for
that cindex are summed in the underlying QMatrix:
`H_A(t) + H_B(t) = H_static + f(t) * (entries_A + entries_B)`.

---

## Part 3: `HamiltonianInner` (type erasure)

Mirrors `QMatrixInner` — 12 variants by `M × C`, wrapping `Hamiltonian<M, i64, C>`.
Methods expose `dim()`, `num_coeff()`, `dtype_name()`, `try_add`, `try_sub`, and
the time-parameterized operations (always returning `Complex<f64>`).

A `with_hamiltonian!` macro follows the same pattern as `with_qmatrix!`.

---

## Part 4: Python bindings

Python callables are wrapped in `quspin-py` as:

```rust
struct PyCoeffFn(Py<PyAny>);  // Py<PyAny> is Send + Sync

fn into_arc(self) -> Arc<dyn Fn(f64) -> Complex<f64> + Send + Sync> {
    Arc::new(move |t: f64| {
        Python::with_gil(|py| {
            let result = self.0.call1(py, (t,))...;
            // accept Python float or complex, convert to Complex<f64>
        })
    })
}
```

The GIL is re-acquired once per `eval_coeffs` call (not per matrix entry), so
overhead is bounded per matrix-vector operation.

---

## Order of implementation

1. `QMatrix` refactor (`M`/`V` split + `remap_cindices`)
2. `operator` module rename
3. `hamiltonian` module — `Hamiltonian<M, I, C>` struct + `HamiltonianInner`
4. `quspin-py` bindings — `PyCoeffFn` wrapper + `HamiltonianInner` exposure

---

## Implementation plan

### Current state (as of 2026-03-29)

| Component | Status | Notes |
|---|---|---|
| `QMatrix<V, I, C>` | Exists | Single type param `V`; no M/V split; no `remap_cindices` |
| `trait Hamiltonian<C>` / `SpinHamiltonian` / etc. | Exists | Needs rename to `operator` module |
| `Hamiltonian<M, I, C>` struct | Not started | |
| `HamiltonianInner` | Not started | |
| `TimeEvolution` / `SchrodingerEq` | Not started | |
| `quspin-py` bindings | Frozen | Full rewrite planned after quspin-core refactor |

---

### Step 0 — `QMatrix` refactor (`qmatrix/matrix.rs`, `qmatrix/ops.rs`, `qmatrix/build.rs`, `qmatrix/dispatch.rs`)

This is a prerequisite for everything else.

**0a — Rename type parameter `V → M`**

Mechanical rename throughout `matrix.rs`, `Entry`, `QMatrix`, `ops.rs`
(`binary_op`), `build.rs` (`build_from_basis`), and `dispatch.rs`
(`QMatrixInner` and `with_qmatrix!`).  No logic changes.

**0b — Split computation methods to accept a separate output type `V`**

Affect: `dot`, `dot_transpose`, `to_csr`, `to_csr_into`, `to_csr_nnz`,
`to_dense`, `to_dense_into`.  Each gains a type parameter `V: Primitive` for
the output/coefficient type, distinct from the stored type `M`.

The conversion path in method bodies becomes:

```rust
let entry_as_v = V::from_complex(entry.value.to_complex()); // M → Complex<f64> → V
let contrib = coeff[c] * entry_as_v * input[col];           // V × V × V → V
```

`check_coeff` and `check_dot_args` are updated to accept `&[V]` rather than
`&[M]`.  Call sites that were `dot::<V>(...)` must now supply `V` explicitly.

Update the `with_qmatrix!` macro: inject `type $M = ...` instead of
`type $V = ...`.

**0c — Add `remap_cindices`**

```rust
pub fn remap_cindices(self, mapping: &[C]) -> Self
```

Iterates `self.data`, replaces each `entry.cindex` with `mapping[entry.cindex.as_usize()]`,
then re-sorts each row by `(col, cindex)` and recomputes `num_coeff`.
Precondition (documented, not checked): the remapping does not create duplicate
`(col, cindex)` keys within any row.

**Tests to add:**
- `dot` with `M = f64`, `V = Complex<f64>` (cross-type call)
- `remap_cindices` round-trip: remap then verify row order and `num_coeff`

---

### Step 1 — `operator` module rename

All changes are in `crates/quspin-core/src/`:

| Old | New |
|---|---|
| `src/hamiltonian/` directory | `src/operator/` directory |
| `hamiltonian/mod.rs`: `pub trait Hamiltonian<C>` | `operator/mod.rs`: `pub trait Operator<C>` |
| `SpinHamiltonian` / `SpinHamiltonianInner` | `SpinOperator` / `SpinOperatorInner` |
| `FermionHamiltonian` / `FermionHamiltonianInner` | `FermionOperator` / `FermionOperatorInner` |
| `BosonHamiltonian` / `BosonHamiltonianInner` | `BosonOperator` / `BosonOperatorInner` |
| `HardcoreHamiltonian` / `HardcoreHamiltonianInner` | `HardcoreOperator` / `HardcoreOperatorInner` |
| `BondHamiltonian` / `BondHamiltonianInner` | `BondOperator` / `BondOperatorInner` |
| `lib.rs`: `pub use hamiltonian::{Hamiltonian, ParseOp}` | `pub use operator::{Operator, ParseOp}` |

`ParseOp`, `BondTerm`, all `*Op`, and `*OpEntry` types are **unchanged**.

Update `qmatrix/build.rs`: `Hamiltonian<C>` bound → `Operator<C>`.

Update `quspin-py` call sites (even though the module is frozen, it must compile).

---

### Step 2 — New `hamiltonian` module (`src/hamiltonian/mod.rs`)

Create the directory `src/hamiltonian/` with `mod.rs`.

**2a — `Hamiltonian<M, I, C>` struct and constructor**

```rust
pub struct Hamiltonian<M: Primitive, I: Index, C: CIndex> {
    matrix: QMatrix<M, I, C>,
    coeff_fns: Vec<Arc<dyn Fn(f64) -> Complex<f64> + Send + Sync>>,
}
```

Constructor validates `coeff_fns.len() == matrix.num_coeff().saturating_sub(1)`.

**2b — `eval_coeffs` and public query methods**

`eval_coeffs(&self, t: f64) -> Vec<Complex<f64>>` prepends the static `1.0`
and evaluates each function at `t`.

`dim()`, `num_coeff()` — delegate to `self.matrix`.

**2c — Time-parameterised output methods**

All output type is fixed to `Complex<f64>`.  Each method calls
`let coeffs = self.eval_coeffs(time)` then delegates to
`self.matrix.{method}::<Complex<f64>>(&coeffs, ...)`:

- `to_csr_nnz(time, drop_zeros) -> Result<usize>`
- `to_csr_into(time, drop_zeros, indptr, indices, data: &mut [Complex<f64>]) -> Result<()>`
- `to_csr(time, drop_zeros) -> Result<(Vec<I>, Vec<I>, Vec<Complex<f64>>)>`
- `to_dense_into(time, output: &mut [Complex<f64>]) -> Result<()>`
- `to_dense(time) -> Result<Vec<Complex<f64>>>`
- `dot(overwrite, time, input, output: &mut [Complex<f64>]) -> Result<()>`
- `dot_transpose(overwrite, time, input, output: &mut [Complex<f64>]) -> Result<()>`

**2d — `Add` / `Sub` with cindex merging**

Implement `Add<Output = Self>` and `Sub<Output = Self>` for
`Hamiltonian<M, I, C>` using the algorithm from the design doc:

1. Deduplicate `coeff_fns` from both operands by `Arc::ptr_eq`, building
   `lhs_cindex_map` and `rhs_cindex_map`.
2. Call `lhs.matrix.remap_cindices(&lhs_cindex_map)` and same for rhs.
3. Call `binary_op` (reuse from `ops.rs`) on the remapped matrices.
4. Wrap in `Hamiltonian { matrix: result_mat, coeff_fns: merged_fns }`.

`cindex 0` (the static part) always maps to `0` in both operands and is never
remapped.

**2e — `HamiltonianInner`**

12-variant enum mirroring `QMatrixInner`: `M × C`, index type fixed to `i64`.

```rust
pub enum HamiltonianInner {
    HMf64U8(Hamiltonian<f64, i64, u8>),
    HMf64U16(Hamiltonian<f64, i64, u16>),
    // … 10 more variants …
}
```

Methods exposed: `dim()`, `num_coeff()`, `dtype_name()`, `try_add`,
`try_sub`, and the time-parameterised operations (all returning
`Complex<f64>`).

Add `with_hamiltonian!` macro following the same pattern as `with_qmatrix!`.

Re-export from `lib.rs`:
```rust
pub mod hamiltonian;
pub use hamiltonian::{Hamiltonian, HamiltonianInner};
```

**Tests to add:**
- `eval_coeffs` returns `[1.0, f(t)]` for a one-function Hamiltonian.
- `dot` at time `t` equals manual `coeff_fns[0](t) * qmatrix.dot(...)`.
- `Add`: two Hamiltonians sharing the same `Arc` function collapse to one cindex.
- `Add`: two Hamiltonians with distinct functions preserve both cindices.

---

### Step 3 — `quspin-py` bindings (deferred)

`quspin-py` is frozen pending a full rewrite.  When that rewrite begins:

- Add `PyCoeffFn(Py<PyAny>)` wrapper implementing `Fn(f64) -> Complex<f64>`
  (GIL acquired once per `eval_coeffs` call).
- Expose `HamiltonianInner` as a Python class.
- Construct `HamiltonianInner` from a `QMatrixInner` + list of Python callables.

---

### Step 4 — Time evolution

**4a — Path A: internal stepper (`src/hamiltonian/evolve.rs`)**

Implement a self-contained DOP853 adaptive Runge-Kutta stepper operating on
`Vec<Complex<f64>>` (no `Copy` constraint, arbitrary runtime dimension):

```rust
pub struct TimeEvolution<M: Primitive, I: Index, C: CIndex> {
    hamiltonian: Hamiltonian<M, I, C>,
}

impl<M, I, C> TimeEvolution<M, I, C> {
    pub fn evolve(
        &self,
        psi: &mut Vec<Complex<f64>>,
        t0: f64,
        tf: f64,
        rtol: f64,
        atol: f64,
    ) -> Result<EvolutionStats, QuSpinError>
}
```

The RHS kernel is `dψ/dt = -i·H(t)·ψ`, computed by calling
`hamiltonian.dot(overwrite=true, t, psi, scratch)` and then
`output[k] = Complex { re: scratch[k].im, im: -scratch[k].re }`.

Define `EvolutionStats { steps: usize, rejected: usize }`.

This path has **no new dependencies**.

**4b — Path B: `differential-equations` bridge (`src/hamiltonian/schrodinger.rs`)**

Add an optional Cargo feature `differential-equations` gating this module.

```toml
[features]
differential-equations = ["dep:differential-equations", "dep:nalgebra", "dep:bytemuck"]
```

Add to `Cargo.toml` dependencies:
```toml
num-complex = { version = "0.4", features = ["bytemuck"] }
bytemuck = { version = "1", optional = true }
```

Implement `SchrodingerEq<M, I, C, const N>` with `ODE<f64, SVector<f64, N>>`,
using zero-copy `bytemuck::cast_slice` to reinterpret the interleaved
`SVector<f64, N>` as `&[Complex<f64>]` (valid because `Complex<f64>` is
`#[repr(C)]` with the same layout as `[f64; 2]`).

This path is opt-in; expose it from `quspin-py` only after Path A is stable.

---

## Part 5: Time evolution — `differential-equations` integration

### The Schrödinger equation

Time evolution of a quantum state `|ψ⟩` under `H(t)` is governed by:

```
d|ψ⟩/dt = -i · H(t) · |ψ⟩     (ℏ = 1)
```

where `|ψ⟩ ∈ ℂ^N`, `N = hamiltonian.dim()`. This is an ODE initial value problem
directly computable via `Hamiltonian::dot`.

### The `Copy` constraint problem

The `differential-equations` crate requires `State<T>: Copy`. From the source:

```rust
pub trait State<T: Real>: Clone + Copy + Debug
    + Add<Output = Self> + Sub<Output = Self> + AddAssign
    + Mul<T, Output = Self> + Div<T, Output = Self> + Neg<Output = Self>
{
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> T;
    fn set(&mut self, i: usize, value: T);
    fn zeros() -> Self;
}
```

Concrete `State` implementations are provided for `f32`, `f64`, `Complex<T>`, and
`SMatrix<T, R, C>` (fixed-size nalgebra matrices). `DVector<T>` is not supported and
cannot be, because heap-allocated types cannot be `Copy`.

Quantum basis dimensions are runtime values, so `SMatrix<f64, N, 1>` (which requires
compile-time `N`) is only usable for small, statically-known system sizes.

### Two integration paths

#### Path A: Internal stepper (recommended for general use)

Implement a `TimeEvolution` struct in quspin-core that drives the Schrödinger equation
directly against `Vec<Complex<f64>>`, bypassing the `Copy` constraint entirely:

```rust
pub struct TimeEvolution<M: Primitive, I: Index, C: CIndex> {
    hamiltonian: Hamiltonian<M, I, C>,
}

impl<M, I, C> TimeEvolution<M, I, C> {
    /// Evolve state |ψ⟩ from t0 to tf using an adaptive DOP853 stepper.
    /// `psi` is updated in-place.
    pub fn evolve(
        &self,
        psi: &mut Vec<Complex<f64>>,
        t0: f64,
        tf: f64,
        rtol: f64,
        atol: f64,
    ) -> Result<EvolutionStats, QuSpinError>
}
```

The `diff` kernel (called at each RK stage) is:

```rust
// dψ/dt = -i · H(t) · ψ
// Computed as: hamiltonian.dot(overwrite=true, time=t, input=psi, output=scratch)
// Then: output[k] = Complex { re: scratch[k].im, im: -scratch[k].re }
//       (multiply by -i: re part → im, im part → -re)
```

This path has no dependency on `differential-equations` and works for arbitrary
runtime-dimension bases.

#### Path B: `differential-equations` bridge (small, fixed-size systems)

For users who want to compose with the `differential-equations` ecosystem, provide a
const-generic bridge. The crate already ships a scalar `ODE<f64, Complex<f64>>`
example (`examples/ode/11_schrodinger`), which confirms complex state is supported
directly — but only as a length-2 scalar, not a vector.

For a vector state of compile-time size `DIM`, the state type is
`SVector<f64, {2*DIM}>` using an **interleaved** layout:

```
y = [re₀, im₀, re₁, im₁, …, re_{DIM-1}, im_{DIM-1}]
```

**Why interleaved, not blocked?** `Complex<f64>` is `#[repr(C)]`, so its in-memory
layout is identical to `[f64; 2]` (re first, then im). An interleaved `SVector<f64,
{2*DIM}>` is therefore **bit-for-bit identical** to `[Complex<f64>; DIM]`, enabling a
zero-copy reinterpret cast — analogous to `reinterpret_cast` in C++.

In Rust this is done via `bytemuck::cast_slice` (safe, zero-cost) once
`num-complex`'s `bytemuck` feature is enabled, which provides `Pod + Zeroable` impls
for `Complex<T>`:

```toml
# Cargo.toml
num-complex = { version = "0.4", features = ["bytemuck"] }
bytemuck     = "1"
```

The `diff` implementation reinterprets the `SVector` backing storage as
`&[Complex<f64>]` and passes it directly to `Hamiltonian::dot` — no copying, no
scratch buffer:

```rust
/// A `differential-equations`-compatible wrapper for time-evolving a quantum state.
///
/// `N` must equal 2 * hamiltonian.dim().
/// State layout: interleaved [re₀, im₀, re₁, im₁, …] so that the backing
/// storage is reinterpret-compatible with &[Complex<f64>; N/2].
pub struct SchrodingerEq<M: Primitive, I: Index, C: CIndex, const N: usize> {
    hamiltonian: Hamiltonian<M, I, C>,
}

impl<M, I, C, const N: usize> ODE<f64, SVector<f64, N>> for SchrodingerEq<M, I, C, N> {
    fn diff(&self, t: f64, y: &SVector<f64, N>, dydt: &mut SVector<f64, N>) {
        // SVector<f64, N> storage is a contiguous [f64; N].
        // Complex<f64> is #[repr(C)] ≡ [f64; 2], so with interleaved layout the
        // cast is valid: bytemuck::cast_slice rejects misaligned/misaligned slices
        // at runtime but will always succeed here (f64 alignment ≥ Complex<f64>).
        let psi:  &    [Complex<f64>] = bytemuck::cast_slice(    y.as_slice());
        let dpsi: &mut [Complex<f64>] = bytemuck::cast_slice_mut(dydt.as_mut_slice());

        // Compute H(t)·ψ, writing result into dpsi (overwrite=true)
        self.hamiltonian.dot(true, t, psi, dpsi).unwrap();

        // Multiply by -i in-place: -i·(a + ib) = b - ia
        for c in dpsi.iter_mut() {
            *c = Complex::new(c.im, -c.re);
        }
    }
}
```

**Setup and usage**:

```rust
const DIM: usize = /* basis.dim() — must be known at compile time */;
const N:   usize = 2 * DIM;

let eq = SchrodingerEq::<f64, i64, u8, N>::new(hamiltonian);

// Pack initial state into interleaved SVector
let psi0: Vec<Complex<f64>> = /* initial wavefunction */;
let y0: SVector<f64, N> = SVector::from_fn(|k, _| {
    if k % 2 == 0 { psi0[k / 2].re } else { psi0[k / 2].im }
});

let problem = ODEProblem::new(&eq, t0, tf, y0);
let mut solver = ExplicitRungeKutta::dop853().rtol(1e-8).atol(1e-10);
let solution = problem.solve(&mut solver)?;

// Unpack final state: reinterpret cast again, then copy out
let yf = solution.last()?.1;
let psi_f: &[Complex<f64>] = bytemuck::cast_slice(yf.as_slice());
```

The round-trip packing cost is paid only once at setup and teardown; each `diff`
call (of which there are O(steps × stages) ≈ hundreds to thousands) is zero-copy.

### Constraint summary

| | Path A (internal stepper) | Path B (`differential-equations`) |
|---|---|---|
| Basis dimension | Runtime (any size) | Compile-time `const N: usize` |
| State allocation | Heap (`Vec`) | Stack (`SVector<f64, N>`) |
| `Copy` required | No | Yes — met via `SVector` |
| `diff` allocation | None (in-place) | None — zero-copy `bytemuck` reinterpret cast |
| State layout | `Vec<Complex<f64>>` | Interleaved `[re₀, im₀, re₁, im₁, …]` |
| Rust edition | Stable | Stable |
| Max practical N | Unlimited | ~10⁵ (stack limit ~8 MB → `N ≤ ~500_000 f64s`, i.e. DIM ≤ ~250_000) |
| Ecosystem composability | quspin-only | Full `differential-equations` API (events, dense output, etc.) |

### Recommendation

Provide **Path A** as the primary API and expose it from `quspin-py`. Provide
**Path B** as an opt-in feature (`differential-equations` is an optional dependency)
for users building pure-Rust pipelines who want access to event detection, dense
output, or other solver features from that crate.
