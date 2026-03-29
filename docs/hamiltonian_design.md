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
