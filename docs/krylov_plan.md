# Krylov Subspace Methods — Integration Plan

Two algorithms, both built on `ndarray-linalg`'s `krylov` module as the
orthogonalization back-end.

---

## Background: what ndarray-linalg provides

| Module | What it gives us |
|---|---|
| `krylov::arnoldi_mgs` | Builds a K-step Krylov basis `(Q, H)` from a matvec closure; `Q` is orthonormal, `H` is upper Hessenberg (tridiagonal for symmetric operators) |
| `lobpcg` | Truncated eigensolver — **not usable directly** because it requires SPD matrices; Hamiltonians are Hermitian but not necessarily positive definite |
| `eig` / LAPACK `eigh` | Dense Hermitian eigensolver — used to solve the small **K × K** projected problem after the Krylov basis is built |

The plan is:

1. Use `arnoldi_mgs` to build the K-dimensional Krylov subspace.
2. For eigenvalues: solve the projected K × K eigenvalue problem with LAPACK `eigh`, then map Ritz vectors back.
3. For time evolution: diagonalise the projected K × K Hamiltonian, apply `e^{-i T_K t}` in that space, project back. This is the **short iterative Lanczos (SIL)** propagator and is much more accurate per matvec than Runge–Kutta for Hamiltonian systems.

Both algorithms share the same Lanczos/Arnoldi kernel.

---

## Architecture

```
quspin-core
└── src/
    └── krylov/
        ├── mod.rs        – public API re-exports
        ├── basis.rs      – LanczosBasis: wraps arnoldi_mgs, owns (Q, T)
        ├── eig.rs        – LanczosEig: Ritz value/vector extraction
        └── propagate.rs  – KrylovPropagator: SIL time-stepper
```

`quspin-py` adds two new Python classes in `src/krylov.rs`:
- `EigSolver` — eigenvalue/eigenvector solver
- `KrylovPropagator` — time-stepping via SIL (replaces or supplements `SchrodingerEq`)

---

## Step 0 — Add dependencies

In `Cargo.toml` workspace:
```toml
ndarray-linalg = { version = "0.18", features = ["openblas-static"] }
```

In `quspin-core/Cargo.toml`:
```toml
ndarray-linalg = { workspace = true }
```

In `quspin-py/Cargo.toml`:
```toml
ndarray-linalg = { workspace = true }
```

- [ ] Add `ndarray-linalg` to workspace `Cargo.toml`
- [ ] Add as dependency in `quspin-core` and `quspin-py`
- [ ] `cargo check --workspace` passes

---

## Step 1 — `LanczosBasis` (quspin-core)

`src/krylov/basis.rs`

```rust
pub struct LanczosBasis {
    /// Orthonormal Krylov vectors, shape (dim, K).
    pub q: Array2<Complex<f64>>,
    /// Projected tridiagonal (or Hessenberg) matrix, shape (K, K).
    pub h: Array2<Complex<f64>>,
    /// Number of steps actually taken (may be < K if early convergence).
    pub steps: usize,
}
```

Constructor:
```rust
impl LanczosBasis {
    /// Build a K-step Arnoldi/Lanczos factorisation.
    ///
    /// `matvec: impl FnMut(ArrayView1<Complex<f64>>) -> Array1<Complex<f64>>`
    /// is called once per step; for time-independent H pass `|v| H.dot(v)`.
    pub fn build<F>(v0: ArrayView1<Complex<f64>>, k: usize, matvec: F, tol: f64) -> Self
    where
        F: FnMut(ArrayView1<Complex<f64>>) -> Array1<Complex<f64>>,
    { ... }
}
```

Internally calls `ndarray_linalg::krylov::arnoldi_mgs`.

- [ ] `src/krylov/basis.rs` implemented and tested
- [ ] `cargo test -p quspin-core` passes

---

## Step 2 — `LanczosEig` (quspin-core)

`src/krylov/eig.rs`

Solves the K × K projected eigenvalue problem using `ndarray_linalg::eigh`
(LAPACK `dsyevd`/`zheevd`), then maps Ritz vectors back to the full space.

```rust
pub struct EigResult {
    pub eigenvalues: Array1<f64>,          // length min(k_req, K)
    pub eigenvectors: Array2<Complex<f64>>, // shape (dim, k_req)
    pub residuals: Array1<f64>,
}
```

```rust
pub fn lanczos_eig(
    dim: usize,
    matvec: impl FnMut(ArrayView1<Complex<f64>>) -> Array1<Complex<f64>>,
    v0: ArrayView1<Complex<f64>>,
    k_krylov: usize,   // Krylov space dimension
    k_wanted: usize,   // how many Ritz pairs to return
    which: Which,      // SA (smallest algebraic) | LA | SM (smallest magnitude)
    tol: f64,
) -> EigResult
```

- [ ] `src/krylov/eig.rs` implemented
- [ ] Unit test: 1-D XX chain ground state matches exact diag (2-site, 4-site)
- [ ] `cargo test -p quspin-core` passes

---

## Step 3 — `KrylovPropagator` (quspin-core)

`src/krylov/propagate.rs`

Short iterative Lanczos (SIL) time stepper.  For each time step `[t, t+dt]`:

1. Build a K-step Krylov basis from `|ψ(t)⟩` using `H(t + dt/2)` (midpoint).
2. Diagonalise the K × K projected Hamiltonian: `T_K = V Λ V†`.
3. Apply `e^{-i Λ dt}` in the K-space: `|ψ(t+dt)⟩ ≈ Q V e^{-i Λ dt} V† e₁`.
4. Optionally check norm preservation and halve `dt` if error exceeds tolerance.

```rust
pub struct KrylovPropagator {
    pub k: usize,    // Krylov dimension per step
    pub dt: f64,     // default step size
    pub tol: f64,    // norm-preservation tolerance
}

impl KrylovPropagator {
    pub fn step(
        &self,
        t: f64,
        psi: ArrayViewMut1<Complex<f64>>,
        matvec: impl FnMut(f64, ArrayView1<Complex<f64>>) -> Array1<Complex<f64>>,
    ) -> f64  // returns actual dt used

    pub fn integrate(
        &self,
        t0: f64,
        t_end: f64,
        psi0: ArrayView1<Complex<f64>>,
        matvec: impl FnMut(f64, ArrayView1<Complex<f64>>) -> Array1<Complex<f64>>,
    ) -> Array1<Complex<f64>>
}
```

- [ ] `src/krylov/propagate.rs` implemented
- [ ] Unit test: Pauli-X 1-site, |0⟩ → t=π/2, matches −i|1⟩ to tol 1e-10
- [ ] Compare norm conservation vs Dopri5 for a medium Hamiltonian
- [ ] `cargo test -p quspin-core` passes

---

## Step 4 — Python bindings

`quspin-py/src/krylov.rs`

### `EigSolver`

```python
class EigSolver:
    def __init__(self, hamiltonian: Hamiltonian, k_krylov: int = 100) -> None: ...

    def solve(
        self,
        time: float,
        v0: npt.NDArray[Any],
        k: int,
        which: str = "SA",   # "SA" | "LA" | "SM"
        tol: float = 1e-10,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]:
        """Returns (eigenvalues, eigenvectors) where eigenvectors has shape (dim, k)."""
```

Calls `HamiltonianInner::dot(overwrite=true, time, ...)` as the matvec; releases
the GIL for the full solve via `py.allow_threads`.

### `KrylovPropagator` (Python)

```python
class KrylovPropagator:
    def __init__(self, hamiltonian: Hamiltonian, k: int = 20, dt: float = 0.01, tol: float = 1e-10) -> None: ...

    def integrate(
        self,
        t0: float,
        t_end: float,
        y0: npt.NDArray[Any],
    ) -> npt.NDArray[Any]: ...

    def integrate_dense(
        self,
        t0: float,
        t_end: float,
        y0: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]: ...
```

- [ ] `src/krylov.rs` with `PyEigSolver` and `PyKrylovPropagator`
- [ ] Register both in `lib.rs`
- [ ] Add to `_rs.pyi`
- [ ] Python tests: ground state energy, norm conservation, Pauli-X evolution
- [ ] `cargo check -p quspin-py` passes

---

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| SPD requirement of LOBPCG | Not used | Hamiltonians are Hermitian but not SPD; implement Lanczos directly |
| Orthogonalisation | `arnoldi_mgs` from ndarray-linalg | Battle-tested, handles near-linear-dependence |
| Projected eigensolver | `ndarray_linalg::eigh` (LAPACK) | K is small (10–200), dense LAPACK is optimal |
| BLAS/LAPACK back-end | `openblas-static` | No system library required on CI |
| Time-dependent H in SIL | Midpoint rule `H(t + dt/2)` | Simple, second-order accurate |
| GIL strategy | `py.allow_threads` + `Python::with_gil` in coeff callbacks | Consistent with existing ODE integrator |
