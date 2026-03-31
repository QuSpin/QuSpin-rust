# Python Bindings Implementation Plan

Rewrite `quspin-py` from scratch against the current `quspin-core` API.
Each step ends with a passing `cargo build` + `cargo test` (Rust side) and a commit.
Check off sub-tasks as they are completed.

**How to resume:** read the checkboxes — the first unchecked step is where to continue.

---

## Step 0 — Fill `HamiltonianInner` and `QMatrixInner` gaps

The audit found that `dot_many` / `dot_transpose_many` were added to `Hamiltonian<M,I,C>`
and `QMatrix<M,I,C>` but not to their type-erased `Inner` wrappers.

- [x] Add `dot_many` to `HamiltonianInner` (delegates via `with_hamiltonian!`)
- [x] Add `dot_transpose_many` to `HamiltonianInner`
- [x] Add `dot_many` to `QMatrixInner` (delegates via `with_qmatrix!`)
- [x] Add `dot_transpose_many` to `QMatrixInner`
- [x] `cargo test -p quspin-core` passes
- [x] Commit: `feat(quspin-core): add dot_many/dot_transpose_many to Inner wrappers`

---

## Step 1 — Delete all dead code from `quspin-py`

The frozen crate references types that are marked for deletion
(`SymGrpBase`, `DitSymGrp`, old `SpinSymGrp`, etc.) and will not compile
once we remove them from `quspin-core`. Wipe the slate clean.

- [x] Delete `src/basis/hardcore.rs` (wraps old `PyHardcoreBasis` / `PySpinSymGrp`)
- [x] Delete `src/basis/dit.rs` (wraps old `PyDitBasis` / `PyDitSymGrp`)
- [x] Delete `src/basis/symmetry.rs` (wraps old `PyFermionicSymGrp`)
- [x] Delete `src/hamiltonian/hardcore.rs` (`PyHardcoreHamiltonian` — old operator type)
- [x] Delete `src/hamiltonian/bond.rs` (`PyBondHamiltonian` / `PyBondTerm`)
- [x] Delete `src/hamiltonian/boson.rs` (`PyBosonHamiltonian`)
- [x] Delete `src/hamiltonian/fermion.rs` (`PyFermionHamiltonian`)
- [x] Delete `src/hamiltonian/parse.rs` (parser helpers for the old operator types)
- [x] Replace `src/qmatrix.rs` with a stub (`pub struct PyQMatrix;`) so it compiles
- [x] Strip `src/lib.rs` down to an empty `#[pymodule]` that registers nothing
- [x] Strip `src/basis/mod.rs` and `src/hamiltonian/mod.rs` to empty `pub mod` declarations
- [x] `cargo check -p quspin-py` passes
- [x] Commit: `chore(quspin-py): delete all dead code ahead of rewrite`

---

## Step 2 — Basis bindings

Expose `SpaceInner` (29-variant enum, already complete in `quspin-core`) to Python
via three typed classes. Each class wraps `SpaceInner` and has:

- constructor (`full`, `subspace`, `symmetric` class-methods)
- properties: `n_sites`, `lhss`, `size`, `is_built`
- methods: `state_at(i) -> str`, `index(state_str) -> int`

### `PySpinBasis` (`src/basis/spin.rs`)
Wraps `SpinBasis` → `SpaceInner`. LHSS is always 2.

- [x] `PySpinBasis::full(n_sites, lhss=2)` — calls `SpinBasis::new(n_sites, lhss, SpaceKind::Full)`
- [ ] `PySpinBasis::subspace(seeds, ham)` — `SpaceKind::Sub` with seed list *(Step 3)*
- [ ] `PySpinBasis::symmetric(seeds, ham, symmetries)` — `SpaceKind::Symm` *(Step 3)*
- [x] Properties: `n_sites`, `lhss`, `size`, `is_built`
- [x] Methods: `state_at`, `index`

### `PyFermionBasis` (`src/basis/fermion.rs`)
Wraps `FermionBasis` → `SpaceInner`. LHSS is always 2.

- [x] `PyFermionBasis::full(n_sites)`
- [ ] `PyFermionBasis::subspace(seeds, ham)` *(Step 3)*
- [ ] `PyFermionBasis::symmetric(seeds, ham, symmetries)` *(Step 3)*
- [x] Properties + methods (same as spin)

### `PyBosonBasis` (`src/basis/boson.rs`)
Wraps `BosonBasis` → `SpaceInner`. LHSS is user-supplied (≥ 2).

- [x] `PyBosonBasis::full(n_sites, lhss)`
- [ ] `PyBosonBasis::subspace(seeds, ham)` *(Step 3)*
- [ ] `PyBosonBasis::symmetric(seeds, ham, symmetries)` *(Step 3)*
- [x] Properties: `n_sites`, `lhss`, `size`, `is_built`
- [x] Methods: `state_at`, `index`

### Symmetry group helpers
The symmetry API needs a small adapter so Python can build lattice / local / inverse
symmetry generators. Expose as plain Python-facing builder (not a pyclass):
pass a list of `(perm, grp_char)` tuples for lattice symmetries and
`(locs, grp_char)` tuples for inversion-like symmetries.

- [ ] `src/basis/symmetry.rs` — helper types for building `SymmetryGrp` from Python lists *(Step 3)*
- [x] Wire `src/basis/mod.rs`
- [x] `cargo check -p quspin-py` passes
- [ ] Write tests in `python/tests/test_basis.py` covering full/subspace/symmetric for all three basis types *(Step 3 adds subspace/symmetric; full tests can be added now)*
- [x] Commit: `feat(quspin-py): add PySpinBasis, PyFermionBasis, PyBosonBasis (full constructors)`

---

## Step 3 — Operator bindings

Expose the four operator types (Pauli/hardcore, bond, boson, fermion) used as
inputs to `PyQMatrix.build`. These replace the old `Py*Hamiltonian` operator
classes (same concept, new implementations matching the current `quspin-core` API).

- [x] `src/operator/pauli.rs` — `PyPauliOperator` wrapping `HardcoreOperatorInner`
  - Constructor: `PauliOperator(terms)` where `terms = [(coeff, op_str, sites, cindex), ...]`
  - Properties: `max_site`, `num_cindices`, `lhss`
- [x] `src/operator/bond.rs` — `PyBondOperator` wrapping `BondOperatorInner`
  - Constructor: `BondOperator(terms)` where `terms = [(matrix_ndarray, bonds, cindex), ...]`
- [x] `src/operator/boson.rs` — `PyBosonOperator` wrapping `BosonOperatorInner`
- [x] `src/operator/fermion.rs` — `PyFermionOperator` wrapping `FermionOperatorInner`
- [x] `src/operator/mod.rs` wiring
- [x] Added `SpinBasis::build_hardcore` to `quspin-core` so `PauliOperator` drives BFS for spin-½ bases
- [x] Added `subspace`/`symmetric` constructors to `PySpinBasis`, `PyFermionBasis`, `PyBosonBasis`
- [x] `cargo check -p quspin-py` passes, `cargo test -p quspin-core` passes
- [ ] Write tests in `python/tests/test_operators.py` *(Step 7)*
- [x] Commit: `feat(quspin-py): add operator bindings (Pauli, bond, boson, fermion)`

---

## Step 4 — `PyQMatrix` rewrite

Rewrite `src/qmatrix.rs` to use the new basis and operator types.
The `build_*` family is replaced by a single dispatch pattern: one `build`
static method per operator kind that accepts the new `Py*Basis` types.

- [x] `PyQMatrix { inner: QMatrixInner }` struct
- [x] `PyQMatrix::build_pauli(op, basis, dtype)` — `PyPauliOperator` + `PySpinBasis` or `PyFermionBasis`
- [x] `PyQMatrix::build_bond(op, basis, dtype)` — `PyBondOperator` + `PySpinBasis`
- [x] `PyQMatrix::build_boson(op, basis, dtype)` — `PyBosonOperator` + `PyBosonBasis`
- [x] `PyQMatrix::build_fermion(op, basis, dtype)` — `PyFermionOperator` + `PyFermionBasis`
- [ ] `dot(coeff, input, output, overwrite)` — 1-D arrays *(deferred to Step 7)*
- [x] `dot_many(coeff, input, output, overwrite)` — 2-D arrays `(dim, n_vecs)`
- [ ] `dot_transpose(coeff, input, output, overwrite)` *(deferred to Step 7)*
- [x] `dot_transpose_many(coeff, input, output, overwrite)`
- [x] `to_csr(coeff, drop_zeros) -> (indptr, indices, data)`
- [ ] `to_dense(coeff) -> ndarray` *(deferred to Step 7)*
- [x] `__add__`, `__sub__`
- [x] Properties: `dim`, `nnz`, `dtype`, `__repr__`
- [x] `cargo check -p quspin-py` passes
- [ ] Write tests in `python/tests/test_qmatrix.py` *(Step 7)*
- [x] Commit: `feat(quspin-py): rewrite PyQMatrix with new basis/operator types`

---

## Step 5 — `PyHamiltonian` binding

Expose `HamiltonianInner` to Python. A `PyHamiltonian` is constructed from a
`PyQMatrix` plus a list of Python callables (one per time-dependent operator string).

- [x] `src/hamiltonian/mod.rs` — `PyHamiltonian { matrix: QMatrixInner, coeff_fns: Vec<PyObject> }`
       (stores callables directly at Python boundary rather than wrapping in Arc; avoids Send+Sync issues)
- [x] `PyHamiltonian::new(qmatrix, coeff_fns)` — validates length, clones `QMatrixInner`
- [x] `dot(time, input, output, overwrite)` — 1-D
- [x] `dot_many(time, input, output, overwrite)` — 2-D `(dim, n_vecs)`
- [ ] `dot_transpose(time, input, output, overwrite)` *(deferred to Step 7)*
- [x] `dot_transpose_many(time, input, output, overwrite)`
- [x] `to_csr(time, drop_zeros) -> (indptr, indices, data)`
- [ ] `to_dense(time) -> ndarray` *(deferred to Step 7)*
- [x] Properties: `dim`, `num_coeff`, `dtype`, `__repr__`
- [x] `cargo check -p quspin-py` passes
- [ ] Write tests in `python/tests/test_hamiltonian.py` *(Step 7)*
- [x] Commit: `feat(quspin-py): add PyHamiltonian binding`

---

## Step 6 — `PySchrodingerEq` binding

Expose time evolution via `ode_solvers`. At the Python level the user provides
a `PyHamiltonian` and an initial state; integration is entirely in Rust.

Because `SchrodingerEq<M,I,C>` is generic, we need a type-erased wrapper.
The cleanest approach: add a `SchrodingerEqInner` enum (12 variants, one per
`HamiltonianInner` variant) in `quspin-core/src/hamiltonian/` or store the
`HamiltonianInner` in a newtype that implements `System` by dispatching through
`HamiltonianInner::dot`.

- [x] `src/schrodinger.rs` — `PySchrodingerEq { matrix: QMatrixInner, coeff_fns: Vec<PyObject> }`
       (inline `SchrodingerSystem<'a,'py>` borrows from self during `integrate`; no SchrodingerEqInner needed)
- [x] `PySchrodingerEq::new(py, hamiltonian: &PyHamiltonian)` — clones QMatrixInner + clone_ref each callable
- [x] `PySchrodingerEq::integrate(t0, t_end, y0, rtol, atol) -> ndarray` — Dopri5, returns final state
- [x] `PySchrodingerEq::integrate_dense(...)` → (times, states) 2-D array
- [x] Properties: `dim`, `__repr__`
- [x] `cargo check -p quspin-py` passes
- [ ] Write tests in `python/tests/test_schrodinger.py` *(Step 7)*
  - Test: Pauli-X 1-site, |0⟩ → t=pi/2 → matches -i|1⟩ (tol 1e-6)
- [x] Commit: `feat(quspin-py): add PySchrodingerEq binding`

---

## Step 7 — Wire up, stubs, and cleanup

- [x] Update `src/lib.rs` `#[pymodule]` to register all new classes (done in Steps 4-6)
- [x] Update `python/quspin_rs/__init__.py` exports
- [x] Write `python/quspin_rs/_rs.pyi` stub file with all public classes and methods
- [x] Update `python/tests/test_rs.py` to use the new class names and API
- [ ] Run full Python test suite: `pytest python/tests/` (requires `maturin develop` first)
- [x] Commit: `feat(quspin-py): wire up module, add .pyi stubs, update tests`
