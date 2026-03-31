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

- [ ] `src/operator/pauli.rs` — `PyPauliOperator` wrapping `HardcoreOperatorInner`
  - Constructor: `PyPauliOperator(terms)` where `terms` is the nested list format from old tests
  - Properties: `max_site`, `num_cindices`
- [ ] `src/operator/bond.rs` — `PyBondOperator` + `PyBondTerm` wrapping `BondOperatorInner`
- [ ] `src/operator/boson.rs` — `PyBosonOperator` wrapping `BosonOperatorInner`
- [ ] `src/operator/fermion.rs` — `PyFermionOperator` wrapping `FermionOperatorInner`
- [ ] `src/operator/mod.rs` wiring
- [ ] `cargo build -p quspin-py` passes
- [ ] Write tests in `python/tests/test_operators.py`
- [ ] Commit: `feat(quspin-py): add operator bindings (Pauli, bond, boson, fermion)`

---

## Step 4 — `PyQMatrix` rewrite

Rewrite `src/qmatrix.rs` to use the new basis and operator types.
The `build_*` family is replaced by a single dispatch pattern: one `build`
static method per operator kind that accepts the new `Py*Basis` types.

- [ ] `PyQMatrix { inner: QMatrixInner }` struct
- [ ] `PyQMatrix::build_pauli(op, basis, dtype)` — `PyPauliOperator` + `PySpinBasis` or `PyFermionBasis`
- [ ] `PyQMatrix::build_bond(op, basis, dtype)` — `PyBondOperator` + `PySpinBasis`
- [ ] `PyQMatrix::build_boson(op, basis, dtype)` — `PyBosonOperator` + `PyBosonBasis`
- [ ] `PyQMatrix::build_fermion(op, basis, dtype)` — `PyFermionOperator` + `PyFermionBasis`
- [ ] `dot(coeff, input, output, overwrite)` — 1-D arrays
- [ ] `dot_many(coeff, input, output, overwrite)` — 2-D arrays `(dim, n_vecs)`
- [ ] `dot_transpose(coeff, input, output, overwrite)`
- [ ] `dot_transpose_many(coeff, input, output, overwrite)`
- [ ] `to_csr(coeff, drop_zeros) -> (indptr, indices, data)`
- [ ] `to_dense(coeff) -> ndarray`
- [ ] `__add__`, `__sub__`
- [ ] Properties: `dim`, `nnz`, `dtype`, `__repr__`
- [ ] `cargo build -p quspin-py` passes
- [ ] Write tests in `python/tests/test_qmatrix.py`
- [ ] Commit: `feat(quspin-py): rewrite PyQMatrix with new basis/operator types`

---

## Step 5 — `PyHamiltonian` binding

Expose `HamiltonianInner` to Python. A `PyHamiltonian` is constructed from a
`PyQMatrix` plus a list of Python callables (one per time-dependent operator string).

- [ ] `src/hamiltonian.rs` — `PyHamiltonian { inner: HamiltonianInner }`
- [ ] `PyHamiltonian::new(qmatrix, coeff_fns)` — wraps each Python callable in an
  `Arc<dyn Fn(f64) -> Complex<f64>>` via a thin trampoline that acquires the GIL
- [ ] `dot(time, input, output, overwrite)` — 1-D
- [ ] `dot_many(time, input, output, overwrite)` — 2-D `(dim, n_vecs)`
- [ ] `dot_transpose(time, input, output, overwrite)`
- [ ] `dot_transpose_many(time, input, output, overwrite)`
- [ ] `to_csr(time, drop_zeros) -> (indptr, indices, data)`
- [ ] `to_dense(time) -> ndarray`
- [ ] Properties: `dim`, `num_coeff`, `dtype`, `__repr__`
- [ ] `cargo build -p quspin-py` passes
- [ ] Write tests in `python/tests/test_hamiltonian.py`
- [ ] Commit: `feat(quspin-py): add PyHamiltonian binding`

---

## Step 6 — `PySchrodingerEq` binding

Expose time evolution via `ode_solvers`. At the Python level the user provides
a `PyHamiltonian` and an initial state; integration is entirely in Rust.

Because `SchrodingerEq<M,I,C>` is generic, we need a type-erased wrapper.
The cleanest approach: add a `SchrodingerEqInner` enum (12 variants, one per
`HamiltonianInner` variant) in `quspin-core/src/hamiltonian/` or store the
`HamiltonianInner` in a newtype that implements `System` by dispatching through
`HamiltonianInner::dot`.

- [ ] Add `SchrodingerEqInner` to `quspin-core` (or a dispatch-capable newtype)
  that implements `System<f64, DVector<f64>>` via `HamiltonianInner::dot`
- [ ] `src/schrodinger.rs` — `PySchrodingerEq { inner: SchrodingerEqInner }`
- [ ] `PySchrodingerEq::new(hamiltonian: &PyHamiltonian)`
- [ ] `PySchrodingerEq::integrate(t0, t_end, y0, rtol, atol) -> ndarray` —
  runs Dopri5 and returns the final state
- [ ] Properties: `dim`, `hamiltonian`
- [ ] `cargo build -p quspin-py` passes
- [ ] Write tests in `python/tests/test_schrodinger.py`
  - Test: Pauli-X 1-site, |0⟩ → t=π/2 → matches -i|1⟩ (tol 1e-6)
- [ ] Commit: `feat(quspin-py): add PySchrodingerEq binding`

---

## Step 7 — Wire up, stubs, and cleanup

- [ ] Update `src/lib.rs` `#[pymodule]` to register all new classes
- [ ] Update `python/quspin_rs/__init__.py` exports
- [ ] Write `python/quspin_rs/_rs.pyi` stub file with all public classes and methods
- [ ] Update `python/tests/test_rs.py` to use the new class names and API
- [ ] Run full Python test suite: `pytest python/tests/`
- [ ] Commit: `feat(quspin-py): wire up module, add .pyi stubs, update tests`
