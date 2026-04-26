# QuSpin-rust

Rust rewrite of the QuSpin quantum many-body physics library. Complete rewrite (not a port) — idiomatic Rust throughout, no backward compatibility constraints.

## Project Structure

Cargo workspace split into seven focused physics crates plus a facade and the PyO3 bindings:

```
crates/
  quspin-types/      # Foundation: QuSpinError, Primitive, dtype, compute, LinearOperator trait
  quspin-bitbasis/   # Bit-level integer manipulation (BitInt, DitManip, Benes network)
  quspin-operator/   # Operator types + *OperatorInner enums (no basis knowledge)
  quspin-basis/      # SpaceInner, SymBasis, BFS, orbit, enumeration
  quspin-expm/       # Taylor-series matrix exponential (generic over LinearOperator<V>)
  quspin-krylov/     # Lanczos, FTLM, LTLM (takes matvec closures, no concrete types)
  quspin-matrix/     # QMatrix, Hamiltonian, QMatrixOperator, apply glue
  quspin-core/       # Pure re-export facade (~90 lines, zero logic)
  quspin-py/         # PyO3 bindings (cdylib named `_rs`)
python/
  quspin_rs/         # Python package (imports _rs extension)
  tests/             # pytest suite
docs/                # Design documents
```

### Dependency DAG

```
quspin-types
     │
quspin-bitbasis
     │
     ├────────────────┬──────────────┬──────────────┐
quspin-operator  quspin-basis   quspin-expm    quspin-krylov
     └────────┬───────┘
       quspin-matrix
            │
       quspin-core  →  quspin-py
```

All four crates at the mid level (`quspin-operator`, `quspin-basis`, `quspin-expm`, `quspin-krylov`) compile in parallel off `quspin-bitbasis`. `quspin-expm` and `quspin-krylov` depend only on `quspin-types`; `quspin-basis` depends on `quspin-bitbasis` (for `BitInt` / `StateGraph`) but has **no** runtime edge to `quspin-operator` — basis BFS drives any `&impl StateGraph` regardless of operator type.

### Key design rules

- **No runtime dispatch at crate boundaries.** `*OperatorInner` enums dispatch the cindex-width choice (`u8` vs `u16`) but that's it. The only `dyn Trait` is `DynLinearOperator<V> = Box<dyn LinearOperator<V> + Send + Sync>`.
- **Static dispatch across crate boundaries** via generics. Rust monomorphises at link time.
- `quspin-core` is a pure facade — never add logic there. Add to the focused crate that owns the domain.
- **`StateGraph` trait** (in `quspin-bitbasis`) is the connectivity abstraction `SpinBasis::build` / `BosonBasis::build` / `FermionBasis::build` / `GenericBasis::build` take. Every `*Operator<C>` and `*OperatorInner` impls it — callers do `basis.build(&op.inner, seeds)` or `basis.build(&op, seeds)`.
- **`OperatorDispatch` trait** (in `quspin-matrix`) carries the basis-dependent methods (`apply_and_project_to`, `apply`) on `*OperatorInner`. Consumers (e.g. `quspin-py`) need `use quspin_core::OperatorDispatch;` for method-call syntax.
- **`Operator<C>` trait** (in `quspin-operator`) defines `max_site`, `num_cindices`, `lhss`, `apply`. All concrete operator types implement it; `StateGraph` is derived from `apply` by dropping the cindex argument.
- **`SymElement<L>` + `add_symmetry`** (in `quspin-basis`) is the single user-facing shape for group elements on `SymBasis`. Constructors `SymElement::lattice(perm)`, `::local(op)`, `::composite(perm, op)` produce one element; the identity is always implicit and rejected if added explicitly. `SymBasis` stores every element in one of three typed vectors (`lattice_only`, `local_only`, `composite`) so the orbit hot loop is variant-free. `SymBasis::build` runs an `O(|G|²·probes)` closure + 1D-character validation (`χ(g·h) = χ(g)·χ(h)`) before BFS, catching missing closure / bad characters / duplicate actions at build time.

## Build & Dev Commands

Requires: Rust (stable), Python >= 3.10, [uv](https://github.com/astral-sh/uv), [just](https://github.com/casey/just)

```sh
just sync              # uv sync --dev --all-extras
just check             # cargo check --workspace (fast, no linking)
just develop           # uv run maturin develop (build + install extension)
just test-rust         # cargo test --workspace
just test-python       # uv run pytest python/tests/ -v
just test-python-fast  # pytest -m "not slow" (skip slow integration tests)
just test              # run both Rust and Python tests
```

## Testing

- **Rust tests:** `cargo test --workspace` (quspin-py has no standalone Rust tests — it requires Python)
- **Single crate:** `cargo test -p quspin-basis` (or any other `quspin-*` crate)
- **Python tests:** `uv run pytest python/tests/ -v`
- **Slow tests:** marked with `@pytest.mark.slow`, skip with `-m "not slow"`
- **Parallel:** `just test-python-parallel` (pytest-xdist, `-n auto`)

## Linting & Formatting

Pre-commit hooks enforce all of these. Run `pre-commit run --all-files` or let them run on commit.

- **Rust:** `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`
- **Python:** black (line-length 88), isort (black profile), ruff, pyright

## Git

- **Default branch:** `main`

## CI

GitHub Actions (`.github/workflows/ci.yaml`): `Swatinem/rust-cache` warms `target/`, then maturin develop + pytest + cargo test + clippy. `Cargo.lock` is tracked so the cache key stays stable across runs. Runs on push to main and PRs.

## Architecture Notes

- **Parallelism:** `rayon` throughout (replaces C++ OpenMP)
- **Multi-word integers:** `ruint::Uint<BITS, LIMBS>` for large bit representations
- **Error handling:** `Result<T, QuSpinError>` in the physics crates; maps to Python exceptions via PyO3
- **Python package:** `quspin-rs`, module `quspin_rs._rs`, built with maturin against `quspin-py`
- **Type stubs:** `python/quspin_rs/_rs.pyi`
- **`large-int` feature:** gates `Uint<512..8192>` variants; declared on `quspin-bitbasis`, `quspin-basis`, `quspin-matrix`, and `quspin-core` (which forwards through the chain). `quspin-py` re-exposes it as its own feature, off by default — `cargo build` / `maturin develop` skip the wide-integer monomorphizations for faster dev iteration. Enable explicitly with `cargo build --features quspin-py/large-int` or `maturin develop --features large-int`. CI runs a separate `cargo check`/`clippy` pass with the feature on so the gated paths don't bit-rot.

See `docs/` for detailed design documents, including `docs/superpowers/specs/2026-04-18-crate-split-design.md` for the crate-split rationale.
