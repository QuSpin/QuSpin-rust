# QuSpin-rust

Rust rewrite of the QuSpin quantum many-body physics library. Complete rewrite (not a port) — idiomatic Rust throughout, no backward compatibility constraints.

## Project Structure

Cargo workspace with two crates (a third, `bitbasis`, is planned):

```
crates/
  quspin-core/   # Pure Rust physics library — no Python dependency
  quspin-py/     # PyO3 bindings (cdylib named `_rs`)
python/
  quspin_rs/     # Python package (imports _rs extension)
  tests/         # pytest suite
docs/            # Design documents
```

**Key design rule:** `quspin-core` is pure generic Rust with no runtime dispatch. Dispatch happens once at the PyO3 boundary in `quspin-py` via `macro_rules!`. Never add variant types or runtime dispatch to `quspin-core`.

## Build & Dev Commands

Requires: Rust (stable), Python >= 3.10, [uv](https://github.com/astral-sh/uv), [just](https://github.com/casey/just)

```sh
just sync              # uv sync --dev --all-extras
just check             # cargo check (fast, no linking)
just develop           # uv run maturin develop (build + install extension)
just test-rust         # cargo test -p quspin-core
just test-python       # uv run pytest python/tests/ -v
just test-python-fast  # pytest -m "not slow" (skip slow integration tests)
just test              # run both Rust and Python tests
```

## Testing

- **Rust tests:** `cargo test -p quspin-core` (quspin-py has no standalone Rust tests — it requires Python)
- **Python tests:** `uv run pytest python/tests/ -v`
- **Slow tests:** marked with `@pytest.mark.slow`, skip with `-m "not slow"`
- **Parallel:** `just test-python-parallel` (pytest-xdist, `-n auto`)

## Linting & Formatting

Pre-commit hooks enforce all of these. Run `pre-commit run --all-files` or let them run on commit.

- **Rust:** `cargo fmt --all` and `cargo clippy -p quspin-core -p quspin-py --all-targets -- -D warnings`
- **Python:** black (line-length 88), isort (black profile), ruff, pyright

## CI

GitHub Actions (`.github/workflows/ci.yaml`): cargo test + clippy on Rust, maturin develop + pytest on Python. Runs on push to main and PRs.

## Architecture Notes

- **Parallelism:** `rayon` throughout (replaces C++ OpenMP)
- **Multi-word integers:** `ruint::Uint<BITS, LIMBS>` for large bit representations
- **Error handling:** `Result<T, QuSpinError>` in core; maps to Python exceptions via PyO3
- **Python package:** `quspin-rs`, module `quspin_rs._rs`, built with maturin
- **Type stubs:** `python/quspin_rs/_rs.pyi`

See `docs/` for detailed design documents.
