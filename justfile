# Default recipe
default:
    @just --list

# ── Dev ─────────────────────────────────────────────────────────────

# Sync Python dependencies
sync:
    uv sync --dev --all-extras

# Type-check Rust (fast, no linking)
check:
    cargo check

# Development install of the Python extension
develop:
    uv run maturin develop

# ── Testing ─────────────────────────────────────────────────────────

# Run Rust tests (excludes quspin-py which requires a Python interpreter)
test-rust:
    cargo test -p bitbasis -p quspin-core

# Run Python tests
test-python:
    uv run --locked pytest python/tests/ -v

# Run fast Python tests only (skip slow integration tests)
test-python-fast:
    uv run --locked pytest python/tests/ -v -m "not slow"

# Run all Python tests in parallel
test-python-parallel:
    uv run --locked pytest python/tests/ -v -n auto

# Run all tests
test: test-rust test-python

# ── Coverage ────────────────────────────────────────────────────────

coverage-run:
    uv run coverage run -m pytest python/tests/

coverage-xml: coverage-run
    uv run coverage xml

coverage-html: coverage-run
    uv run coverage html

coverage-report: coverage-run
    uv run coverage report

coverage-open: coverage-html
    open htmlcov/index.html

coverage: coverage-run coverage-xml coverage-report

# Run Rust tests with coverage and generate Cobertura XML
coverage-rust:
    cargo llvm-cov --cobertura --output-path rust-coverage.xml -p bitbasis -p quspin-core

# ── Clean ────────────────────────────────────────────────────────────

clean:
    cargo clean
    rm -rf dist/
