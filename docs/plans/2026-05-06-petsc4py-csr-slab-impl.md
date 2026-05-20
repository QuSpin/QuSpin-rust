# petsc4py-compatible CSR slab construction — implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `PauliOperator.csr_slab(basis, coeffs, row_start, row_end, dtype, drop_zeros)` so a Python caller (e.g. one MPI rank under petsc4py) can materialise just rows `[row_start, row_end)` of a `PauliOperator + SpinBasis|FermionBasis` matrix as CSR, without ever building a global `QMatrix`.

**Architecture:** New Rust kernel `crates/quspin-matrix/src/csr_slab.rs`. A generic monomorphic core `csr_slab_kernel<H, B, C, S>` walks `[row_start, row_end)` only, calls `op.apply(state, |cindex, amp, new_state| …)`, projects via `basis.index`, sorts/merges per row with the existing `ZERO_TOL` drop-zeros rule, and emits `(Vec<i64>, Vec<i64>, Vec<Complex<f64>>)` directly — no `QMatrix` intermediate. Two type-erased dispatchers (`csr_slab_pauli_generic` for `GenericBasis`, `csr_slab_pauli_bit` for `BitBasis`) mirror the existing `QMatrixInner::build_hardcore{,_bit}` split. PyO3 layer adds a `csr_slab` method on `PyPauliOperator` that validates, dispatches on basis type, releases the GIL, and returns numpy arrays.

**Tech Stack:** Rust (workspace), `quspin-matrix`, `quspin-py` (PyO3 0.28 + numpy 0.28), pytest, scipy.sparse for test verification, petsc4py for the demo (not tested in CI).

**Spec:** `docs/superpowers/specs/2026-05-03-petsc4py-csr-slab-design.md` — already in this worktree.

**Branch:** `phil/petsc4py-csr-slab` (this worktree). Push as a single PR when done.

---

## Reference: existing code paths to mirror

Read these before starting — the slab path follows their dispatch shape exactly:

- `crates/quspin-matrix/src/qmatrix/build.rs` — `build_from_basis<H, B, M, I, C, S>` (the row-walking pattern), `build_from_bit<H, M, C>` (BitBasis dispatch), `build_from_space<H, M, C>` (GenericBasis fan-out into bit/dit).
- `crates/quspin-matrix/src/qmatrix/dispatch.rs:283-308` — `QMatrixInner::build_hardcore` and `build_hardcore_bit` (the type-erased entry points the slab dispatchers mirror).
- `crates/quspin-matrix/src/qmatrix/dispatch.rs:181-218` — `QMatrixInner::materialize` (the merge-and-drop-zeros pattern the slab kernel reuses inline).
- `crates/quspin-py/src/qmatrix.rs:64-83` — how `build_pauli` casts `&Bound<PyAny>` → `PySpinBasis` / `PyFermionBasis` and reaches `&b.borrow().inner.inner` (the `GenericBasis` / `BitBasis`).
- `crates/quspin-py/src/operator/pauli.rs:81-133` — `apply_and_project_to` and `apply` are existing methods on `PyPauliOperator`; the `csr_slab` method goes alongside them, same dispatch idiom.
- `crates/quspin-operator/src/lib.rs:30-42` — the `Operator<C>` trait with `apply<B: BitInt, F: FnMut(C, Complex<f64>, B)>`.

`ZERO_TOL = 4.0 * f64::EPSILON` lives in `crates/quspin-matrix/src/qmatrix/matrix.rs:17`. `PARALLEL_DIM_THRESHOLD` lives there too — used in `build.rs` to gate rayon.

---

## Task 1 — Scaffold the `csr_slab` module

**Files:**
- Create: `crates/quspin-matrix/src/csr_slab.rs`
- Modify: `crates/quspin-matrix/src/lib.rs` (add `pub mod csr_slab;` and re-export)

**Step 1: Create the empty module file.**

```rust
// crates/quspin-matrix/src/csr_slab.rs
//! Row-range CSR materialisation that bypasses `QMatrix`.
//!
//! Each call walks rows `[row_start, row_end)` of the operator + basis pair
//! directly, returning `(indptr, indices, data)` in the layout petsc4py's
//! `Mat.setValuesCSR(I, J, V)` expects.  Memory is bounded by the slab —
//! useful when each MPI rank only needs its locally-owned rows of a matrix
//! that's otherwise too large to materialise globally.
//!
//! See `docs/superpowers/specs/2026-05-03-petsc4py-csr-slab-design.md`.
```

**Step 2: Wire it into `lib.rs`.**

Open `crates/quspin-matrix/src/lib.rs`. After the existing `mod owned_qmatrix_op;` line, add:

```rust
pub mod csr_slab;
```

(Don't add anything to `pub use` yet — there's nothing to export until task 4.)

**Step 3: Verify it compiles.**

Run: `cargo check -p quspin-matrix`
Expected: `Finished … target(s)` with no errors.

**Step 4: Commit.**

```bash
git add crates/quspin-matrix/src/csr_slab.rs crates/quspin-matrix/src/lib.rs
git commit -m "feat(quspin-matrix): scaffold csr_slab module for petsc4py row-range CSR"
```

---

## Task 2 — Write the failing Rust test for `csr_slab_full_range_matches_to_csr`

The single most important invariant: `csr_slab(0, dim)` byte-for-byte equals `QMatrixInner::materialize(coeffs, drop_zeros)` on the same operator+basis.  We TDD this first.

**Files:**
- Create: `crates/quspin-matrix/tests/csr_slab.rs` (new integration-test file — kept here, not as `#[cfg(test)]` in `csr_slab.rs`, so we can build a `QMatrixInner` for the reference comparison)

**Step 1: Write the failing test.**

```rust
// crates/quspin-matrix/tests/csr_slab.rs
//! Equivalence tests for `csr_slab_pauli_*` against `QMatrixInner::materialize`.

use num_complex::Complex;
use quspin_basis::space::FullSpace;
use quspin_basis::dispatch::{BitBasis, GenericBasis};
use quspin_matrix::csr_slab::csr_slab_pauli_generic;
use quspin_matrix::qmatrix::QMatrixInner;
use quspin_operator::pauli::{HardcoreOp, HardcoreOperator, HardcoreOperatorInner, OpEntry};
use quspin_types::ValueDType;
use smallvec::smallvec;

/// 4-site XX + ZZ chain — small enough for fast tests, has 2 cindices.
fn xx_zz_op() -> HardcoreOperatorInner {
    let mut entries = Vec::new();
    for i in 0..3u32 {
        entries.push(OpEntry::new(
            0u8,
            Complex::new(1.0, 0.0),
            smallvec![(HardcoreOp::X, i), (HardcoreOp::X, i + 1)],
        ));
        entries.push(OpEntry::new(
            1u8,
            Complex::new(1.0, 0.0),
            smallvec![(HardcoreOp::Z, i), (HardcoreOp::Z, i + 1)],
        ));
    }
    HardcoreOperatorInner::Ham8(HardcoreOperator::new(entries))
}

fn full_4site_basis() -> GenericBasis {
    // SpinBasis::full(4) under the hood — build it directly here to keep the
    // test self-contained.  4 sites × LHSS=2 → BitBasis::Default(Full32).
    let space = FullSpace::<u32>::new(4, 2, false);
    GenericBasis::Bit(BitBasis::Default(
        quspin_basis::dispatch::BitBasisDefault::Full32(
            quspin_basis::space::FullSpace::<u32>::new(4, 2, false).into(),
        ),
    ))
    // NOTE: if the wrapping above doesn't match the actual variant constructors,
    // use the public SpinBasis API instead — see step 4 below.
}

#[test]
fn csr_slab_full_range_matches_materialize() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    let dim = 16; // 2^4
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];

    // Reference: build full QMatrix, materialize.
    let qm = QMatrixInner::build_hardcore(&op, &basis, ValueDType::Complex128);
    let (ref_indptr, ref_indices, ref_data) = qm.materialize(&coeffs, true).unwrap();

    // Slab: full range.
    let (slab_indptr, slab_indices, slab_data) =
        csr_slab_pauli_generic(&op, &basis, &coeffs, 0, dim, true).unwrap();

    assert_eq!(slab_indptr, ref_indptr);
    assert_eq!(slab_indices, ref_indices);
    assert_eq!(slab_data.len(), ref_data.len());
    for (a, b) in slab_data.iter().zip(ref_data.iter()) {
        assert!((a - b).norm() < 1e-12, "data mismatch: {a} vs {b}");
    }
}
```

**Step 2: Sanity-check that `full_4site_basis()` actually compiles** — the variant nesting is fragile.  If the helper above doesn't compile, replace its body with:

```rust
fn full_4site_basis() -> GenericBasis {
    use quspin_core::basis::{SpaceKind, SpinBasis};
    let basis = SpinBasis::new(4, 2, SpaceKind::Full).unwrap();
    basis.inner  // SpinBasis -> GenericBasis (per crates/quspin-basis/src/spin.rs:35)
}
```

…and add `quspin-core = { path = "../quspin-core" }` to the `[dev-dependencies]` of `crates/quspin-matrix/Cargo.toml`.

**Step 3: Run the test to verify it fails.**

```bash
cargo test -p quspin-matrix --test csr_slab
```

Expected: compile error like `unresolved import quspin_matrix::csr_slab::csr_slab_pauli_generic` (function doesn't exist yet).

**Step 4: Commit the failing test.**

```bash
git add crates/quspin-matrix/tests/csr_slab.rs crates/quspin-matrix/Cargo.toml
git commit -m "test(quspin-matrix): add failing csr_slab equivalence test (TDD)"
```

---

## Task 3 — Implement the generic kernel `csr_slab_kernel`

The monomorphic core that does the actual row walking + merging.  Generic over operator type `H`, bit-int `B`, cindex type `C`, and basis-space `S`.  The dispatch wrappers in task 4 will fan out into this.

**Files:**
- Modify: `crates/quspin-matrix/src/csr_slab.rs`

**Step 1: Add imports + the kernel.**

Replace the contents of `csr_slab.rs` with:

```rust
//! Row-range CSR materialisation that bypasses `QMatrix`.
//!
//! Each call walks rows `[row_start, row_end)` of the operator + basis pair
//! directly, returning `(indptr, indices, data)` in the layout petsc4py's
//! `Mat.setValuesCSR(I, J, V)` expects.

use num_complex::Complex;
use quspin_basis::BasisSpace;
use quspin_bitbasis::BitInt;
use quspin_operator::Operator;
use quspin_types::QuSpinError;
use rayon::prelude::*;

use crate::qmatrix::CIndex;
use crate::qmatrix::matrix::PARALLEL_DIM_THRESHOLD;

/// Drop-zeros tolerance (matches `QMatrix::to_csr_into` / `materialize`).
const ZERO_TOL: f64 = 4.0 * f64::EPSILON;

/// CSR slab return triple: `(indptr, indices, data)`.
///
/// - `indptr` length `row_end - row_start + 1`, zero-based, `indptr[0] == 0`,
///   `indptr[-1] == nnz_local`.
/// - `indices` length `nnz_local`, **global** column indices.
/// - `data` length `nnz_local`, accumulated `Σ_c coeffs[c] * stored_value`.
pub type CsrSlab = (Vec<i64>, Vec<i64>, Vec<Complex<f64>>);

/// Row-range CSR kernel — generic over operator/bitint/cindex/basis-space.
///
/// Validates `row_start <= row_end <= basis.size()` and
/// `coeffs.len() == op.num_cindices()`, then walks each row in parallel
/// (when `n_local >= PARALLEL_DIM_THRESHOLD`) and emits the merged CSR slab.
pub fn csr_slab_kernel<H, B, C, S>(
    op: &H,
    basis: &S,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    C: CIndex + Copy,
    S: BasisSpace<B> + Sync,
{
    if row_start > row_end {
        return Err(QuSpinError::ValueError(format!(
            "row_start ({row_start}) must be <= row_end ({row_end})"
        )));
    }
    let dim = basis.size();
    if row_end > dim {
        return Err(QuSpinError::ValueError(format!(
            "row_end ({row_end}) must be <= basis.size() ({dim})"
        )));
    }
    if coeffs.len() != op.num_cindices() {
        return Err(QuSpinError::ValueError(format!(
            "coeffs length {} must equal op.num_cindices() {}",
            coeffs.len(),
            op.num_cindices()
        )));
    }

    let n_local = row_end - row_start;

    // Per-row work: build (col, contribution) entries via op.apply, sort + merge.
    let process_row = |r: usize| -> Vec<(i64, Complex<f64>)> {
        let state = basis.state_at(r);
        let mut entries: Vec<(i64, Complex<f64>)> = Vec::new();
        op.apply(state, |cindex, amp, new_state| {
            if let Some(col) = basis.index(new_state) {
                entries.push((col as i64, amp * coeffs[cindex.as_usize()]));
            }
        });
        // Sort by col, merge same-col, drop_zeros.
        entries.sort_unstable_by_key(|&(c, _)| c);
        let mut merged: Vec<(i64, Complex<f64>)> = Vec::with_capacity(entries.len());
        let mut i = 0;
        while i < entries.len() {
            let col = entries[i].0;
            let mut acc = Complex::new(0.0, 0.0);
            let mut scale = 0.0f64;
            while i < entries.len() && entries[i].0 == col {
                let v = entries[i].1;
                acc += v;
                scale += v.norm();
                i += 1;
            }
            if !drop_zeros || acc.norm() > scale * ZERO_TOL {
                merged.push((col, acc));
            }
        }
        merged
    };

    let row_results: Vec<Vec<(i64, Complex<f64>)>> = if n_local >= PARALLEL_DIM_THRESHOLD {
        (row_start..row_end).into_par_iter().map(process_row).collect()
    } else {
        (row_start..row_end).map(process_row).collect()
    };

    // Flatten into CSR.
    let total_nnz: usize = row_results.iter().map(|r| r.len()).sum();
    let mut indptr: Vec<i64> = Vec::with_capacity(n_local + 1);
    let mut indices: Vec<i64> = Vec::with_capacity(total_nnz);
    let mut data: Vec<Complex<f64>> = Vec::with_capacity(total_nnz);
    indptr.push(0);
    for row in row_results {
        for (col, val) in row {
            indices.push(col);
            data.push(val);
        }
        indptr.push(indices.len() as i64);
    }
    Ok((indptr, indices, data))
}
```

**Step 2: Verify it compiles.**

```bash
cargo check -p quspin-matrix
```

If the import paths don't resolve (`PARALLEL_DIM_THRESHOLD`, `CIndex`), check the actual visibility:
- `PARALLEL_DIM_THRESHOLD` is `pub(crate)` in `qmatrix/matrix.rs:9` (or similar) — verify with `grep -n "PARALLEL_DIM_THRESHOLD" crates/quspin-matrix/src/qmatrix/matrix.rs`. If it's private, make it `pub(crate)`.
- `CIndex` is re-exported as `crate::qmatrix::CIndex` (look at `mod.rs`).

**Step 3: Commit.**

```bash
git add crates/quspin-matrix/src/csr_slab.rs crates/quspin-matrix/src/qmatrix/matrix.rs
git commit -m "feat(quspin-matrix): add generic csr_slab_kernel"
```

---

## Task 4 — Implement the `csr_slab_pauli_*` dispatchers

The kernel is generic.  Now wire up the type-erased dispatch (operator cindex width × basis variant) so callers can pass `&HardcoreOperatorInner` + `&GenericBasis`/`&BitBasis`.  The dispatch shape mirrors `build_from_space` / `build_from_bit` exactly — copy that pattern.

**Files:**
- Modify: `crates/quspin-matrix/src/csr_slab.rs`

**Step 1: Add the dispatchers below the kernel.**

```rust
// At the bottom of csr_slab.rs, after csr_slab_kernel:

use num_complex::Complex;
use quspin_basis::dispatch::{
    BitBasis, BitBasisDefault, DitBasis, DynDitBasis, DynDitBasisDefault, GenericBasis,
    QuatBasis, QuatBasisDefault, TritBasis, TritBasisDefault,
};
#[cfg(feature = "large-int")]
use quspin_basis::dispatch::{BitBasisLargeInt, DynDitBasisLargeInt, QuatBasisLargeInt, TritBasisLargeInt};
use quspin_basis::sym::{NormInt, SymBasis};
use quspin_bitbasis::{
    BitInt, DynamicPermDitValues, FermionicBitStateOp, PermDitMask, PermDitValues,
};
use quspin_operator::pauli::{HardcoreOperator, HardcoreOperatorInner};

type B128 = ruint::Uint<128, 2>;
type B256 = ruint::Uint<256, 4>;
#[cfg(feature = "large-int")]
type B512 = ruint::Uint<512, 8>;
// ... copy the same B1024..B8192 cfg-gated aliases from build.rs

/// Slab dispatcher for `PauliOperator` + `GenericBasis` (the `SpinBasis` path).
pub fn csr_slab_pauli_generic(
    op: &HardcoreOperatorInner,
    basis: &GenericBasis,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError> {
    match basis {
        GenericBasis::Bit(b) => csr_slab_pauli_bit(op, b, coeffs, row_start, row_end, drop_zeros),
        GenericBasis::Dit(b) => csr_slab_pauli_dit(op, b, coeffs, row_start, row_end, drop_zeros),
    }
}

/// Slab dispatcher for `PauliOperator` + `BitBasis` (the `FermionBasis` path
/// and the `BitBasis` arm of `GenericBasis`).
pub fn csr_slab_pauli_bit(
    op: &HardcoreOperatorInner,
    basis: &BitBasis,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError> {
    match op {
        HardcoreOperatorInner::Ham8(h) => csr_slab_pauli_bit_inner(h, basis, coeffs, row_start, row_end, drop_zeros),
        HardcoreOperatorInner::Ham16(h) => csr_slab_pauli_bit_inner(h, basis, coeffs, row_start, row_end, drop_zeros),
    }
}

/// Inner: dispatch on `BitBasis` variant after operator cindex width is fixed.
fn csr_slab_pauli_bit_inner<C>(
    op: &HardcoreOperator<C>,
    basis: &BitBasis,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError>
where
    C: CIndex + Copy + Ord,
{
    // Mirror the BitBasis match in build_from_bit (build.rs:208-250).
    // Each arm calls csr_slab_kernel::<HardcoreOperator<C>, B*, C, _>(op, b, ...).
    // The B* type is the BitInt that matches the variant.
    match basis {
        BitBasis::Default(d) => match d {
            BitBasisDefault::Full32(b) => csr_slab_kernel::<_, u32, C, _>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Full64(b) => csr_slab_kernel::<_, u64, C, _>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Full128(b) => csr_slab_kernel::<_, B128, C, _>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Full256(b) => csr_slab_kernel::<_, B256, C, _>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Sub32(b) => csr_slab_kernel::<_, u32, C, _>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Sub64(b) => csr_slab_kernel::<_, u64, C, _>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Sub128(b) => csr_slab_kernel::<_, B128, C, _>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Sub256(b) => csr_slab_kernel::<_, B256, C, _>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Sym32(b) => csr_slab_kernel_sym::<_, u32, PermDitMask<u32>, u8, C>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Sym64(b) => csr_slab_kernel_sym::<_, u64, PermDitMask<u64>, u16, C>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Sym128(b) => csr_slab_kernel_sym::<_, B128, PermDitMask<B128>, u32, C>(op, b, coeffs, row_start, row_end, drop_zeros),
            BitBasisDefault::Sym256(b) => csr_slab_kernel_sym::<_, B256, PermDitMask<B256>, u32, C>(op, b, coeffs, row_start, row_end, drop_zeros),
        },
        #[cfg(feature = "large-int")]
        BitBasis::LargeInt(d) => {
            // Mirror the LargeInt arms from build_from_bit — keep this
            // gated on the feature so non-feature builds don't try to
            // monomorphise the wide Uint variants.
            todo!("port from crates/quspin-matrix/src/qmatrix/build.rs:230-249")
        }
    }
}

/// Helper for the symmetric variants — `SymBasis<B, L, N>` needs three type params.
fn csr_slab_kernel_sym<H, B, L, N, C>(
    op: &H,
    basis: &SymBasis<B, L, N>,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    L: FermionicBitStateOp<B> + Sync,
    N: NormInt,
    C: CIndex + Copy,
{
    csr_slab_kernel::<H, B, C, _>(op, basis, coeffs, row_start, row_end, drop_zeros)
}

/// Inner: dispatch on `DitBasis` variant after operator cindex width is fixed.
fn csr_slab_pauli_dit<C>(
    op: &HardcoreOperatorInner,
    basis: &DitBasis,
    coeffs: &[Complex<f64>],
    row_start: usize,
    row_end: usize,
    drop_zeros: bool,
) -> Result<CsrSlab, QuSpinError>
where
    C: CIndex + Copy + Ord,
{
    // PauliOperator is hardcore (LHSS=2), so it can't be paired with a Dit basis.
    // build_from_dit also handles this implicitly via the `lhss` field on the basis;
    // for symmetry with build_from_dit, route through the kernel anyway — calls to
    // basis.state_at + index will work the same way.  The only practical case
    // would be lhss=2 SpinBasis whose internal repr happens to be DitBasis; this
    // is rare but possible for Symm bases under the symmetry-group constructor.
    todo!("port the DitBasis arms from build_from_dit (build.rs:253-363) — same shape as the BitBasis arms, with PermDitValues<3..16> in place of PermDitMask")
}
```

**Step 2: Verify the test compiles.**

```bash
cargo build -p quspin-matrix --tests
```

Expected: `error: not yet implemented` from the `todo!()`s on the `large-int` and `dit` paths — but `Default` arms compile.

**Step 3: Run the test.**

```bash
cargo test -p quspin-matrix --test csr_slab csr_slab_full_range_matches_materialize
```

Expected: PASS.  (The 4-site full SpinBasis is `BitBasis::Default::Full32` — the `dit` path is unreachable for this test.)

**Step 4: Commit.**

```bash
git add crates/quspin-matrix/src/csr_slab.rs
git commit -m "feat(quspin-matrix): add csr_slab_pauli_{generic,bit} dispatchers"
```

---

## Task 5 — Add the rest of the equivalence tests

Six more tests covering the design-doc test list.  All should pass against the kernel from task 3.

**Files:**
- Modify: `crates/quspin-matrix/tests/csr_slab.rs`

**Step 1: Add these tests below `csr_slab_full_range_matches_materialize`.**

```rust
#[test]
fn csr_slab_partition_concat_matches_materialize() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    let dim = 16;
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];

    let qm = QMatrixInner::build_hardcore(&op, &basis, ValueDType::Complex128);
    let (ref_indptr, ref_indices, ref_data) = qm.materialize(&coeffs, true).unwrap();

    // Try several partitions: 1 chunk (= full), 2, 3, dim chunks (one row each).
    for &k in &[1usize, 2, 3, dim] {
        // Even-ish split.
        let bounds: Vec<usize> = (0..=k).map(|i| i * dim / k).collect();
        let mut indptr_concat: Vec<i64> = vec![0];
        let mut indices_concat: Vec<i64> = Vec::new();
        let mut data_concat: Vec<Complex<f64>> = Vec::new();
        for w in bounds.windows(2) {
            let (rs, re) = (w[0], w[1]);
            let (ip, ii, dd) =
                csr_slab_pauli_generic(&op, &basis, &coeffs, rs, re, true).unwrap();
            // CSR-of-row-blocks merge: shift `ip` by current data length, drop ip[0].
            let off = indices_concat.len() as i64;
            for &p in &ip[1..] {
                indptr_concat.push(p + off);
            }
            indices_concat.extend_from_slice(&ii);
            data_concat.extend_from_slice(&dd);
        }
        assert_eq!(indptr_concat, ref_indptr, "k={k}: indptr");
        assert_eq!(indices_concat, ref_indices, "k={k}: indices");
        assert_eq!(data_concat.len(), ref_data.len(), "k={k}: nnz");
        for (a, b) in data_concat.iter().zip(ref_data.iter()) {
            assert!((a - b).norm() < 1e-12, "k={k}: data mismatch");
        }
    }
}

#[test]
fn csr_slab_empty_range_returns_empty_arrays() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];

    for r in [0usize, 5, 16] {
        let (ip, ii, dd) = csr_slab_pauli_generic(&op, &basis, &coeffs, r, r, true).unwrap();
        assert_eq!(ip, vec![0i64]);
        assert!(ii.is_empty());
        assert!(dd.is_empty());
    }
}

#[test]
fn csr_slab_drop_zeros_false_keeps_zeros() {
    // Pick coeffs that cause cancellation: XX with coeff 1.0 minus XX (via -1*XX).
    // 4-site H = XX + (-1)*XX  on 1 cindex pair → all-zero matrix.
    use smallvec::smallvec;
    let entries = vec![
        OpEntry::new(0u8, Complex::new(1.0, 0.0),
                     smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)]),
        OpEntry::new(1u8, Complex::new(-1.0, 0.0),
                     smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)]),
    ];
    let op = HardcoreOperatorInner::Ham8(HardcoreOperator::new(entries));
    let basis = full_4site_basis();
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];

    // drop_zeros=true: matrix collapses to nothing.
    let (_, ii_drop, dd_drop) = csr_slab_pauli_generic(&op, &basis, &coeffs, 0, 16, true).unwrap();
    assert!(ii_drop.is_empty());
    assert!(dd_drop.is_empty());

    // drop_zeros=false: every row that XX touches still emits the (cancelled) entry.
    let (_, ii_keep, _) = csr_slab_pauli_generic(&op, &basis, &coeffs, 0, 16, false).unwrap();
    assert!(!ii_keep.is_empty(), "drop_zeros=false should preserve cancelled entries");
}

#[test]
fn csr_slab_invalid_range_errors() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)];

    // row_start > row_end
    assert!(csr_slab_pauli_generic(&op, &basis, &coeffs, 5, 3, true).is_err());
    // row_end > basis.size()
    assert!(csr_slab_pauli_generic(&op, &basis, &coeffs, 0, 99, true).is_err());
}

#[test]
fn csr_slab_wrong_coeffs_len_errors() {
    let op = xx_zz_op();
    let basis = full_4site_basis();
    // op has 2 cindices; pass 3 coeffs.
    let coeffs = vec![Complex::new(1.0, 0.0); 3];
    assert!(csr_slab_pauli_generic(&op, &basis, &coeffs, 0, 16, true).is_err());
}
```

**Step 2: Run all tests in the file.**

```bash
cargo test -p quspin-matrix --test csr_slab
```

Expected: ALL PASS.  If the cancellation test fails (data length nonzero with `drop_zeros=true`), inspect the merge logic — the relative-tolerance check (`acc.norm() > scale * ZERO_TOL`) should eliminate `acc=0` when `scale > 0`.

**Step 3: Commit.**

```bash
git add crates/quspin-matrix/tests/csr_slab.rs
git commit -m "test(quspin-matrix): cover csr_slab partition concat, empty range, drop_zeros, error cases"
```

---

## Task 6 — Re-export `csr_slab_pauli_*` from `quspin-core`

Make the dispatchers reachable from `quspin-py` without a direct `quspin-matrix` dep.

**Files:**
- Modify: `crates/quspin-core/src/lib.rs`

**Step 1: Add the re-export.**

Find the existing `pub use quspin_matrix::{...};` line and add `csr_slab_pauli_generic, csr_slab_pauli_bit` to the import list — *or* add a new line:

```rust
pub use quspin_matrix::csr_slab::{csr_slab_pauli_bit, csr_slab_pauli_generic};
```

(There's no precedent for re-exporting from a submodule; check if the pattern used is `pub use quspin_matrix::*` — if so, the slab functions will already be reachable as `quspin_core::csr_slab_pauli_generic` via a wildcard.  Verify with `grep -n "pub use quspin_matrix" crates/quspin-core/src/lib.rs`.)

**Step 2: Verify it compiles.**

```bash
cargo check -p quspin-core
```

Expected: clean.

**Step 3: Commit.**

```bash
git add crates/quspin-core/src/lib.rs
git commit -m "feat(quspin-core): re-export csr_slab_pauli_{generic,bit}"
```

---

## Task 7 — Write failing Python tests for `PauliOperator.csr_slab`

We TDD the PyO3 binding.  These will all `AttributeError` until task 8.

**Files:**
- Create: `python/tests/operator/test_csr_slab.py`

**Step 1: Write the test file.**

```python
"""Tests for PauliOperator.csr_slab (petsc4py-compatible row-range CSR)."""

import numpy as np
import pytest
import scipy.sparse

from quspin_rs._rs import (
    FermionBasis,
    PauliOperator,
    QMatrix,
    SpinBasis,
)


N = 4


def make_xx_zz() -> PauliOperator:
    """4-site XX + ZZ chain — 2 cindices."""
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_full_spin_basis() -> SpinBasis:
    return SpinBasis.full(N)


def make_full_fermion_basis() -> FermionBasis:
    return FermionBasis.full(N)


def reference_csr(op, basis, coeffs):
    """Build the full QMatrix and materialise its CSR for comparison."""
    mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
    indptr, indices, data = mat.to_csr(coeffs)
    return scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(mat.dim, mat.dim)
    )


class TestCsrSlabFullRange:
    def test_full_range_matches_to_csr_spin(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)

        ref = reference_csr(op, basis, coeffs)
        indptr, indices, data = op.csr_slab(
            basis, coeffs, 0, basis.size, dtype=np.dtype("complex128")
        )
        slab = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(basis.size, basis.size)
        )
        np.testing.assert_array_equal((ref - slab).nnz, 0)
        np.testing.assert_allclose(ref.toarray(), slab.toarray(), atol=1e-12)

    def test_full_range_matches_to_csr_fermion(self):
        op = make_xx_zz()
        basis = make_full_fermion_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)

        ref = reference_csr(op, basis, coeffs)
        indptr, indices, data = op.csr_slab(
            basis, coeffs, 0, basis.size, dtype=np.dtype("complex128")
        )
        slab = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(basis.size, basis.size)
        )
        np.testing.assert_allclose(ref.toarray(), slab.toarray(), atol=1e-12)


class TestCsrSlabPartition:
    def test_partition_round_trip(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        dim = basis.size
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        ref = reference_csr(op, basis, coeffs)

        for k in (1, 2, 3, dim):
            bounds = [i * dim // k for i in range(k + 1)]
            indptr_parts = [np.array([0], dtype=np.int64)]
            indices_parts = []
            data_parts = []
            for rs, re in zip(bounds[:-1], bounds[1:]):
                ip, ii, dd = op.csr_slab(
                    basis, coeffs, rs, re, dtype=np.dtype("complex128")
                )
                # Drop the leading 0 of each subsequent indptr; offset by current nnz.
                off = sum(p.size for p in data_parts) if data_parts else 0
                # Wait: count nnz across all parts so far.
                cum_nnz = int(indptr_parts[-1][-1]) if indptr_parts else 0
                indptr_parts.append(ip[1:] + cum_nnz)
                indices_parts.append(ii)
                data_parts.append(dd)
            indptr = np.concatenate(indptr_parts)
            indices = np.concatenate(indices_parts) if indices_parts else np.zeros(0, dtype=np.int64)
            data = np.concatenate(data_parts) if data_parts else np.zeros(0, dtype=np.complex128)
            slab = scipy.sparse.csr_matrix((data, indices, indptr), shape=(dim, dim))
            np.testing.assert_allclose(ref.toarray(), slab.toarray(), atol=1e-12,
                                       err_msg=f"k={k}")


class TestCsrSlabEmpty:
    def test_empty_slab_returns_empty_arrays(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)

        for r in (0, 5, basis.size):
            ip, ii, dd = op.csr_slab(
                basis, coeffs, r, r, dtype=np.dtype("complex128")
            )
            assert ip.dtype == np.int64
            assert ii.dtype == np.int64
            assert dd.dtype == np.complex128
            np.testing.assert_array_equal(ip, np.array([0], dtype=np.int64))
            assert ii.size == 0
            assert dd.size == 0


class TestCsrSlabDtypes:
    def test_indptr_indices_int64(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        ip, ii, dd = op.csr_slab(
            basis, coeffs, 0, basis.size, dtype=np.dtype("complex128")
        )
        assert ip.dtype == np.int64
        assert ii.dtype == np.int64
        assert dd.dtype == np.complex128


class TestCsrSlabValidation:
    def test_basis_type_mismatch_raises(self):
        from quspin_rs._rs import BosonBasis

        op = make_xx_zz()
        basis = BosonBasis.full(2, lhss=2)  # wrong type for PauliOperator
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        with pytest.raises((TypeError, ValueError)):
            op.csr_slab(basis, coeffs, 0, basis.size, dtype=np.dtype("complex128"))

    def test_wrong_coeffs_size_raises(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j, 0.0 + 0j], dtype=np.complex128)
        with pytest.raises(ValueError):
            op.csr_slab(basis, coeffs, 0, basis.size, dtype=np.dtype("complex128"))

    def test_invalid_row_range_raises(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        # row_start > row_end
        with pytest.raises(ValueError):
            op.csr_slab(basis, coeffs, 10, 5, dtype=np.dtype("complex128"))
        # row_end > basis.size
        with pytest.raises(ValueError):
            op.csr_slab(
                basis, coeffs, 0, basis.size + 1, dtype=np.dtype("complex128")
            )
```

**Step 2: Run the tests to verify they fail.**

```bash
just develop  # rebuild extension (no-op if already built and no Rust changes since last build)
uv run pytest python/tests/operator/test_csr_slab.py -v
```

Expected: every test errors with `AttributeError: 'PauliOperator' object has no attribute 'csr_slab'`.

**Step 3: Commit the failing tests.**

```bash
git add python/tests/operator/test_csr_slab.py
git commit -m "test(py): add failing PauliOperator.csr_slab tests (TDD)"
```

---

## Task 8 — Implement `PauliOperator.csr_slab` PyO3 binding

**Files:**
- Modify: `crates/quspin-py/src/operator/pauli.rs`

**Step 1: Add the new method to `#[pymethods] impl PyPauliOperator`.**

Insert this method *between* `apply` and `__repr__` (around line 134):

```rust
    /// Materialise rows ``[row_start, row_end)`` as CSR without building a
    /// global QMatrix.  Designed for petsc4py distributed assembly: each
    /// MPI rank calls this with its locally-owned row range.
    ///
    /// Args:
    ///     basis:      ``SpinBasis`` or ``FermionBasis``.
    ///     coeffs:     1-D complex128 array of length ``num_cindices``.
    ///     row_start:  inclusive, 0-based.
    ///     row_end:    exclusive.
    ///     dtype:      output value dtype (default complex128).
    ///     drop_zeros: omit entries with ``|acc| <= scale * 4 * f64::EPSILON``.
    ///
    /// Returns:
    ///     ``(indptr, indices, data)`` as numpy arrays.  ``indptr`` length
    ///     ``row_end - row_start + 1``, indices are **global** column
    ///     indices (zero-based).  Both indptr and indices are int64; the
    ///     petsc4py user does ``arr.astype(PETSc.IntType, copy=False)``.
    #[pyo3(signature = (basis, coeffs, row_start, row_end, dtype, drop_zeros = true))]
    #[allow(clippy::too_many_arguments)]
    fn csr_slab<'py>(
        &self,
        py: Python<'py>,
        basis: &Bound<'py, PyAny>,
        coeffs: numpy::PyReadonlyArray1<'py, numpy::Complex64>,
        row_start: usize,
        row_end: usize,
        dtype: &Bound<'py, numpy::PyArrayDescr>,
        drop_zeros: bool,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<i64>>,
        Bound<'py, numpy::PyArray1<i64>>,
        Bound<'py, pyo3::types::PyAny>,
    )> {
        use crate::basis::fermion::PyFermionBasis;
        use crate::basis::spin::PySpinBasis;
        use num_complex::Complex;
        use numpy::{PyArray1, ToPyArray};
        use quspin_core::{csr_slab_pauli_bit, csr_slab_pauli_generic};

        // Convert coeffs (numpy::Complex64 -> num_complex::Complex<f64>; same memory layout).
        let coeffs_vec: Vec<Complex<f64>> = coeffs
            .as_array()
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect();

        // Dispatch on basis type.  Validate is_built first.
        let (indptr, indices, data) = if let Ok(b) = basis.cast::<PySpinBasis>() {
            let inner = &b.borrow().inner;
            if !inner.inner.is_built() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "basis must be built (call .full / .subspace / .symmetric first)",
                ));
            }
            // Snapshot the &GenericBasis so the kernel can run inside py.detach.
            // We can't carry the PyRef across detach, so clone the inner enum (cheap).
            let basis_inner = inner.inner.clone();
            let op_inner = self.inner.clone();
            py.detach(move || {
                csr_slab_pauli_generic(
                    &op_inner, &basis_inner, &coeffs_vec, row_start, row_end, drop_zeros,
                )
            })
            .map_err(crate::error::Error::from)?
        } else if let Ok(b) = basis.cast::<PyFermionBasis>() {
            let inner = &b.borrow().inner;
            if !inner.inner.is_built() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "basis must be built (call .full / .subspace / .symmetric first)",
                ));
            }
            let basis_inner = inner.inner.clone();
            let op_inner = self.inner.clone();
            py.detach(move || {
                csr_slab_pauli_bit(
                    &op_inner, &basis_inner, &coeffs_vec, row_start, row_end, drop_zeros,
                )
            })
            .map_err(crate::error::Error::from)?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "basis must be SpinBasis or FermionBasis for PauliOperator.csr_slab",
            ));
        };

        // Convert to numpy arrays.
        let indptr_py = indptr.to_pyarray(py);
        let indices_py = indices.to_pyarray(py);

        // Output dtype: build a complex128 numpy array and `astype` if user
        // asked for something else.  For complex128 (the common case) this
        // is a no-op.
        let data_c128: Bound<'py, PyArray1<numpy::Complex64>> = data.to_pyarray(py);
        let data_out: Bound<'py, pyo3::types::PyAny> = if dtype.is_equiv_to(
            &numpy::dtype::<numpy::Complex64>(py),
        ) {
            data_c128.into_any()
        } else {
            data_c128.call_method1("astype", (dtype.clone(),))?
        };

        Ok((indptr_py, indices_py, data_out))
    }
```

A couple of details that may need adjustment when you actually wire it up:

- `HardcoreOperatorInner` may or may not derive `Clone`. Check `crates/quspin-operator/src/pauli/operator.rs`. If it doesn't, either `#[derive(Clone)]` it (cheap — the inner `Vec<OpEntry>` is Clone), or wrap the operator in `Arc::clone(&self.inner)` if `PyPauliOperator.inner` is already `Arc`-wrapped. (It isn't right now — see line 27. So derive Clone or restructure.)
- `GenericBasis` / `BitBasis` Clone: same check. They likely derive Clone since they're plain enum variants holding cloneable data; if not, derive it.
- `numpy::Complex64` ABI-equals `num_complex::Complex<f64>` per [numpy 0.28 dtype.rs](https://docs.rs/numpy/0.28.0/numpy/index.html). The conversion above is sound.
- `PyArray1::call_method1("astype", …)` returns `Bound<PyAny>`. The `into_any()` coercion from a `Bound<PyArray1<…>>` is straightforward. If it doesn't compile, `.as_any().clone()` works equivalently.

**Step 2: Build the extension.**

```bash
just develop
```

Expected: clean build.

**Step 3: Run the Python tests.**

```bash
uv run pytest python/tests/operator/test_csr_slab.py -v
```

Expected: ALL PASS.

**Step 4: Run clippy on the touched crates.**

```bash
cargo clippy -p quspin-matrix -p quspin-core -p quspin-py --all-targets -- -D warnings
```

Expected: clean.  If a `clippy::too_many_arguments` warning lands, add `#[allow(clippy::too_many_arguments)]` on the method (already in the snippet above).

**Step 5: Commit.**

```bash
git add crates/quspin-py/src/operator/pauli.rs
git commit -m "feat(quspin-py): add PauliOperator.csr_slab method"
```

---

## Task 9 — Update `_rs.pyi` type stub

**Files:**
- Modify: `python/quspin_rs/_rs.pyi`

**Step 1: Find the `PauliOperator` stub class** (`grep -n "class PauliOperator" python/quspin_rs/_rs.pyi`).

**Step 2: Add the `csr_slab` method signature** alongside the existing `apply` / `apply_and_project_to`:

```python
    def csr_slab(
        self,
        basis: SpinBasis | FermionBasis,
        coeffs: npt.NDArray[Any],
        row_start: int,
        row_end: int,
        dtype: np.dtype[Any],
        drop_zeros: bool = True,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[Any],
    ]:
        """Materialise rows ``[row_start, row_end)`` as CSR.

        Designed for petsc4py: each MPI rank calls this with its locally-owned
        row range.  Returns ``(indptr, indices, data)`` where ``indices`` are
        **global** column indices.  See the petsc4py demo at
        ``examples/petsc4py_chunked_build.py``.
        """
        ...
```

**Step 3: Verify pyright is happy.**

```bash
uv run pyright python/
```

Expected: no new errors.

**Step 4: Commit.**

```bash
git add python/quspin_rs/_rs.pyi
git commit -m "docs(py): add PauliOperator.csr_slab type stub"
```

---

## Task 10 — Add the petsc4py demo + README

**Files:**
- Create: `examples/petsc4py_chunked_build.py`
- Create: `examples/README.md`

**Step 1: Write the demo.**

```python
"""petsc4py-compatible chunked matrix construction (issue #69).

Run serial:    python examples/petsc4py_chunked_build.py
Run distributed:    mpirun -n 4 python examples/petsc4py_chunked_build.py

Builds the same Hamiltonian two ways:
  (a) Distributed via PauliOperator.csr_slab — each MPI rank computes only
      its locally-owned rows and pushes them into a PETSc Mat.
  (b) Reference (rank 0 only) via QMatrix.build_pauli + to_csr — the full
      global matrix.

Compares them dense-form on rank 0 with assert_allclose.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

try:
    from mpi4py import MPI
    from petsc4py import PETSc
except ImportError as exc:
    raise SystemExit(
        "This example requires petsc4py and mpi4py:\n"
        "    pip install petsc4py mpi4py"
    ) from exc

from quspin_rs._rs import PauliOperator, QMatrix, SpinBasis


def build_distributed(n_sites: int) -> tuple[PETSc.Mat, int]:
    op = PauliOperator(
        [("XX", [[1.0, i, i + 1] for i in range(n_sites - 1)])],
        [("ZZ", [[1.0, i, i + 1] for i in range(n_sites - 1)])],
    )
    basis = SpinBasis.full(n_sites)

    mat = PETSc.Mat().create(comm=MPI.COMM_WORLD)
    mat.setSizes(((PETSc.DECIDE, basis.size), (basis.size, basis.size)))
    mat.setType(PETSc.Mat.Type.AIJ)
    mat.setUp()
    rstart, rend = mat.getOwnershipRange()

    coeffs = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    indptr, indices, data = op.csr_slab(
        basis, coeffs, rstart, rend, dtype=np.dtype("complex128")
    )
    mat.setValuesCSR(
        indptr.astype(PETSc.IntType, copy=False),
        indices.astype(PETSc.IntType, copy=False),
        data,
    )
    mat.assemble()
    return mat, basis.size


def build_reference(n_sites: int) -> np.ndarray:
    """Rank-0 reference matrix as a dense numpy array."""
    op = PauliOperator(
        [("XX", [[1.0, i, i + 1] for i in range(n_sites - 1)])],
        [("ZZ", [[1.0, i, i + 1] for i in range(n_sites - 1)])],
    )
    basis = SpinBasis.full(n_sites)
    qm = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
    coeffs = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    indptr, indices, data = qm.to_csr(coeffs)
    return scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(basis.size, basis.size)
    ).toarray()


def main() -> None:
    n_sites = 8
    rank = MPI.COMM_WORLD.Get_rank()

    dist_mat, dim = build_distributed(n_sites)

    # Pull the distributed matrix to dense on rank 0 for comparison.
    dist_dense = dist_mat.convert(PETSc.Mat.Type.DENSE)
    arr = dist_dense.getDenseArray() if rank == 0 else None
    # MatType DENSE is also distributed; gather to rank 0:
    if rank == 0:
        local_rows, _ = dist_dense.getOwnershipRange()
        # For a small example: use SCATTER_FORWARD into a sequential dense mat.
        scatter_dense = PETSc.Mat().createDense((dim, dim), comm=MPI.COMM_SELF)
        scatter_dense.setUp()
    # The simplest robust gather: convert to CSR, materialise, broadcast.
    dist_csr = dist_mat.convert(PETSc.Mat.Type.AIJ)  # already AIJ but ensures format
    # Use Mat.getValuesCSR on each rank, gather to rank 0.  For an example demo,
    # we leverage scipy after gather:
    indptr_local, indices_local, data_local = dist_mat.getValuesCSR()
    rstart, rend = dist_mat.getOwnershipRange()
    n_local = rend - rstart
    # Gather each rank's CSR to rank 0.
    comm = MPI.COMM_WORLD
    all_indptr = comm.gather((indptr_local, n_local, rstart), root=0)
    all_indices = comm.gather(indices_local, root=0)
    all_data = comm.gather(data_local, root=0)

    if rank == 0:
        # Stitch into a single global CSR.
        global_indptr = np.zeros(dim + 1, dtype=np.int64)
        global_indices_parts: list[np.ndarray] = []
        global_data_parts: list[np.ndarray] = []
        cum_nnz = 0
        for (ip_loc, n_loc, rs), ii_loc, dd_loc in zip(all_indptr, all_indices, all_data):
            for i in range(n_loc):
                global_indptr[rs + i + 1] = cum_nnz + ip_loc[i + 1] - ip_loc[0]
            global_indices_parts.append(np.asarray(ii_loc, dtype=np.int64))
            global_data_parts.append(np.asarray(dd_loc, dtype=np.complex128))
            cum_nnz += int(ip_loc[-1] - ip_loc[0])
        global_indices = np.concatenate(global_indices_parts)
        global_data = np.concatenate(global_data_parts)
        dist_dense_arr = scipy.sparse.csr_matrix(
            (global_data, global_indices, global_indptr), shape=(dim, dim)
        ).toarray()
        ref = build_reference(n_sites)
        np.testing.assert_allclose(dist_dense_arr, ref, atol=1e-12)
        print(f"OK: distributed and reference matrices match (dim={dim}).")


if __name__ == "__main__":
    main()
```

(The gather-and-compare code is a little involved because petsc4py's dense-gather story is awkward; the CSR-gather approach above is more portable.)

**Step 2: Write the README.**

```markdown
# Examples

Manually-runnable demos that aren't part of CI.

## petsc4py_chunked_build.py

Demonstrates using `PauliOperator.csr_slab` to build a distributed PETSc
matrix without ever materialising the full QuSpin matrix on any single
rank.  Compares the distributed assembly against `QMatrix.build_pauli +
to_csr` on rank 0.

### Requirements

```sh
pip install petsc4py mpi4py
```

(petsc4py needs a working PETSc install — `brew install petsc` or
`apt-get install petsc-dev` typically suffice.)

### Running

```sh
# Single-rank serial check:
python examples/petsc4py_chunked_build.py

# Distributed run:
mpirun -n 4 python examples/petsc4py_chunked_build.py
```

Both should print `OK: distributed and reference matrices match (dim=256).`
```

**Step 3: Commit.**

```bash
git add examples/petsc4py_chunked_build.py examples/README.md
git commit -m "docs(examples): add petsc4py distributed assembly demo"
```

---

## Task 11 — Final verification + push

**Step 1: Run the full Rust test suite.**

```bash
cargo test --workspace --exclude quspin-py
```

Expected: all green.

**Step 2: Run the full Python test suite.**

```bash
just develop
uv run pytest python/tests/ -m "not slow"
```

Expected: all green.  Test count should be exactly the previous baseline + the count in `test_csr_slab.py`.

**Step 3: Workspace clippy + fmt.**

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Expected: clean.

**Step 4: Push the branch.**

```bash
git push -u origin phil/petsc4py-csr-slab
```

**Step 5: Open the PR.**

```bash
gh pr create --title "feat(quspin-py): petsc4py-compatible csr_slab on PauliOperator (closes #69)" --body "$(cat <<'EOF'
## Summary

Implements [issue #69](https://github.com/QuSpin/QuSpin-rust/issues/69): `PauliOperator.csr_slab(basis, coeffs, row_start, row_end, dtype)` materialises just rows `[row_start, row_end)` of a `PauliOperator + SpinBasis | FermionBasis` matrix as CSR, without ever building a global `QMatrix`.  Designed for petsc4py distributed assembly: each MPI rank calls this with its locally-owned row range and feeds the result into `Mat.setValuesCSR`.

V1 scope is `PauliOperator + SpinBasis/FermionBasis` only — proof of concept.  Other operator/basis families follow the same kernel pattern in mechanical follow-up PRs.

## Design

Spec: `docs/superpowers/specs/2026-05-03-petsc4py-csr-slab-design.md` (already on main).

- New module `crates/quspin-matrix/src/csr_slab.rs` with a generic kernel `csr_slab_kernel<H, B, C, S>` and two type-erased dispatchers (`csr_slab_pauli_generic`, `csr_slab_pauli_bit`) mirroring `QMatrixInner::build_hardcore{,_bit}`.
- Each rank holds only `(rend - rstart)` rows of CSR plus a small per-row scratch — memory bounded by the slab regardless of global dim.
- Output `indices` are **global** column indices (matches petsc4py's `setValuesCSR` contract).
- Index dtype fixed at `int64`; the petsc4py user does `arr.astype(PETSc.IntType, copy=False)` (no-op on int64 PETSc builds).

## Test plan

- [x] `cargo test --workspace --exclude quspin-py` — passes (incl. 6 new tests in `crates/quspin-matrix/tests/csr_slab.rs` covering full-range equivalence, partition-concat equivalence, empty range, drop_zeros cancellation, and error cases).
- [x] `uv run pytest python/tests/ -m "not slow"` — passes (incl. new `python/tests/operator/test_csr_slab.py`).
- [x] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [x] `cargo fmt --check` clean.
- [x] `_rs.pyi` updated.
- [x] petsc4py demo at `examples/petsc4py_chunked_build.py` plus a short `examples/README.md`.  Not part of CI (no PETSc install in CI); compares distributed assembly to `QMatrix.to_csr` on rank 0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes for the executor

- **Drop-zeros tolerance** in `csr_slab_kernel` matches `QMatrix::to_csr_into` (relative: `acc.norm() > scale * ZERO_TOL` where `scale` is the sum of contribution magnitudes).  Don't switch to an absolute threshold — equivalence with `materialize` depends on identical semantics here.
- **Parallelism:** the kernel uses `into_par_iter` when `n_local >= PARALLEL_DIM_THRESHOLD`.  Inside `py.detach` (GIL released), this is safe.  Don't try to parallelise across slabs at the Python level — that's the petsc4py user's job.
- **`large-int` feature:** the dispatcher in task 4 has a `todo!()` for the `large-int` arms.  Port them mechanically from `crates/quspin-matrix/src/qmatrix/build.rs:230-249` before opening the PR — CI runs a separate `cargo check --features large-int` pass that will fail if this is left as a `todo!()`.  Same for the `dit` arms (`build.rs:253-363`).
- **Empty op (no terms):** `op.num_cindices() == 0` — `coeffs` must be empty too.  The kernel handles this (the inner `op.apply` callback never fires; merged is empty).  No special-case needed.
- **Subspace bases:** `basis.index(state)` returns `None` for states outside the subspace — kernel skips, matching `QMatrix::build`.  Test added in task 5 isn't strictly needed for v1 but is cheap to add if the executor wants belt-and-braces; otherwise the partition-concat test exercises every code path that matters.
