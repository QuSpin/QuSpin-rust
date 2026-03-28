# Orbit Computation: SIMD / Vectorisation Plan

## Background

The symmetric basis is built by calling `check_refstate` on every candidate
state.  For a basis of millions of states this is the hot loop.  The current
scalar implementation processes one state at a time and allocates a `Vec` per
state inside `iter_images`; both of these are obstacles to SIMD.

The functions in question live in
`crates/quspin-core/src/basis/symmetry/group/orbit.rs`.

---

## Two parallelism opportunities

### 1. Inter-state parallelism (the main win)

For `u32`/`u64` states, a SIMD register holds 8 or 4 states simultaneously
(AVX2 / AVX-512).  Applying the same lattice permutation to a batch of states
and then comparing them all at once is a natural fit for auto-vectorisation.

This is the primary target.

### 2. Intra-orbit parallelism (marginal)

A typical orbit has `lattice.len() × (1 + local.len())` images — often 8–64
entries.  This is too small for SIMD to be efficient on its own, and the
heap allocation inside `iter_images` would dominate anyway.  Not worth
pursuing directly; it becomes free once the batch path is in place.

---

## Step 1 — Eliminate the per-state heap allocation ✅ DONE

`iter_images` previously returned a `Vec<(B, Complex<f64>)>`.  This has been
replaced with `SmallVec<[(B, Complex<f64>); 64]>`, eliminating heap allocation
for orbits up to 64 images (covers 32-site translation × 1 local op).

**Implemented in:** `orbit.rs`

---

## Step 2 — Add batch variants of the orbit functions ✅ DONE

Batch-processing counterparts have been added alongside the existing scalar
functions.  The batch variants' inner loops are structured so that the compiler
can auto-vectorise across `states`.

### `check_refstate_batch`

```rust
pub(crate) fn check_refstate_batch<B, L>(
    lattice: &[LatticeElement],
    local: &[L],
    states: &[B],
    out: &mut [(B, f64)],
);
```

Key implementation details:
- Norm accumulated as `u32` (matches state register width) and cast to `f64`
  at the end — avoids the lane-width mismatch that halves SIMD throughput when
  using `f64` directly.
- `(s == *state) as u32` — explicit branchless form rather than relying on
  branch-to-cmov lowering.
- Representative updated with `o.0.max(s)` — guaranteed branchless
  (`vpmaxud`/`vpmaxuq` in SIMD).
- A separate `norms: Vec<u32>` buffer keeps the norm accumulation loop
  homogeneous with the state type, enabling clean auto-vectorisation of both
  the max-update and equality-count passes.

### `get_refstate_batch`

```rust
pub(crate) fn get_refstate_batch<B, L>(
    lattice: &[LatticeElement],
    local: &[L],
    states: &[B],
    out: &mut [(B, Complex<f64>)],
);
```

Tracks `(best: B, best_coeff: Complex<f64>)` per state.  The coefficient
select (`if cond { lat.grp_char } else { o.1 }`) involves `Complex<f64>` so
the inner loop does not auto-vectorise as cleanly as `check_refstate_batch`;
the primary benefit is **loop amortisation** — orbit elements are iterated
once per batch rather than once per state.

### Branchless updates

All representative updates use the select pattern:
```rust
let cond = s > best;
best = best.max(s);              // cmov / vpmaxud
best_coeff = if cond { c } else { best_coeff };  // fcsel / blendvpd
```

### Callers updated

- `sym.rs` (`SymmetricSubspace::build`): BFS candidates are collected per step
  and `check_refstate_batch` is called once per BFS step.
- `qmatrix/build.rs` (`build_from_symmetric`): `ham.apply` outputs are
  collected into a `row_buf` per row, then `get_refstate_batch` is called once
  per row.  Both buffers are allocated once outside the row loop and reused.

**Trait surface:** `SymGrp` exposes `get_refstate_batch` and
`check_refstate_batch` with scalar default implementations; `HardcoreSymmetryGrp`
overrides both with the amortised free functions.

---

## Step 3 — Benes permutation network

The `LatticeElement` currently uses `PermDitLocations` (a sequence of masked
shifts and ORs).  The planned replacement is a Benes network — a fixed
sequence of butterfly stages.

Each butterfly stage is:

```
for each adjacent pair (i, j):
    if swap_mask[i]:
        (state[i], state[j]) = (state[j], state[i])   // or the bit equivalent
```

This structure maps perfectly onto SIMD because:
- Every state in a batch goes through the *same* sequence of stages.
- Each stage is a uniform conditional-swap — one `vperm`, `vblend`, or bit-shift
  per stage across the whole batch.
- The number of stages is `O(log² N)` for N sites, so the total work per state
  is small.

When the Benes backend lands, `check_refstate_batch` will get the vectorisation
for free because `lat.apply_state` will itself be a short, regular sequence of
SIMD-friendly operations.

**No interface change is needed at the `orbit.rs` level** — the batch function
calls `lat.apply_state` in a loop; swapping the backend of `LatticeElement` is
sufficient.

---

## Step 4 — Explicit SIMD intrinsics (optional / future)

If auto-vectorisation proves insufficient, explicit SIMD can be introduced
behind a cfg flag:

```rust
#[cfg(target_feature = "avx2")]
fn check_refstate_batch_avx2(...) { ... }

#[cfg(not(target_feature = "avx2"))]
fn check_refstate_batch(...) { ... }  // scalar fallback
```

This is only worth doing once profiling shows the auto-vectorised batch
variant is leaving performance on the table.

---

## Summary of changes and their order

| Step | File(s) | Change | Status |
|------|---------|--------|--------|
| 1 | `orbit.rs` | Replace `Vec` with `SmallVec<[_; 64]>` in `iter_images` | ✅ Done |
| 2 | `orbit.rs`, `traits.rs`, `spin.rs`, `sym.rs`, `build.rs` | Batch variants + branchless updates + callers | ✅ Done |
| 3 | `bitbasis/transform.rs`, `LatticeElement` | Benes permutation backend | Pending |
| 4 | `orbit.rs` | Explicit SIMD intrinsics if needed | Pending (post-profiling) |

---

## Notes on large B types

For `B256`, `B512`, … a single state fills or exceeds one SIMD register.
Inter-state vectorisation does not apply to these.  For large-site systems the
performance path is different (sparse matrix × vector rather than explicit
orbit enumeration), so this is not a practical concern.
