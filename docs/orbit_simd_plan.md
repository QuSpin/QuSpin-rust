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

## Step 1 — Eliminate the per-state heap allocation

`iter_images` currently returns a `Vec<(B, Complex<f64>)>`.  The orbit size
is fixed for a given group (it does not depend on the state), so this
allocation can be removed.

**Target signature:**

```rust
// Orbit size = lattice.len() * (1 + local.len()), computed once at group
// construction and stored on the group.
fn iter_images<B, L, const N: usize>(
    lattice: &[LatticeElement],
    local: &[L],
    state: B,
    out: &mut [MaybeUninit<(B, Complex<f64>)>; N],
)
```

Or more pragmatically, use `smallvec::SmallVec<[(B, Complex<f64>); 16]>` as a
short-term fix.  A capacity of 16 covers the common cases (4-site translation
× 1 flip = 8; 8-site translation × 1 flip = 16) without any heap allocation.

**Files to change:**
- `orbit.rs` — replace `Vec` with `SmallVec`
- `spin.rs` / `dit.rs` — no interface change needed; callers use
  `get_refstate` / `check_refstate` which are unaffected

---

## Step 2 — Add batch variants of the orbit functions

Introduce batch-processing counterparts alongside the existing scalar ones:

```rust
/// Scalar (existing).
pub(crate) fn check_refstate<B, L>(
    lattice: &[LatticeElement],
    local: &[L],
    state: B,
) -> (B, f64);

/// Batch: process `states.len()` states in one call.
/// Writes `(representative, norm)` for each input state into `out`.
/// The slices must have equal length.
pub(crate) fn check_refstate_batch<B, L>(
    lattice: &[LatticeElement],
    local: &[L],
    states: &[B],
    out: &mut [(B, f64)],
);
```

The batch variant's inner loop is structured so that the compiler can
auto-vectorise across `states`:

```rust
// Pseudocode — one lattice element applied to a whole batch of states.
for (state, result) in states.iter().zip(out.iter_mut()) {
    let s = lat.apply(*state);   // same operation on every element
    if s > result.0 { result.0 = s; }
    if s == *state  { result.1 += 1.0; }
}
```

Because each iteration is independent and the operation is uniform, LLVM will
emit SIMD code for `u32`/`u64` batches when compiled with `-C target-cpu=native`
or with explicit target features.

**Callers** — the symmetric subspace builder in `sym.rs` iterates over all
candidate states; replace the per-state call with a batch call over chunks:

```rust
const BATCH: usize = 256;
for chunk in candidates.chunks(BATCH) {
    check_refstate_batch(&lattice, &local, chunk, &mut scratch[..chunk.len()]);
    // process scratch
}
```

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

When the Benes backend lands, the batch `check_refstate_batch` will get the
vectorisation for free because `lat.apply` will itself be a short, regular
sequence of SIMD-friendly operations.

**No interface change is needed at the `orbit.rs` level** — the batch function
calls `lat.apply` in a loop; swapping the backend of `LatticeElement::apply`
is sufficient.

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

| Step | File(s) | Change | Prerequisite |
|------|---------|--------|--------------|
| 1 | `orbit.rs` | Replace `Vec` with `SmallVec` in `iter_images` | None |
| 2 | `orbit.rs`, `sym.rs` | Add `*_batch` variants; update basis builder | Step 1 |
| 3 | `bitbasis/transform.rs`, `LatticeElement` | Benes permutation backend | Step 2 |
| 4 | `orbit.rs` | Explicit SIMD intrinsics if needed | Step 3 + profiling |

Step 1 is a safe, self-contained improvement that can land any time.
Steps 2–4 should wait until the Benes network is designed.

---

## Notes on large B types

For `B256`, `B512`, … a single state fills or exceeds one SIMD register.
Inter-state vectorisation does not apply to these.  For large-site systems the
performance path is different (sparse matrix × vector rather than explicit
orbit enumeration), so this is not a practical concern.
