# Symmetry-group API: explicit group elements with composite lattice+local support

**Date:** 2026-04-19
**Status:** Approved
**Relates to:** Post-StateTransitions follow-up (see spec §8 of `2026-04-19-state-graph-decoupling.md`); QuSpin/QuSpin#390 (particle-hole sign convention).

---

## 1. Motivation

The current symmetry-group API on `SymBasis` takes lattice (site-permutation) elements and local (per-site) elements through two independent methods:

```rust
basis.add_lattice(char, perm);
basis.add_local(char, local_op);
```

and the orbit walker iterates the **cartesian product** `⟨lattice⟩ × ⟨local⟩`. That produces a group `|lattice| × |local|` in size.

This cannot express diagonal symmetries. A concrete example: a system whose only spatial symmetry is `P·Z` (site reflection composed with global spin inversion), where `P` and `Z` individually are **not** symmetries of the Hamiltonian. The group is `{I, PZ}` of order 2, not `{I, P, Z, PZ}` of order 4. The current API forces the latter and rejects valid states from the sector.

Staggered / shift-anti-translation symmetries have the same shape. `⟨T·P⟩` at length `L=8` has 2L = 16 elements along a single diagonal; `⟨T, P⟩` has 2L × 2 = 32.

The refactor makes group elements explicit: the user enumerates each element as a `(character, lattice_perm?, local_op?)` triple. No cartesian product. No implicit generator closure. The user is responsible for listing the full group; the builder validates closure and character consistency at `build()` time.

**Primary goal:** support combined lattice+local symmetry generators with correct group closure.
**Secondary goal:** make character assignment explicit (generators leave `χ(gh)` ambiguous; explicit elements pair each `χ` with its group element directly).
**Non-goal:** any change to the BFS or matrix-build machinery.

---

## 2. Current structure

### 2.1 API

```rust
impl SymBasis<B, L, N> {
    pub fn add_lattice(&mut self, char: Complex<f64>, perm: &[usize]);
    pub fn add_local(&mut self, char: Complex<f64>, local_op: L);
}
```

### 2.2 Storage

```rust
pub struct SymBasis<B, L, N> {
    pub(crate) lattice: Vec<BenesLatticeElement<B>>,
    pub(crate) local: Vec<(Complex<f64>, L)>,
    // …
}
```

### 2.3 Orbit walker

Today in `orbit.rs`:

```rust
for lat in &basis.lattice {
    for (chi, loc) in &basis.local {
        // apply lat + loc to state, update running char, track rep
    }
}
```

The cartesian product is hard-coded. Whatever the user adds is enumerated with every combination, producing the full `⟨lattice⟩ × ⟨local⟩`.

---

## 3. Target design

### 3.1 Public API

One method, taking a typed element:

```rust
impl SymBasis<B, L, N> {
    /// Add an explicit group element with its representation character.
    ///
    /// The user enumerates every non-identity element of the group (the
    /// identity is implicit, `χ(I) = 1`). The builder validates closure
    /// and character consistency at `build()` time; silently incorrect
    /// inputs are rejected, not tolerated.
    pub fn add_symmetry(
        &mut self,
        char: Complex<f64>,
        element: SymElement<L>,
    ) -> Result<(), QuSpinError>;

    /// Convenience: add the non-identity powers of a cyclic generator
    /// `g^1, g^2, …, g^{order-1}` with characters `char(k)` for k = 1..order.
    ///
    /// Pure sugar over `add_symmetry` — no special code path in the walker.
    pub fn add_cyclic(
        &mut self,
        generator: SymElement<L>,
        order: usize,
        char: impl Fn(usize) -> Complex<f64>,
    ) -> Result<(), QuSpinError>;
}
```

The old `add_lattice`, `add_local`, and `add_inv` are **removed**. Callers use `SymElement::lattice(perm)`, `SymElement::local(op)`, or `SymElement::composite(perm, op)` and pass the result into `add_symmetry`.

### 3.2 `SymElement<L>` type

The user-facing element type is a struct with two optional components:

```rust
// in quspin-basis or quspin-bitbasis — TBD at implementation
pub struct SymElement<L> {
    perm: Option<BenesPermDitLocations<B>>,
    local: Option<L>,
}

impl<L> SymElement<L> {
    /// Pure-lattice element: site permutation, no local transform.
    pub fn lattice(perm: &[usize]) -> Self;

    /// Pure-local element: local transform, no site permutation.
    pub fn local(op: L) -> Self;

    /// Composite element: site permutation composed with local transform.
    /// (Lattice and local components commute — they act on orthogonal
    /// degrees of freedom — so "site perm first, then local" and the
    /// reverse are equivalent.)
    pub fn composite(perm: &[usize], op: L) -> Self;

    /// Composition: `a.compose(b)` returns the group element representing
    /// `a · b`. Used internally by `add_cyclic` to compute powers of a
    /// generator.
    pub fn compose(&self, other: &Self) -> Self
    where
        L: Compose;  // trait method on L: Compose, with blanket impls for our Ls
}
```

The struct fields are `pub(crate)` (or private); the three constructors are the only way to build a `SymElement`, which rules out the illegal `(None, None)` state at the type level.

`L: Compose` is a helper trait (perm-like composition: `compose(self, other) -> Self`) implemented for `PermDitMask<B>`, `PermDitValues<N>`, `DynamicPermDitValues`, and `SignedPermDitMask<B>` (see §3.4).

### 3.3 Storage: three typed vectors

Inside `SymBasis`, each added element is dispatched *once at insert time* into one of three vectors based on which fields are `Some`:

```rust
pub struct SymBasis<B, L, N> {
    /// Pure lattice elements: only a Benes permutation, no local transform.
    pub(crate) lattice_only: Vec<(Complex<f64>, BenesLatticeElement<B>)>,

    /// Pure local elements: only a local transform, no permutation.
    pub(crate) local_only: Vec<(Complex<f64>, L)>,

    /// Composite elements: both a permutation and a local transform,
    /// applied atomically as one group element.
    pub(crate) composite: Vec<(Complex<f64>, BenesLatticeElement<B>, L)>,

    // … existing fields (states, index_map, lhss, fermionic, …) unchanged
}
```

The orbit walker iterates three homogeneous loops:

```rust
for (chi, perm) in &self.lattice_only {
    let s = perm.apply(state);
    // update character, check rep
}
for (chi, loc) in &self.local_only {
    let s = loc.apply(state);
    // update character, check rep
}
for (chi, perm, loc) in &self.composite {
    let s_perm = perm.apply(state);
    let s = loc.apply(s_perm);
    // update character, check rep
}
```

Each loop is type-homogeneous. The per-element "does this have a perm? does it have a local?" branch that a single-vector design would require is gone — the branch is done once, at insert time.

**Critical invariant:** the three-vector decomposition breaks the cartesian-product assumption of the current walker. What the walker iterates is exactly the set of elements the user added, plus the implicit identity. The always-on validation (§3.5) ensures that set is actually a closed group — without the validator, a malformed input (missing elements, inconsistent characters) would silently corrupt the sector.

### 3.4 Fermion signs: particle-hole and friends

Some local operators carry state-dependent fermion signs. The canonical case is the particle-hole transformation `C` on a fermionic basis: `|n_i⟩ → |1 − n_i⟩` at every site, with a sign that depends on the occupied-site set and a per-site "ph_sign" array. See QuSpin/QuSpin#390 for the Python-side history: for 1D the sign is `(−1)^i`, for higher-dim or non-uniform lattices it isn't, so the user must be able to supply the per-site array.

The abstraction:

```rust
// New trait extending BitStateOp<B>. Default sign is 1.0; sign-carrying
// ops override.
pub trait FermionicBitStateOp<B: BitInt>: BitStateOp<B> {
    /// Fermion sign contributed by this local op when applied to `state`.
    /// Default: 1.0 (no fermion sign).
    fn fermion_sign(&self, _state: B) -> f64 {
        1.0
    }
}

// Blanket impl for the existing (sign-free) types:
impl<B: BitInt> FermionicBitStateOp<B> for PermDitMask<B> {}
impl<B: BitInt, const N: usize> FermionicBitStateOp<B> for PermDitValues<N> {}
impl<B: BitInt> FermionicBitStateOp<B> for DynamicPermDitValues {}

// New type for particle-hole:
pub struct SignedPermDitMask<B> {
    mask: B,
    /// Per-site sign; length = n_sites. Default for 1D: (−1)^i.
    sign: Vec<f64>,
}

impl<B: BitInt> BitStateOp<B> for SignedPermDitMask<B> {
    fn apply(&self, state: B) -> B {
        state ^ self.mask
    }
}

impl<B: BitInt> FermionicBitStateOp<B> for SignedPermDitMask<B> {
    fn fermion_sign(&self, state: B) -> f64 {
        // Product over occupied sites (bits set in `state`).
        let mut s = 1.0;
        for i in 0..self.sign.len() {
            if (state & B::from_u64(1u64 << i)) != B::from_u64(0) {
                s *= self.sign[i];
            }
        }
        s
    }
}
```

The symmetry walker consults `FermionicBitStateOp::fermion_sign` **only** when the basis's `fermionic` flag is `true`. For non-fermionic bases the method isn't called, so the default-`1.0` branch is dead code — no overhead on spin-1/2 paths.

**Spin inversion on spin-1/2:** uses the existing `PermDitMask<B>` (no sign, zero overhead). Spin inversion on a fermionic basis, if ever needed, gets its own dedicated type with its own sign convention; it is **out of scope** for this refactor.

**Basis parameterisation:** a basis with particle-hole instantiates `SymBasis<B, SignedPermDitMask<B>, N>`; a basis with only spin-inversion instantiates `SymBasis<B, PermDitMask<B>, N>`. Different `L` → different monomorphisation → no runtime dispatch cost.

### 3.5 Always-on validation at `build()` time

Group order is tiny (typically ≤ 2L, rarely > 100). `O(n²)` validation runs in microseconds — negligible compared to BFS — so we always validate, even in release builds.

Two checks run at the start of `build()` before any BFS work:

**(a) Closure.** For every pair `(g_i, g_j)` in the full list of added elements (plus identity), compute `g_i · g_j` by applying the composite perm and local to a canonical probe state (identity acts as the reference). The result must match one of the elements in the list (or the identity). If any pair's product is not in the set, return `QuSpinError::ValueError("group is not closed: ...")`.

**(b) Character consistency.** For each pair `(g_i, g_j)` with product `g_k`, verify `χ(g_i · g_j) ≈ χ(g_i) · χ(g_j)`. Tolerances follow the BFS `AMP_CANCEL_TOL` convention. If any pair fails, return `QuSpinError::ValueError("character table inconsistent: χ(g_i g_j) ≠ χ(g_i) χ(g_j) for ...")`.

Error messages identify the specific offending pair by index so the user can find the mistake in their input list.

### 3.6 Orbit walker changes

Changes are mechanical given §3.3:

- `get_refstate`, `check_refstate`, `get_refstate_batch`, `check_refstate_batch`, and `iter_images` lose their current nested `for lat in lattice { for loc in local { … } }` loops.
- Replace with three sequential loops (one per storage vector) sharing the same running-character / candidate-rep logic.
- When `self.fermionic` is `true`, multiply `loc.fermion_sign(state_before_loc)` into the running character in the `local_only` and `composite` loops.
- Fermion sign from the permutation part (current `BenesLatticeElement` machinery) is unchanged.

---

## 4. Migration impact

### 4.1 `quspin-basis` internals

- `SymBasis` struct: `lattice` + `local` fields replaced with `lattice_only` + `local_only` + `composite`.
- `add_lattice` / `add_local` methods removed; `add_symmetry` / `add_cyclic` added.
- `orbit.rs`: walker rewritten with three homogeneous loops and fermion-sign multiplication.
- New `SymElement<L>` type with typed constructors.
- Always-on validation in `build()`.

### 4.2 `quspin-bitbasis`

- New `FermionicBitStateOp<B>: BitStateOp<B>` trait with default `fermion_sign = 1.0`.
- Blanket impls for `PermDitMask<B>`, `PermDitValues<N>`, `DynamicPermDitValues`.
- New `SignedPermDitMask<B>` type with `mask` + `sign: Vec<f64>` fields.

### 4.3 `quspin-py`

Every call site that uses `add_lattice` / `add_local` / `add_inv` migrates to `add_symmetry(χ, SymElement::…)`. The Python-facing method names (`add_lattice`, `add_inv`, etc.) stay as `#[pyo3(name = …)]` annotations; only their Rust bodies change.

If particle-hole is exposed to Python in this PR, a new Python method `add_particle_hole(ph_sign: Vec<f64>)` wraps the `SignedPermDitMask` + `SymElement::local` + `add_symmetry` chain.

### 4.4 Breaking changes

This is a **breaking change** to the basis Rust API:

- `SymBasis::add_lattice`, `SymBasis::add_local`, `SymBasis::add_inv` are removed.
- Migration: wrap in `SymElement::lattice(perm)` / `SymElement::local(op)` and pass to `add_symmetry`.
- Users who relied on the cartesian-product semantics (e.g. adding translation and spin-inversion separately expecting the full `⟨T⟩ × ⟨Z⟩` group) must now enumerate the group explicitly, or use `add_cyclic` for the cyclic part and additional `add_symmetry` calls for the products.

CHANGELOG.md will carry a before/after migration recipe.

---

## 5. Out of scope / known follow-ups

- **Spin inversion on fermionic bases.** Mathematically distinct from particle-hole's sign. Addressed in a separate refactor once the use case concretely surfaces.
- **Non-abelian groups.** The whole "characters multiply" story assumes 1D reps. Non-abelian support would need an explicit character-table API and matrix-valued characters; not needed now.
- **Automatic group closure.** The user lists elements; the library validates but does not *generate* closure from generators. A `from_generators` helper that closes under composition is a plausible follow-up, but explicit is the chosen default (§1, secondary goal).
- **Direct-product helper** (`add_direct_product(g_cyclic_1, g_cyclic_2)`). Trivially composed from two `add_cyclic` calls plus a few `add_symmetry` calls for the cross-products; not needed until a caller asks.

---

## 6. Decision log

| # | Question | Decision |
|---|----------|----------|
| 1 | Identity implicit or explicit? | Implicit; reject empty `SymElement` / `(None, None)` as a runtime error. |
| 2 | Keep `add_lattice` / `add_local` / `add_inv`? | Drop all three; single `add_symmetry` with typed `SymElement` constructors. |
| 3 | Closure + character validation cost | Always-on at `build()`. `O(n²)` for group-order `n ≤ ~100` is negligible vs BFS. |
| 4 | Cyclic-group sugar? | Yes: `add_cyclic(g, order, char_fn)`. Pure sugar over `add_symmetry`. Also add `SymElement::compose`. |
| 5 | `SymElement<L>` shape | Struct with `Option<Perm>` + `Option<L>`, constructors only way to build (illegal state prevention). |
| 6 | `SymBasis` storage | Three typed vectors (`lattice_only`, `local_only`, `composite`) for dispatch-free walker loops. |
| 7 | Fermion signs | `FermionicBitStateOp<B>: BitStateOp<B>` trait with default `1.0`. New `SignedPermDitMask<B>` type carries per-site `sign: Vec<f64>` for particle-hole. Spin-1/2 paths use unmodified `PermDitMask<B>` — no overhead. |
