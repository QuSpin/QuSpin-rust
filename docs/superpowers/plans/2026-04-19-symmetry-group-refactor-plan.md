# Implementation Plan: Symmetry-group API refactor

**Spec:** `docs/superpowers/specs/2026-04-19-symmetry-group-refactor.md`
**Date:** 2026-04-19
**Branch:** `phil/symmetry-group-refactor` (off `main`)

---

## Prerequisites

- PR #57 (`StateTransitions` decoupling) merged.
- `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` clean on `main`.
- Read the spec; §3.3 (three-vector storage), §3.4 (fermion signs), §3.5 (validation) are the load-bearing sections.

---

## Step 1 — Add `FermionicBitStateOp` trait

File: `crates/quspin-bitbasis/src/transform.rs` (existing module that already holds `BitStateOp<B>`).

```rust
/// Extension of `BitStateOp<B>` that reports a fermion sign when the
/// operator is applied to a state on a fermionic basis.
pub trait FermionicBitStateOp<B: BitInt>: BitStateOp<B> {
    /// Fermion sign contributed when applied to `state`. Default: 1.0.
    ///
    /// Consulted by the symmetry walker only when the enclosing
    /// `SymBasis` has `fermionic = true`; on non-fermionic bases the
    /// method is dead code and the branch is optimised out.
    fn fermion_sign(&self, _state: B) -> f64 {
        1.0
    }
}
```

Blanket impls for the existing types (keep them sign-free):

```rust
impl<B: BitInt> FermionicBitStateOp<B> for PermDitMask<B> {}
impl<B: BitInt, const N: usize> FermionicBitStateOp<B> for PermDitValues<N> {}
impl<B: BitInt> FermionicBitStateOp<B> for DynamicPermDitValues {}
```

Re-export `FermionicBitStateOp` from `quspin-bitbasis/src/lib.rs`.

Verification: `cargo check -p quspin-bitbasis`, `cargo test -p quspin-bitbasis` still green (no behaviour change for existing types).

---

## Step 2 — Add `SignedPermDitMask` type

File: `crates/quspin-bitbasis/src/transform.rs` (add alongside `PermDitMask`).

```rust
/// Bit-flip at every set bit in `mask`, with a per-site fermion sign.
///
/// Used for particle-hole transformations on a fermionic basis. When
/// applied to `state`:
///   - returns `state ^ mask` (same as `PermDitMask`),
///   - `fermion_sign(state) = ∏_{i: bit i of state is set} sign[i]`.
///
/// The per-site `sign` array has length `n_sites`. For a 1D fermionic
/// chain the canonical choice is `sign[i] = (-1)^i`.
///
/// Cross-reference: QuSpin/QuSpin#390 for the Python-side history of
/// the `ph_sign` argument on `*_fermion_basis_general`.
pub struct SignedPermDitMask<B> {
    mask: B,
    sign: Vec<f64>,
}

impl<B: BitInt> SignedPermDitMask<B> {
    pub fn new(mask: B, sign: Vec<f64>) -> Self {
        Self { mask, sign }
    }

    /// Convenience: `(-1)^i` per-site sign for a 1D chain of length `n_sites`.
    pub fn default_1d(mask: B, n_sites: usize) -> Self {
        let sign = (0..n_sites).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        Self::new(mask, sign)
    }
}

impl<B: BitInt> BitStateOp<B> for SignedPermDitMask<B> {
    fn apply(&self, state: B) -> B {
        state ^ self.mask
    }
}

impl<B: BitInt> FermionicBitStateOp<B> for SignedPermDitMask<B> {
    fn fermion_sign(&self, state: B) -> f64 {
        let mut s = 1.0;
        for (i, &si) in self.sign.iter().enumerate() {
            if (state & B::from_u64(1u64 << i)) != B::from_u64(0) {
                s *= si;
            }
        }
        s
    }
}
```

Unit tests in the same file:
- `apply` equals `PermDitMask::apply` for identical masks.
- `fermion_sign` on `|0000⟩` is always `1.0`.
- `fermion_sign` with `sign = [1,-1,1,-1]` on `|1010⟩` is `1 · 1 = 1`.
- `fermion_sign` with `sign = [1,-1,1,-1]` on `|1100⟩` is `1 · -1 = -1`.

Re-export `SignedPermDitMask` from `quspin-bitbasis/src/lib.rs`.

Verification: `cargo test -p quspin-bitbasis`.

---

## Step 3 — Add `SymElement<L>` type + `Compose` trait

File: `crates/quspin-basis/src/sym_element.rs` (new).

```rust
use num_complex::Complex;
use quspin_bitbasis::{BenesPermDitLocations, BitInt};

/// Composition trait for local operator types. Required for
/// `SymElement::compose` (and hence for `add_cyclic` powers).
pub trait Compose: Sized {
    fn compose(&self, other: &Self) -> Self;
}

// Implemented for each concrete L used by SymBasis:
impl<B: BitInt> Compose for PermDitMask<B> { … }
impl<const N: usize> Compose for PermDitValues<N> { … }
impl Compose for DynamicPermDitValues { … }
impl<B: BitInt> Compose for SignedPermDitMask<B> { … }

/// An explicit non-identity element of the symmetry group.
///
/// Built via [`lattice`], [`local`], or [`composite`] — the struct
/// fields are private so an illegal all-`None` instance cannot be
/// constructed. The identity element is implicit in every `SymBasis`
/// and is never represented with a `SymElement`.
pub struct SymElement<B: BitInt, L> {
    perm: Option<BenesPermDitLocations<B>>,
    local: Option<L>,
}

impl<B: BitInt, L: Compose> SymElement<B, L> {
    pub fn lattice(perm: &[usize], lhss: usize, fermionic: bool) -> Self;
    pub fn local(op: L) -> Self;
    pub fn composite(perm: &[usize], op: L, lhss: usize, fermionic: bool) -> Self;

    pub fn compose(&self, other: &Self) -> Self;
}
```

Composition rules:
- Lattice · Lattice = Lattice (perm composition)
- Local · Local = Local (op composition via `Compose::compose`)
- Lattice · Local = Composite (both components present)
- Composite · X = Composite (component-wise)

Add `mod sym_element; pub use sym_element::{SymElement, Compose};` to `quspin-basis/src/lib.rs`.

Unit tests:
- `SymElement::lattice(perm).compose(SymElement::lattice(perm)) == SymElement::lattice(perm²)`
- `SymElement::local(op).compose(SymElement::lattice(p)) == SymElement::composite(p, op)`
- `compose` is associative on a handful of sample elements.

Verification: `cargo test -p quspin-basis`.

---

## Step 4 — `SymBasis` storage: three vectors

File: `crates/quspin-basis/src/sym.rs`.

Replace:
```rust
pub(crate) lattice: Vec<BenesLatticeElement<B>>,
pub(crate) local: Vec<(Complex<f64>, L)>,
```

with:
```rust
pub(crate) lattice_only: Vec<(Complex<f64>, BenesLatticeElement<B>)>,
pub(crate) local_only:   Vec<(Complex<f64>, L)>,
pub(crate) composite:    Vec<(Complex<f64>, BenesLatticeElement<B>, L)>,
```

Update `SymBasis::new_empty` to initialise all three empty. Remove `add_lattice` and `add_local`.

Add `add_symmetry`:

```rust
impl<B: BitInt, L: BitStateOp<B> + FermionicBitStateOp<B> + Compose, N: NormInt>
    SymBasis<B, L, N>
{
    pub fn add_symmetry(
        &mut self,
        char: Complex<f64>,
        element: SymElement<B, L>,
    ) -> Result<(), QuSpinError> {
        let SymElement { perm, local } = element;
        match (perm, local) {
            (Some(p), None) => {
                self.lattice_only.push((char, BenesLatticeElement::from_perm(p, self.n_sites)));
            }
            (None, Some(l)) => {
                self.local_only.push((char, l));
            }
            (Some(p), Some(l)) => {
                self.composite.push((char, BenesLatticeElement::from_perm(p, self.n_sites), l));
            }
            (None, None) => {
                // Guarded by SymElement constructors — unreachable in practice,
                // but guard defensively if someone bypasses the typed builder.
                return Err(QuSpinError::ValueError(
                    "empty symmetry element: identity is implicit, do not add it".into(),
                ));
            }
        }
        Ok(())
    }
}
```

Add `add_cyclic`:

```rust
pub fn add_cyclic(
    &mut self,
    generator: SymElement<B, L>,
    order: usize,
    char_fn: impl Fn(usize) -> Complex<f64>,
) -> Result<(), QuSpinError> {
    if order < 2 {
        return Err(QuSpinError::ValueError(
            "add_cyclic requires order >= 2".into(),
        ));
    }
    let mut gk = generator.clone();  // g^1
    for k in 1..order {
        self.add_symmetry(char_fn(k), gk.clone())?;
        if k + 1 < order {
            gk = gk.compose(&generator);
        }
    }
    Ok(())
}
```

Verification: `cargo check -p quspin-basis`. Tests and walker won't pass yet — step 6 fixes the walker.

---

## Step 5 — Implement closure + character validation

File: `crates/quspin-basis/src/sym.rs`, new `fn validate_group(&self)` called at the start of `SymBasis::build`.

```rust
fn validate_group(&self) -> Result<(), QuSpinError> {
    // Collect all elements including identity at index 0.
    let n = 1 + self.lattice_only.len() + self.local_only.len() + self.composite.len();
    let elements: Vec<(Complex<f64>, Option<&BenesLatticeElement<B>>, Option<&L>)> =
        std::iter::once((Complex::new(1.0, 0.0), None, None))
            .chain(self.lattice_only.iter().map(|(c, p)| (*c, Some(p), None)))
            .chain(self.local_only.iter().map(|(c, l)| (*c, None, Some(l))))
            .chain(self.composite.iter().map(|(c, p, l)| (*c, Some(p), Some(l))))
            .collect();

    // Probe: apply g_i · g_j to state = 0 (or a canonical state), find which
    // element of the list matches the resulting (perm, local) action. Since
    // perms and locals are state-independent transformations, we can compare
    // by applying to a small set of probe states (e.g. the first few states).

    for i in 0..n {
        for j in 0..n {
            let composed = compose_action(&elements[i], &elements[j]);
            let k = elements.iter().position(|e| action_eq(e, &composed));
            match k {
                None => return Err(QuSpinError::ValueError(format!(
                    "group is not closed: g_{i} · g_{j} is not in the supplied \
                     element list (index 0 = implicit identity)"
                ))),
                Some(k) => {
                    let expected = elements[i].0 * elements[j].0;
                    let actual = elements[k].0;
                    if (expected - actual).norm() > CHAR_TOL {
                        return Err(QuSpinError::ValueError(format!(
                            "character table inconsistent: χ(g_{i} g_{j}) = {expected}, \
                             but χ(g_{k}) = {actual}"
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

const CHAR_TOL: f64 = 1e-10;
```

Helpers:
- `compose_action`: takes two (perm, local) pairs and produces their composition as a third pair. Perm composition uses existing machinery; local uses `Compose::compose`.
- `action_eq`: applies both actions to a fixed sequence of probe states (e.g. `[0, 1, 2, ..., lhss^n_sites - 1]` truncated to 8 or so) and checks state-by-state equality. This is the cheap empirical equality check; since the space of permutations × local ops is finite and our ops are faithfully represented, 8 probe states are more than enough to detect distinct actions.

Verification: add tests in `sym.rs`:
- Closure check catches missing element (add `PZ` only but ALSO `P` — group is `⟨P, Z⟩` which requires `Z` and `PZ` too; user gave only `{P}` so closure fails because `P · P = I` but `P · P² = P` — actually let me pick a cleaner one). A clean failing case: add `T` (translation by 1) on L=4, forget `T²`, `T³`. Closure check detects `T · T = T²` is missing.
- Character check catches `χ(g²) ≠ χ(g)²` mismatch.

---

## Step 6 — Orbit walker changes

Files:
- `crates/quspin-basis/src/orbit.rs` (`get_refstate`, `check_refstate`, `get_refstate_batch`, `check_refstate_batch`, `iter_images`).

Replace the nested cartesian-product loop with three sequential homogeneous loops, sharing the same best-rep-so-far + accumulated-character bookkeeping. Concretely:

```rust
pub(crate) fn get_refstate<B, L, E>(
    lattice_only: &[(Complex<f64>, BenesLatticeElement<B>)],
    local_only: &[(Complex<f64>, L)],
    composite: &[(Complex<f64>, BenesLatticeElement<B>, L)],
    fermionic: bool,
    state: B,
) -> (B, Complex<f64>) {
    let mut best_state = state;
    let mut best_char = Complex::new(1.0, 0.0);  // identity contribution

    for (chi, perm) in lattice_only {
        let s = perm.apply(state);
        let sign = if fermionic { perm.fermion_sign(state) } else { 1.0 };
        update_best(&mut best_state, &mut best_char, s, chi * sign);
    }
    for (chi, loc) in local_only {
        let s = loc.apply(state);
        let sign = if fermionic { loc.fermion_sign(state) } else { 1.0 };
        update_best(&mut best_state, &mut best_char, s, chi * sign);
    }
    for (chi, perm, loc) in composite {
        let perm_state = perm.apply(state);
        let s = loc.apply(perm_state);
        let sign_perm = if fermionic { perm.fermion_sign(state) } else { 1.0 };
        let sign_loc  = if fermionic { loc.fermion_sign(perm_state) } else { 1.0 };
        update_best(&mut best_state, &mut best_char, s, chi * sign_perm * sign_loc);
    }

    (best_state, best_char)
}
```

The `fermionic` branch predicate is constant per basis — LLVM can hoist it out of the hot loop (or we can manually specialise into `get_refstate_fermion` / `get_refstate_spin` if benchmarks show an issue).

Apply the same three-loop transformation to `check_refstate`, the batch variants, and `iter_images`.

Verification: `cargo test -p quspin-basis`. All 77 basis tests must pass. Tests that depended on the cartesian-product cost-accounting (e.g. expecting `|G| = |lattice| × |local|`) will need their expected sizes updated — those will surface as test failures and be fixed individually.

---

## Step 7 — Update existing `make_space_inner` / dispatch macros

File: `crates/quspin-basis/src/lib.rs`.

The `SpaceKind::Symm` branch of `make_space_inner` doesn't change — it still constructs a `SymBasis<B, L, N>` with `new_empty`, and the caller then calls `add_symmetry`. But the `with_sym_basis_mut!` / `with_dit_sym_basis_mut!` / etc. macros may need inspection: they're used by `quspin-py` to access the basis mutably and call `add_lattice` / `add_local`. After this refactor those calls migrate to `add_symmetry` with typed elements.

Verification: grep `with_sym_basis_mut` in `quspin-py/src/basis/` and confirm every call site is updated.

---

## Step 8 — `quspin-py` migration

Files: `crates/quspin-py/src/basis/{spin,boson,fermion,generic}.rs`.

Every place that today calls `basis.add_lattice(χ, perm)` or `basis.add_local(χ, op)` migrates to:

```rust
// Before
basis.add_lattice(chi, &perm);
basis.add_local(chi, op);

// After
basis.add_symmetry(chi, SymElement::lattice(&perm, lhss, fermionic))?;
basis.add_symmetry(chi, SymElement::local(op))?;
```

For the particle-hole case (fermion basis only), expose a new Python-facing method on the fermion basis class that wraps the `SignedPermDitMask` + `SymElement::local` + `add_symmetry` chain:

```rust
#[pyo3(signature = (ph_sign = None))]
fn add_particle_hole(&mut self, ph_sign: Option<Vec<f64>>) -> PyResult<()> {
    let mask = /* all-sites mask for the basis */;
    let sign = ph_sign.unwrap_or_else(|| default_1d_signs(self.n_sites));
    let op = SignedPermDitMask::new(mask, sign);
    let elem = SymElement::local(op);
    self.inner.add_symmetry(Complex::new(1.0, 0.0), elem)
        .map_err(Error::from)?;
    Ok(())
}
```

Python-facing method names for the existing `add_lattice` / `add_inv` PyO3 wrappers stay the same (they're `#[pyo3(name = …)]` annotations); only the Rust bodies change to construct a `SymElement` and call `add_symmetry`.

Verification: `uv run maturin develop && uv run pytest python/tests/ -x -q -m "not slow"`.

---

## Step 9 — Docs

- `CLAUDE.md`: no DAG change (intra-crate refactor), but update the "Key design rules" list to mention `SymElement` as the user-facing element type and the three-vector storage as the walker-performance invariant.
- `docs/superpowers/specs/2026-04-19-state-graph-decoupling.md`: cross-link the new spec from the "out of scope" section (the symmetry refactor was called out as a follow-up).
- `CHANGELOG.md`: add a breaking-changes entry with before/after migration recipe for `add_lattice` / `add_local` / `add_inv`.

---

## Step 10 — Final verification

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uv run maturin develop
uv run pytest python/tests/ -x -q -m "not slow"

# Sanity for the invariant that motivated this refactor:
# a basis with only a PZ composite element has |G| = 2, not 4.
cargo test -p quspin-basis composite_pz_group_has_order_2 -- --nocapture
```

Open PR targeting `main`.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `fermionic` branch in the walker hot loop costs noticeable perf | Manual specialisation into `*_fermion` / `*_spin` functions. Benchmark first; almost certainly unnecessary because LLVM will hoist the constant branch. |
| Validation false-positives due to tolerance | Use the `AMP_CANCEL_TOL` convention; make the tolerance a pub constant so call-sites that legitimately need looser comparison can bypass. |
| `compose` on `PermDitValues<N>` is awkward because `N` is a const generic | `Compose::compose` is type-preserving, so `PermDitValues<3>·PermDitValues<3> → PermDitValues<3>`. Implement by composing the underlying permutation arrays. |
| Migration surface in quspin-py larger than expected | Grep for `add_lattice` / `add_local` / `add_inv` before starting; each is a mechanical single-line rewrite. |

---

## Decision log

- **Identity:** implicit, rejected as error if user tries to add it.
- **API surface:** single `add_symmetry` method; `add_lattice` / `add_local` / `add_inv` removed. Typed constructors (`SymElement::lattice`, `::local`, `::composite`) are the only way to build an element.
- **Storage:** three typed vectors (`lattice_only`, `local_only`, `composite`), populated at insert time. Walker iterates three homogeneous loops.
- **Validation:** always-on closure + character checks at `build()`. O(n²) in group order; sub-millisecond for realistic `n`.
- **Fermion signs:** new trait `FermionicBitStateOp: BitStateOp` with default `fermion_sign = 1.0`. New concrete type `SignedPermDitMask<B>` carries `mask: B` + `sign: Vec<f64>`. Walker consults the sign method only when `basis.fermionic == true`.
- **Out of scope:** spin-inversion on fermions (needs its own sign convention), non-abelian groups, automatic generator closure, direct-product helper.
