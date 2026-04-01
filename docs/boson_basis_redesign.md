# BosonBasis Redesign Plan

## Goal

Remove `SpinSymGrp`, `DitSymGrp`, and `SymmetryGrpInner` from the `BosonBasis` builder
path entirely. `BosonBasis` becomes:

```rust
pub struct BosonBasis {
    pub n_sites: usize,
    pub lhss:    usize,
    space_kind:  SpaceKind,
    pub inner:   SpaceInner,
}
```

## Key Insight

`SymmetryGrpInner` is redundant. `SpaceInner::Sym*` variants hold
`SymBasis<B, PermDitMask<B>, N>` and `SpaceInner::DitSym*` hold
`SymBasis<B, DynamicPermDitValues, N>`. `SymBasis` already stores:

```
lattice: Vec<BenesLatticeElement<B>>   ← same field as SymGrpBase
local:   Vec<(Complex<f64>, L)>        ← same field as SymGrpBase
states:  Vec<(B, N)>                   ← populated during build()
index_map: HashMap<B, usize>           ← populated during build()
```

`SymGrpBase` / `SymmetryGrpInner` is only used as a pre-build staging area for
`lattice` and `local` before `SymBasis::from_grp` copies them in. We can skip this
entirely by starting with an empty `SymBasis` and adding group elements directly to it.

The same logic applies to `Subspace`: it is `Vec<B>` + `HashMap<B, usize>`, which
starts empty and gets populated during `build`.

Both `SymBasis` and `Subspace` will carry an explicit `built: bool` flag (see below)
to track whether `build()` has been called. This is necessary because `SymBasis::build()`
only inserts a state if its orbit norm is positive — a seed with zero norm produces no
states, leaving `states` empty even after build has run. Relying on
`!self.states.is_empty()` would give a false negative in that case, allowing
`build_boson` to be called a second time.

---

## Required Additions to Existing Types

### 1. `SymBasis<B, L, N>` — add builder methods

Currently constructed only via `from_grp(SymGrpBase)`. Add a `built: bool` field and
the following methods:

```rust
pub struct SymBasis<B: BitInt, L, N: NormInt> {
    // ... existing fields ...
    built: bool,   // ← new: set to true at the start of build(), never reset
}

impl<B: BitInt, L, N: NormInt> SymBasis<B, L, N> {
    /// Construct an empty basis with no group elements and no states.
    pub fn new_empty(lhss: usize, n_sites: usize, fermionic: bool) -> Self;

    /// Add a lattice (site-permutation) element. Valid before build().
    pub fn push_lattice(&mut self, grp_char: Complex<f64>, perm: &[usize]);

    /// True once build() has been called, regardless of whether any states were added.
    pub fn is_built(&self) -> bool { self.built }
}
```

`push_lattice` computes a `BenesLatticeElement<B>` from `perm` and appends it to
`self.lattice` — the same logic currently in `SymGrpBase::push_lattice`.

`built` is set to `true` at the very start of `build()`, before any state is evaluated.
This ensures that a seed whose orbit norm is zero (which produces no entries in `states`)
still marks the basis as built, preventing a second `build_boson` call.

### 2. `Subspace<B>` — add `is_built`

`Subspace::build()` unconditionally inserts the seed, so the false-negative case does
not arise there. Nevertheless, add the same `built: bool` flag for consistency:

```rust
pub struct Subspace<B: BitInt> {
    // ... existing fields ...
    built: bool,   // ← new
}

impl<B: BitInt> Subspace<B> {
    pub fn new_empty(n_sites: usize, lhss: usize) -> Self;
    pub fn is_built(&self) -> bool { self.built }
}
```

### 3. `SpaceInner` — add dispatch methods for builder phase

```rust
impl SpaceInner {
    /// Push a lattice element. Errors if not a Sym*/DitSym* variant.
    pub fn push_lattice(
        &mut self,
        grp_char: Complex<f64>,
        perm: &[usize],
    ) -> Result<(), QuSpinError>;

    /// True if states have been populated (i.e. build() has been called).
    pub fn is_built(&self) -> bool;
}
```

`push_lattice` matches on all 18 `Sym*`/`DitSym*` variants and calls
`sym_basis.push_lattice(...)`. Returns an error for `Full*` / `Sub*`.

`is_built` returns:
- `Full*` → always `true`
- `Sub*` → `subspace.is_built()`
- `Sym*` / `DitSym*` → `sym_basis.is_built()`

---

## `BosonBasis` Construction

### `new(n_sites, lhss, space_kind)`

| space_kind | lhss  | `inner` at construction |
|------------|-------|-------------------------|
| `Full`     | >= 2  | `select_b_for_n_sites!` → `SpaceInner::Full32/64(FullSpace::new(...))` |
| `Sub`      | any   | `select_b_for_n_sites!` → `SpaceInner::Sub*(Subspace::new_empty(...))` |
| `Symm`     | == 2  | `select_b_for_n_sites!` → `SpaceInner::Sym*(SymBasis::new_empty(lhss=2, ...))` |
| `Symm`     | > 2   | `select_b_for_n_sites!` → `SpaceInner::DitSym*(SymBasis::new_empty(lhss, ...))` |

Validation at `new()`:
- `lhss < 2` → error
- `Full` with `n_sites > 64` → error
- `Sub`/`Symm` with `n_sites > 8192` → error

---

## `BosonBasis` Methods

### `add_lattice`

```rust
pub fn add_lattice(
    &mut self,
    grp_char: Complex<f64>,
    perm: &[usize],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind != Symm`
- Errors if `inner.is_built()` (too late, build already called)
- Errors if `perm.len() != self.n_sites`
- Delegates to `self.inner.push_lattice(grp_char, perm)`

### `build_boson`

```rust
pub fn build_boson(
    &mut self,
    ham: &BosonHamiltonianInner,
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind == Full` ("Full basis requires no build")
- Errors if `inner.is_built()` (already built)

The build step mutates `self.inner` in place:

```
Sub* variant:
    with_plain_basis_mut!(&mut self.inner, B, subspace, {
        enumerate ham from seeds → populate subspace.states + subspace.index_map
    })

Sym* variant (lhss=2):
    with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })

DitSym* variant (lhss>2):
    with_dit_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })
```

---

## Mutable Dispatch

The existing macros (`with_sym_basis!`, etc.) take `&SpaceInner`. We need `&mut`
variants, or equivalent methods on `SpaceInner`. The simplest approach is adding
`_mut` macro variants that mirror the existing ones but bind `mut`:

```rust
macro_rules! with_sym_basis_mut {
    ($inner:expr, $B:ident, $basis:ident, $body:block) => {
        match $inner {
            SpaceInner::Sym32(ref mut $basis) => { type $B = u32; $body }
            SpaceInner::Sym64(ref mut $basis) => { type $B = u64; $body }
            // ... 9 variants
            _ => unreachable!()
        }
    }
}
```

---

## What Does NOT Change

- `SpaceInner` enum shape (29 variants) — unchanged, just gets new methods
- `SymBasis<B, L, N>` struct shape — gains one `built: bool` field; otherwise unchanged
- `Subspace<B>` struct shape — gains one `built: bool` field; otherwise unchanged
- `with_sym_basis!`, `with_dit_sym_basis!`, `with_plain_basis!` macros — unchanged
- `select_b_for_n_sites!` macro — unchanged
- `BosonHamiltonianInner`, `FermionHamiltonianInner` — unchanged

---

## Summary of Steps

1. Add `SymBasis::new_empty`, `push_lattice`, `is_built`
2. Add `Subspace::new_empty`, `is_built`
3. Add `SpaceInner::push_lattice`, `is_built`
4. Add `with_sym_basis_mut!`, `with_dit_sym_basis_mut!`, `with_plain_basis_mut!` macros
5. Write `boson_basis.rs` using the new struct and methods
6. Write `fermion_basis.rs` using the same infrastructure (see FermionBasis extension below)
7. Add `build_bond` to both `BosonBasis` and `FermionBasis` (see BondHamiltonian extension below)

---

## Extension: `FermionBasis`

`FermionBasis` follows the same design as `BosonBasis` with three differences:

1. No `lhss` field — fermions are always `lhss = 2`.
2. `fermionic: true` is passed to `SymBasis::new_empty`, enabling Jordan-Wigner sign
   computation inside `BenesLatticeElement::grp_char_for()`.
3. Build uses `FermionHamiltonianInner` and only ever dispatches through `Sym*` variants
   (never `DitSym*`).

### Struct

```rust
pub struct FermionBasis {
    pub n_sites: usize,
    space_kind:  SpaceKind,
    pub inner:   SpaceInner,
}
```

### `new(n_sites, space_kind)`

| space_kind | `inner` at construction |
|------------|-------------------------|
| `Full`     | `select_b_for_n_sites!` → `SpaceInner::Full*(FullSpace::new(...))` |
| `Sub`      | `select_b_for_n_sites!` → `SpaceInner::Sub*(Subspace::new_empty(...))` |
| `Symm`     | `select_b_for_n_sites!` → `SpaceInner::Sym*(SymBasis::new_empty(lhss=2, n_sites, fermionic=true))` |

Validation at `new()`:
- `Full` with `n_sites > 64` → error
- `Sub`/`Symm` with `n_sites > 8192` → error

### `add_lattice`

Identical to `BosonBasis::add_lattice`.

### `build_fermion`

```rust
pub fn build_fermion(
    &mut self,
    ham: &FermionHamiltonianInner,
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind == Full`
- Errors if `inner.is_built()`

Dispatch:

```
Sub* variant:
    with_plain_basis_mut!(&mut self.inner, B, subspace, {
        enumerate ham from seeds → populate subspace.states + subspace.index_map
    })

Sym* variant (always, since lhss=2):
    with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })
```

`DitSym*` variants are unreachable for `FermionBasis` and need not be matched.

---

## Extension: `BondHamiltonian`

`BondHamiltonianInner` already exists in the codebase. It differs from
`BosonHamiltonianInner` and `FermionHamiltonianInner` in that operators are expressed as
dense two-site matrices rather than strings of creation/annihilation operators. This
makes it fully generic: `lhss` is inferred from the matrix shape (`sqrt(nrows)`), and
the same `apply` callback pattern is used, so it slots into both build paths without
changes to `SymBasis` or `Subspace`.

### What `BondHamiltonianInner` looks like

```rust
pub enum BondHamiltonianInner {
    Ham8(BondHamiltonian<u8>),
    Ham16(BondHamiltonian<u16>),
}

pub struct BondHamiltonian<C> {
    terms: Vec<BondTerm<C>>,
    lhss: usize,           // inferred from matrix shape: sqrt(term.matrix.nrows())
    max_site: usize,
    num_cindices: usize,
}

pub struct BondTerm<C> {
    pub cindex: C,
    pub matrix: Array2<Complex<f64>>,  // (lhss² × lhss²) two-site interaction matrix
    pub bonds: Vec<(u32, u32)>,        // site pairs (si, sj) to apply the matrix to
}
```

Matrix encoding: row/column index `a * lhss + b` encodes site `si` with occupancy `a`
and site `sj` with occupancy `b`. `matrix[[out_row, in_col]] = ⟨out|M|in⟩`.

`BondHamiltonianInner` already implements the shared `Hamiltonian<C>` trait, so the
`build_bond` dispatch body is identical to `build_boson`/`build_fermion` — only the
type of `ham` and the lhss validation differ.

### `BosonBasis::build_bond`

```rust
pub fn build_bond(
    &mut self,
    ham: &BondHamiltonianInner,
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind == Full`
- Errors if `inner.is_built()`
- Errors if `ham.lhss() != self.lhss` (matrix lhss must match basis lhss)

Dispatch is identical to `build_boson`:

```
Sub* variant:
    with_plain_basis_mut!(&mut self.inner, B, subspace, {
        enumerate ham from seeds → populate subspace.states + subspace.index_map
    })

Sym* variant (lhss=2):
    with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })

DitSym* variant (lhss>2):
    with_dit_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })
```

### `FermionBasis::build_bond`

```rust
pub fn build_bond(
    &mut self,
    ham: &BondHamiltonianInner,
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind == Full`
- Errors if `inner.is_built()`
- Errors if `ham.lhss() != 2` (fermions are always lhss=2; a bond matrix for lhss>2 is a caller error)

Dispatch is identical to `build_fermion` (`Sym*` only, `DitSym*` unreachable):

```
Sub* variant:
    with_plain_basis_mut!(&mut self.inner, B, subspace, {
        enumerate ham from seeds → populate subspace.states + subspace.index_map
    })

Sym* variant (always, since lhss=2):
    with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })
```

---

## Extension: `SpinBasis`

`SpinBasis` follows the same design as `BosonBasis` with two differences:

1. `add_inv` — a new method that adds a spin-inversion local symmetry element.
2. `build_spin` takes a `SpinHamiltonianInner` instead of `BosonHamiltonianInner`.

### Struct

```rust
pub struct SpinBasis {
    pub n_sites: usize,
    pub lhss:    usize,
    space_kind:  SpaceKind,
    pub inner:   SpaceInner,
}
```

### `new(n_sites, lhss, space_kind)`

Identical to `BosonBasis::new`. Validation:
- `lhss < 2` → error
- `Full` with `n_sites > 64` → error
- `Sub`/`Symm` with `n_sites > 8192` → error

### `add_lattice`

Identical to `BosonBasis::add_lattice`.

### `add_inv`

```rust
pub fn add_inv(
    &mut self,
    locs: Option<Vec<u32>>,
) -> Result<(), QuSpinError>
```

Adds the spin-inversion local symmetry: maps spin projection `m → -m` at the specified
sites (or all sites if `locs` is `None`). The group character is always `−1.0`.

- Errors if `space_kind != Symm`
- Errors if `inner.is_built()`

**Implementation differs by lhss:**

`lhss = 2` — `Sym*` variants hold `PermDitMask<B>`:

Spin-inversion at each target site is a single-bit XOR. Build a mask with one bit set per
target site (using `DynamicDitManip` to get bit positions), then push a
`(grp_char, PermDitMask::new(mask))` onto `sym_basis.local`.

```
mask = 0
for loc in resolved_locs:
    mask |= manip.site_mask(loc)   // sets the single bit for site `loc`
sym_basis.local.push((grp_char, PermDitMask::new(mask)))
```

`lhss > 2` — `DitSym*` variants hold `DynamicPermDitValues`:

The inversion permutation `n → lhss - 1 - n` reverses the local state ordering
(highest spin at dit=0 maps to lowest spin and vice versa). Construct once and apply at
all target sites.

```
perm: Vec<u8> = (0..lhss).rev().collect()   // [lhss-1, lhss-2, ..., 1, 0]
locs_usize: Vec<usize> = resolved_locs as usize
sym_basis.local.push((grp_char, DynamicPermDitValues::new(lhss, perm, locs_usize)))
```

This is the same permutation already constructed by `SymGrpBase::push_spin_inv`
in the old code path.

**Required addition to `SymBasis` and `SpaceInner`:**

`add_inv` needs a `push_local` builder method on `SymBasis`, analogous to `push_lattice`:

```rust
impl<B: BitInt, L, N: NormInt> SymBasis<B, L, N> {
    /// Add a local symmetry element. Valid before build().
    pub fn push_local(&mut self, grp_char: Complex<f64>, local_op: L);
}
```

And corresponding dispatch methods on `SpaceInner`:

```rust
impl SpaceInner {
    /// Push a local mask element onto Sym* variants (lhss=2). Errors for others.
    pub fn push_local_mask<B: BitInt>(
        &mut self,
        grp_char: Complex<f64>,
        mask: B,
    ) -> Result<(), QuSpinError>;

    /// Push a local permutation element onto DitSym* variants (lhss>2). Errors for others.
    pub fn push_local_perm(
        &mut self,
        grp_char: Complex<f64>,
        perm: Vec<u8>,
        locs: Vec<usize>,
    ) -> Result<(), QuSpinError>;
}
```

`SpinBasis::add_inv` dispatches to `push_local_mask` for `Sym*` (lhss=2) and
`push_local_perm` for `DitSym*` (lhss>2), selecting the right branch from `self.lhss`.

### `build_spin`

```rust
pub fn build_spin(
    &mut self,
    ham: &SpinHamiltonianInner,
    seeds: &[Vec<u8>],
) -> Result<(), QuSpinError>
```

- Errors if `space_kind == Full`
- Errors if `inner.is_built()`
- Errors if `ham.lhss() != self.lhss`

Dispatch is identical to `BosonBasis::build_boson`:

```
Sub* variant:
    with_plain_basis_mut!(&mut self.inner, B, subspace, {
        enumerate ham from seeds → populate subspace.states + subspace.index_map
    })

Sym* variant (lhss=2):
    with_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })

DitSym* variant (lhss>2):
    with_dit_sym_basis_mut!(&mut self.inner, B, sym_basis, {
        for seed in seeds { sym_basis.build(seed, |s| ham.apply(s)) }
    })
```

### `build_bond`

Identical to `BosonBasis::build_bond` (see above). Errors if `ham.lhss() != self.lhss`.
