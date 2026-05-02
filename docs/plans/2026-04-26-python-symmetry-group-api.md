# Python Symmetry-Group API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the loose `symmetries=[(perm, (re, im))]` / `local_symmetries=[…]`
keyword arguments on every `*Basis.symmetric(...)` constructor with a single
first-class `SymmetryGroup` object, plus opaque `Lattice` / `Local` / `Composite`
element constructors.

**Architecture:** New `PySymElement` `#[pyclass]` in `quspin-py` carries the
LHSS-agnostic triple `(perm, perm_vals, locs)`. A pure-Python `SymmetryGroup`
class (in `python/quspin_rs/symmetry.py`) collects `(element, character)` pairs
and ships them to a new `add_symmetry_raw` dispatch entry on
`GenericBasis` / `BitBasis` / `DitBasis` at construction time. The Rust dispatch
gets one new per-family inner-enum method (`add_composite`); per-family
validation already lives where the typed local op is constructed.

**Tech Stack:** Rust (quspin-basis, quspin-py), PyO3 0.23, Python 3.10+,
pytest, maturin.

**Spec:** `docs/superpowers/specs/2026-04-26-python-symmetry-group-api-design.md`

---

## Phase 1 — Rust dispatch entry: `add_composite` + `add_symmetry_raw`

### Task 1: Bit family — `add_composite` on `BitBasisDefault` and `BitBasisLargeInt`

**Files:**
- Modify: `crates/quspin-basis/src/dispatch/bit.rs`
- Test: same file (`#[cfg(test)] mod tests`)

**Step 1: Write failing tests**

```rust
// crates/quspin-basis/src/dispatch/bit.rs (under existing #[cfg(test)] mod tests)
#[test]
fn add_composite_pz_4_sites() {
    use quspin_bitbasis::PermDitMask;
    use crate::SpaceKind;
    let mut basis = crate::dispatch::BitBasis::new(4, SpaceKind::Symm, false).unwrap();
    // PZ on 4 sites: perm=[3,2,1,0], perm_vals=[1,0]
    basis.add_composite(
        num_complex::Complex::new(-1.0, 0.0),
        &[3, 2, 1, 0],
        vec![1, 0],
        vec![0, 1, 2, 3],
    ).unwrap();
    // Successful add — exact group_order check is implicit in the no-error flow.
}

#[test]
fn add_composite_rejects_non_sym_variant() {
    use crate::SpaceKind;
    let mut basis = crate::dispatch::BitBasis::new(4, SpaceKind::Sub, false).unwrap();
    let r = basis.add_composite(
        num_complex::Complex::new(1.0, 0.0),
        &[1, 0, 3, 2],
        vec![1, 0],
        vec![0, 1, 2, 3],
    );
    assert!(r.is_err());
}
```

**Step 2: Run tests, expect fail**

```sh
cargo test -p quspin-basis bit::tests::add_composite -- --exact 2>&1 | tail -20
```
Expected: compile error — `add_composite` does not exist on `BitBasisDefault`.

**Step 3: Add `add_composite` to `BitBasisDefault` and `BitBasisLargeInt`**

In `bit.rs`, add to `impl BitBasisDefault`:

```rust
/// Add a composite (lattice + local) element. `perm_vals` must be `[1, 0]`
/// (Bit family LHSS=2). `locs` is validated against `n_sites`.
pub fn add_composite(
    &mut self,
    grp_char: Complex<f64>,
    perm: &[usize],
    perm_vals: Vec<u8>,
    locs: Vec<usize>,
) -> Result<(), QuSpinError> {
    validate_perm_vals(&perm_vals, 2)?;
    if perm_vals != [1, 0] {
        return Err(QuSpinError::ValueError(format!(
            "add_composite on Bit family requires perm_vals=[1,0], got {perm_vals:?}"
        )));
    }
    validate_locs(&locs, self.n_sites())?;
    // perm validation runs inside SymBasis::add_symmetry on the perm component.
    match self {
        Self::Sym32(b) => b.add_symmetry(
            grp_char,
            SymElement::composite(perm, PermDitMask::new(build_mask::<u32>(&locs))),
        ),
        Self::Sym64(b) => b.add_symmetry(
            grp_char,
            SymElement::composite(perm, PermDitMask::new(build_mask::<u64>(&locs))),
        ),
        Self::Sym128(b) => b.add_symmetry(
            grp_char,
            SymElement::composite(perm, PermDitMask::new(build_mask::<B128>(&locs))),
        ),
        Self::Sym256(b) => b.add_symmetry(
            grp_char,
            SymElement::composite(perm, PermDitMask::new(build_mask::<B256>(&locs))),
        ),
        _ => Err(QuSpinError::ValueError(
            "add_composite requires a Sym* variant on BitBasisDefault".into(),
        )),
    }
}
```

Add the analogous method to `impl BitBasisLargeInt` (under `#[cfg(feature = "large-int")]`), covering Sym512/1024/2048/4096/8192 with `B512`/`B1024`/`B2048`/`B4096`/`B8192`.

Add the family-level delegation on `impl BitBasis`:

```rust
#[inline]
pub fn add_composite(
    &mut self,
    grp_char: Complex<f64>,
    perm: &[usize],
    perm_vals: Vec<u8>,
    locs: Vec<usize>,
) -> Result<(), QuSpinError> {
    match self {
        Self::Default(inner) => inner.add_composite(grp_char, perm, perm_vals, locs),
        #[cfg(feature = "large-int")]
        Self::LargeInt(inner) => inner.add_composite(grp_char, perm, perm_vals, locs),
    }
}
```

**Step 4: Run tests, expect pass**

```sh
cargo test -p quspin-basis bit::tests::add_composite -- --exact 2>&1 | tail -10
cargo check --workspace --features quspin-py/large-int 2>&1 | tail -5
```
Expected: both add_composite tests PASS; large-int check succeeds.

**Step 5: Commit**

```sh
git add crates/quspin-basis/src/dispatch/bit.rs
git commit -m "feat(basis): add Bit-family add_composite for lattice+local symmetry elements"
```

---

### Task 2: Trit / Quat / DynDit `add_composite`

**Files:**
- Modify: `crates/quspin-basis/src/dispatch/trit.rs`
- Modify: `crates/quspin-basis/src/dispatch/quat.rs`
- Modify: `crates/quspin-basis/src/dispatch/dit.rs`

**Step 1: Write failing tests**

For each of the three files, add a test analogous to `add_composite_pz_4_sites` in
Task 1 but using the family's local-op type. Example for `trit.rs`:

```rust
#[test]
fn add_composite_3site_lhss3() {
    use quspin_bitbasis::PermDitValues;
    use crate::SpaceKind;
    let mut basis = crate::dispatch::DitBasis::new(3, 3, SpaceKind::Symm).unwrap();
    if let crate::dispatch::DitBasis::Trit(ref mut t) = basis {
        // perm cycles 3 sites; perm_vals swaps states 0<->1, leaves 2.
        t.add_composite(
            num_complex::Complex::new(1.0, 0.0),
            &[1, 2, 0],
            vec![1, 0, 2],
            vec![0, 1, 2],
        ).unwrap();
    } else {
        panic!("expected Trit variant");
    }
}
```

Mirror in `quat.rs` (4 sites, lhss=4) and `dit.rs` (3 sites, lhss=5).

**Step 2: Run, expect fail.** `cargo test -p quspin-basis -- add_composite_` — compile error.

**Step 3: Implement.** For each of `trit.rs`, `quat.rs`, `dit.rs`:

Add `add_composite` on the inner `*Default` enum (and `*LargeInt` when feature
is on). Body mirrors Task 1's `BitBasisDefault::add_composite` but:
- No `[1, 0]` restriction on `perm_vals` (just the bijection check
  `validate_perm_vals(&perm_vals, lhss)` and `validate_locs`).
- Local op constructor differs:
  - Trit: `PermDitValues::<3>::new(arr_3, locs)` (use `try_into` to get `[u8; 3]`
    after `validate_perm_vals`).
  - Quat: `PermDitValues::<4>::new(arr_4, locs)`.
  - DynDit: `DynamicPermDitValues::new(self.lhss(), perm_vals, locs)`.
- The wrap into `SymElement::composite(perm, op)` is identical.

Add family-level delegation on `impl TritBasis` / `QuatBasis` / `DynDitBasis`,
identical to the bit-family pattern.

**Step 4: Run tests, expect pass.**

```sh
cargo test -p quspin-basis -- add_composite 2>&1 | tail -15
cargo check --workspace --features quspin-py/large-int 2>&1 | tail -3
```

**Step 5: Commit.**

```sh
git add crates/quspin-basis/src/dispatch/{trit,quat,dit}.rs
git commit -m "feat(basis): add Trit/Quat/DynDit add_composite for lattice+local elements"
```

---

### Task 3: `add_symmetry_raw` on the umbrella enums

**Files:**
- Modify: `crates/quspin-basis/src/dispatch.rs` (impls on `GenericBasis` and `DitBasis`)
- Test: same file's `#[cfg(test)] mod tests`

**Step 1: Write failing tests**

```rust
#[test]
fn add_symmetry_raw_routes_lattice() {
    let mut basis = GenericBasis::new(3, 2, SpaceKind::Symm, false).unwrap();
    basis.add_symmetry_raw(
        Complex::new(1.0, 0.0),
        Some(&[1, 2, 0][..]),  // perm
        None, None,            // no local op
    ).unwrap();
}

#[test]
fn add_symmetry_raw_routes_local_bit() {
    let mut basis = GenericBasis::new(3, 2, SpaceKind::Symm, false).unwrap();
    basis.add_symmetry_raw(
        Complex::new(-1.0, 0.0),
        None,
        Some(vec![1, 0]),
        Some(vec![0, 1, 2]),
    ).unwrap();
}

#[test]
fn add_symmetry_raw_routes_composite_bit() {
    let mut basis = GenericBasis::new(3, 2, SpaceKind::Symm, false).unwrap();
    basis.add_symmetry_raw(
        Complex::new(-1.0, 0.0),
        Some(&[1, 2, 0][..]),
        Some(vec![1, 0]),
        Some(vec![0, 1, 2]),
    ).unwrap();
}

#[test]
fn add_symmetry_raw_rejects_identity() {
    let mut basis = GenericBasis::new(3, 2, SpaceKind::Symm, false).unwrap();
    let r = basis.add_symmetry_raw(Complex::new(1.0, 0.0), None, None, None);
    assert!(r.is_err());
}

#[test]
fn add_symmetry_raw_routes_dit_local() {
    let mut basis = GenericBasis::new(3, 3, SpaceKind::Symm, false).unwrap();
    basis.add_symmetry_raw(
        Complex::new(1.0, 0.0),
        None,
        Some(vec![1, 0, 2]),
        Some(vec![0, 1, 2]),
    ).unwrap();
}
```

**Step 2: Run, expect fail.**
`cargo test -p quspin-basis dispatch::tests::add_symmetry_raw_ -- --exact` — compile error.

**Step 3: Implement.**

In `dispatch.rs`, add to `impl GenericBasis`:

```rust
/// Add a symmetry element by its untyped triple. Routes to add_lattice /
/// add_local / add_composite per the (perm, perm_vals) shape; identity
/// (both None) returns an error.
pub fn add_symmetry_raw(
    &mut self,
    grp_char: Complex<f64>,
    perm: Option<&[usize]>,
    perm_vals: Option<Vec<u8>>,
    locs: Option<Vec<usize>>,
) -> Result<(), QuSpinError> {
    let n_sites = self.n_sites();
    let locs = locs.unwrap_or_else(|| (0..n_sites).collect());
    match (perm, perm_vals) {
        (Some(p), None) => self.add_lattice(grp_char, p.to_vec()),
        (None, Some(v)) => self.add_local(grp_char, v, locs),
        (Some(p), Some(v)) => match self {
            Self::Bit(b) => b.add_composite(grp_char, p, v, locs),
            Self::Dit(b) => b.add_composite(grp_char, p, v, locs),
        },
        (None, None) => Err(QuSpinError::ValueError(
            "add_symmetry_raw: empty element (identity is implicit)".into(),
        )),
    }
}
```

Add the analogous method on `impl DitBasis` (routes to its three family
arms' `add_composite`).

**Step 4: Run, expect pass.**
`cargo test -p quspin-basis dispatch::tests -- add_symmetry_raw 2>&1 | tail -10`
Expected: 5 PASS.

**Step 5: Commit.**

```sh
git add crates/quspin-basis/src/dispatch.rs
git commit -m "feat(basis): add_symmetry_raw entry on GenericBasis/DitBasis"
```

---

## Phase 2 — PyO3 element handle

### Task 4: `PySymElement` class + `Lattice` / `Local` / `Composite` constructors

**Files:**
- Create: `crates/quspin-py/src/basis/sym_element.rs`
- Modify: `crates/quspin-py/src/basis/mod.rs` (add `pub mod sym_element;` + re-exports)
- Modify: `crates/quspin-py/src/lib.rs` (register `Lattice`/`Local`/`Composite`/`SymElement` on the `_rs` module)
- Test: `python/tests/test_symmetry_group.py` (new file — adds the first three tests below)

**Step 1: Write failing tests (Python)**

```python
# python/tests/test_symmetry_group.py
import pytest
from quspin_rs import Lattice, Local, Composite, SymElement


class TestSymElementConstructors:
    def test_lattice_repr_roundtrip(self):
        a = Lattice([1, 2, 0])
        b = Lattice([1, 2, 0])
        assert a == b
        assert hash(a) == hash(b)
        assert "Lattice" in repr(a)

    def test_local_default_locs_is_none(self):
        a = Local([1, 0])
        b = Local([1, 0], locs=None)
        assert a == b

    def test_local_explicit_locs(self):
        a = Local([1, 0], locs=[0, 2])
        assert a != Local([1, 0])  # default locs vs explicit aren't equal
        assert "Local" in repr(a)

    def test_composite_repr(self):
        c = Composite([2, 1, 0], [1, 0])
        assert "Composite" in repr(c)

    def test_lattice_rejects_negative_int_with_hint(self):
        with pytest.raises(ValueError, match="Composite"):
            Lattice([-1, 0, 1])

    def test_isinstance_symelement(self):
        assert isinstance(Lattice([0]), SymElement)
        assert isinstance(Local([1, 0]), SymElement)
        assert isinstance(Composite([0], [1, 0]), SymElement)
```

**Step 2: Run, expect fail.**
```sh
just develop && uv run pytest python/tests/test_symmetry_group.py -v 2>&1 | tail -10
```
Expected: import error — `Lattice` etc. don't exist.

**Step 3: Implement `PySymElement` and constructors.**

Create `crates/quspin-py/src/basis/sym_element.rs`:

```rust
//! PyO3 element handle for SymmetryGroup. LHSS-agnostic — holds the
//! untyped triple (perm, perm_vals, locs); LHSS-specific construction
//! happens at *Basis.symmetric(...) time inside the dispatch enum.

use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum SymElementKind { Lattice, Local, Composite }

#[pyclass(name = "SymElement", module = "quspin_rs._rs", frozen)]
#[derive(Clone)]
pub struct PySymElement {
    pub(crate) kind: SymElementKind,
    pub(crate) perm: Option<Vec<usize>>,
    pub(crate) perm_vals: Option<Vec<u8>>,
    pub(crate) locs: Option<Vec<usize>>,
}

#[pymethods]
impl PySymElement {
    fn __repr__(&self) -> String {
        match self.kind {
            SymElementKind::Lattice => format!("Lattice(perm={:?})", self.perm.as_ref().unwrap()),
            SymElementKind::Local => format!(
                "Local(perm_vals={:?}, locs={:?})",
                self.perm_vals.as_ref().unwrap(),
                self.locs,
            ),
            SymElementKind::Composite => format!(
                "Composite(perm={:?}, perm_vals={:?}, locs={:?})",
                self.perm.as_ref().unwrap(),
                self.perm_vals.as_ref().unwrap(),
                self.locs,
            ),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.perm == other.perm
            && self.perm_vals == other.perm_vals
            && self.locs == other.locs
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.kind.hash(&mut h);
        self.perm.hash(&mut h);
        self.perm_vals.hash(&mut h);
        self.locs.hash(&mut h);
        h.finish()
    }
}

#[pyfunction]
#[pyo3(signature = (perm))]
pub fn lattice(perm: Vec<i64>) -> PyResult<PySymElement> {
    if perm.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err(
            "Lattice perm contains negative integers — old-QuSpin used \
             negative ints to encode spin-flip+permutation; use \
             Composite(perm, perm_vals=[1, 0]) instead.",
        ));
    }
    let perm: Vec<usize> = perm.into_iter().map(|v| v as usize).collect();
    Ok(PySymElement {
        kind: SymElementKind::Lattice,
        perm: Some(perm),
        perm_vals: None,
        locs: None,
    })
}

#[pyfunction]
#[pyo3(signature = (perm_vals, locs = None))]
pub fn local(perm_vals: Vec<u64>, locs: Option<Vec<usize>>) -> PyResult<PySymElement> {
    let perm_vals: Vec<u8> = perm_vals
        .into_iter()
        .map(|v| u8::try_from(v).map_err(|_| PyValueError::new_err("perm_vals values must fit in u8")))
        .collect::<PyResult<_>>()?;
    Ok(PySymElement {
        kind: SymElementKind::Local,
        perm: None,
        perm_vals: Some(perm_vals),
        locs,
    })
}

#[pyfunction]
#[pyo3(signature = (perm, perm_vals, locs = None))]
pub fn composite(
    perm: Vec<i64>,
    perm_vals: Vec<u64>,
    locs: Option<Vec<usize>>,
) -> PyResult<PySymElement> {
    if perm.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err(
            "Composite perm contains negative integers — perm indexes sites; \
             use perm_vals to encode local-op action.",
        ));
    }
    let perm: Vec<usize> = perm.into_iter().map(|v| v as usize).collect();
    let perm_vals: Vec<u8> = perm_vals
        .into_iter()
        .map(|v| u8::try_from(v).map_err(|_| PyValueError::new_err("perm_vals values must fit in u8")))
        .collect::<PyResult<_>>()?;
    Ok(PySymElement {
        kind: SymElementKind::Composite,
        perm: Some(perm),
        perm_vals: Some(perm_vals),
        locs,
    })
}
```

In `crates/quspin-py/src/basis/mod.rs`:

```rust
pub mod sym_element;
pub use sym_element::PySymElement;
```

In `crates/quspin-py/src/lib.rs`, inside the `#[pymodule]` body:

```rust
m.add_class::<crate::basis::sym_element::PySymElement>()?;
m.add_function(wrap_pyfunction!(crate::basis::sym_element::lattice, m)?)?;
m.add_function(wrap_pyfunction!(crate::basis::sym_element::local, m)?)?;
m.add_function(wrap_pyfunction!(crate::basis::sym_element::composite, m)?)?;
```

(Capitalised wrappers — the Python-facing names need to be `Lattice`/`Local`/`Composite`, not lowercase. Use `#[pyfunction(name = "Lattice")]` etc. on the Rust functions, or wrap on the Python side. Prefer the PyO3 attribute.)

**Step 4: Run tests, expect pass.**

```sh
just develop && uv run pytest python/tests/test_symmetry_group.py::TestSymElementConstructors -v 2>&1 | tail -10
```

**Step 5: Commit.**

```sh
git add crates/quspin-py/src/basis/sym_element.rs \
        crates/quspin-py/src/basis/mod.rs \
        crates/quspin-py/src/lib.rs \
        python/tests/test_symmetry_group.py
git commit -m "feat(py): SymElement PyO3 class + Lattice/Local/Composite constructors"
```

---

### Task 5: `_order` helper

**Files:**
- Modify: `crates/quspin-py/src/basis/sym_element.rs`
- Test: `python/tests/test_symmetry_group.py` (add `TestOrder`)

**Step 1: Write failing tests**

```python
class TestOrder:
    def test_lattice_4cycle(self):
        from quspin_rs._rs import _order
        assert _order(Lattice([1, 2, 3, 0]), n_sites=4, lhss=2) == 4

    def test_lattice_two_2cycles(self):
        from quspin_rs._rs import _order
        assert _order(Lattice([1, 0, 3, 2]), n_sites=4, lhss=2) == 2

    def test_lattice_3cycle_plus_2cycle(self):
        from quspin_rs._rs import _order
        # sites 0->1->2->0 (3-cycle) and 3<->4 (2-cycle): order = lcm(3,2) = 6
        assert _order(Lattice([1, 2, 0, 4, 3]), n_sites=5, lhss=2) == 6

    def test_local_z2_swap(self):
        from quspin_rs._rs import _order
        assert _order(Local([1, 0]), n_sites=4, lhss=2) == 2

    def test_local_z3_cycle(self):
        from quspin_rs._rs import _order
        assert _order(Local([1, 2, 0]), n_sites=4, lhss=3) == 3

    def test_composite_lcm(self):
        from quspin_rs._rs import _order
        # perm has order 4 (4-cycle), perm_vals has order 2 -> composite order 4
        assert _order(Composite([1, 2, 3, 0], [1, 0]), n_sites=4, lhss=2) == 4

    def test_identity_order_is_one(self):
        from quspin_rs._rs import _order
        assert _order(Lattice([0, 1, 2]), n_sites=3, lhss=2) == 1
```

**Step 2: Run, expect fail.** Import error on `_order`.

**Step 3: Implement.**

In `sym_element.rs`:

```rust
/// Compute the order of a permutation `p` over `0..N` as LCM of cycle lengths.
fn perm_order(p: &[usize]) -> usize {
    let n = p.len();
    let mut visited = vec![false; n];
    let mut lcm: usize = 1;
    for start in 0..n {
        if visited[start] { continue; }
        let mut len = 0usize;
        let mut i = start;
        while !visited[i] {
            visited[i] = true;
            i = p[i];
            len += 1;
        }
        if len > 1 { lcm = lcm_u(lcm, len); }
    }
    lcm
}

fn lcm_u(a: usize, b: usize) -> usize { a / gcd_u(a, b) * b }

fn gcd_u(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let t = a % b; a = b; b = t; }
    a
}

#[pyfunction]
pub fn _order(elem: &PySymElement, n_sites: usize, lhss: usize) -> usize {
    let _ = n_sites;
    let perm_o = elem.perm.as_deref().map(perm_order).unwrap_or(1);
    let pv_o = elem.perm_vals.as_deref().map(|v| {
        let v_us: Vec<usize> = v.iter().map(|&x| x as usize).collect();
        perm_order(&v_us)
    }).unwrap_or(1);
    let _ = lhss; // length of perm_vals already encodes lhss
    lcm_u(perm_o, pv_o)
}
```

Register in `lib.rs`: `m.add_function(wrap_pyfunction!(_order, m)?)?;`.

**Step 4: Run, expect pass.**
```sh
just develop && uv run pytest python/tests/test_symmetry_group.py::TestOrder -v 2>&1 | tail -10
```

**Step 5: Commit.**
```sh
git commit -am "feat(py): _order helper for SymElement (LCM of cycle lengths)"
```

---

### Task 6: `_compose` helper

**Files:**
- Modify: `crates/quspin-py/src/basis/sym_element.rs`
- Test: add `TestCompose` to `python/tests/test_symmetry_group.py`

**Step 1: Write failing tests**

```python
class TestCompose:
    def test_lattice_lattice_stays_lattice(self):
        from quspin_rs._rs import _compose
        a = Lattice([1, 2, 0])  # 3-cycle
        b = Lattice([1, 2, 0])
        c = _compose(a, b)
        assert c == Lattice([2, 0, 1])  # (a∘b)[s] = a[b[s]]

    def test_local_local_stays_local(self):
        from quspin_rs._rs import _compose
        a = Local([1, 0])
        b = Local([1, 0])
        c = _compose(a, b)
        assert c == Local([0, 1])  # involution squared = identity perm_vals

    def test_lattice_local_promotes_to_composite(self):
        from quspin_rs._rs import _compose
        a = Lattice([1, 2, 0])
        b = Local([1, 0])
        c = _compose(a, b)
        assert isinstance(c, SymElement)
        # repr indicates Composite kind
        assert "Composite" in repr(c)
```

**Step 2: Run, expect fail.** Import error on `_compose`.

**Step 3: Implement.**

```rust
fn compose_perms(a: &[usize], b: &[usize]) -> Vec<usize> {
    assert_eq!(a.len(), b.len(), "perm length mismatch");
    (0..b.len()).map(|s| a[b[s]]).collect()
}

#[pyfunction]
pub fn _compose(a: &PySymElement, b: &PySymElement) -> PyResult<PySymElement> {
    let perm = match (&a.perm, &b.perm) {
        (None, None) => None,
        (Some(p), None) | (None, Some(p)) => Some(p.clone()),
        (Some(x), Some(y)) => Some(compose_perms(x, y)),
    };
    let perm_vals = match (&a.perm_vals, &b.perm_vals) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v.clone()),
        (Some(x), Some(y)) => {
            // Compose as permutations of 0..lhss; both must be the same length.
            if x.len() != y.len() {
                return Err(PyValueError::new_err(
                    "perm_vals length mismatch in compose",
                ));
            }
            let xu: Vec<usize> = x.iter().map(|&v| v as usize).collect();
            let yu: Vec<usize> = y.iter().map(|&v| v as usize).collect();
            Some(compose_perms(&xu, &yu).into_iter().map(|v| v as u8).collect())
        }
    };
    let locs = match (&a.locs, &b.locs) {
        (None, x) | (x, None) => x.clone(),
        // Both explicit: take union (sorted, dedup).
        (Some(x), Some(y)) => {
            let mut s: std::collections::BTreeSet<usize> = x.iter().copied().collect();
            s.extend(y.iter().copied());
            Some(s.into_iter().collect())
        }
    };
    let kind = match (perm.is_some(), perm_vals.is_some()) {
        (true, false) => SymElementKind::Lattice,
        (false, true) => SymElementKind::Local,
        (true, true) => SymElementKind::Composite,
        (false, false) => return Err(PyValueError::new_err("compose produced identity")),
    };
    Ok(PySymElement { kind, perm, perm_vals, locs })
}
```

Register in `lib.rs`.

**Step 4: Run, expect pass.**
```sh
just develop && uv run pytest python/tests/test_symmetry_group.py::TestCompose -v 2>&1 | tail -10
```

**Step 5: Commit.**
```sh
git commit -am "feat(py): _compose helper for SymElement"
```

---

### Task 7: `_add_to_basis` helper (PyO3 → dispatch enum bridge)

**Files:**
- Modify: `crates/quspin-py/src/basis/sym_element.rs`
- Test: defer end-to-end; smoke-test via `TestSymGroupValidate` in Phase 3.

**Step 1: Stub the test in Task 12** (validate calls this internally).

For now, write a Rust-only smoke test in `sym_element.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use quspin_core::basis::{GenericBasis, SpaceKind};

    #[test]
    fn add_to_basis_lattice() {
        let mut b = GenericBasis::new(3, 2, SpaceKind::Symm, false).unwrap();
        let elem = PySymElement {
            kind: SymElementKind::Lattice,
            perm: Some(vec![1, 2, 0]),
            perm_vals: None,
            locs: None,
        };
        elem.add_to_basis(&mut b, num_complex::Complex::new(1.0, 0.0)).unwrap();
    }
}
```

**Step 2: Run, expect fail.** Compile error — method missing.

**Step 3: Implement.**

```rust
impl PySymElement {
    /// Ship this element into a GenericBasis via add_symmetry_raw.
    /// Used by the Python SymmetryGroup at *Basis.symmetric(...) time.
    pub fn add_to_basis(
        &self,
        basis: &mut quspin_core::basis::GenericBasis,
        grp_char: num_complex::Complex<f64>,
    ) -> Result<(), quspin_core::error::QuSpinError> {
        basis.add_symmetry_raw(
            grp_char,
            self.perm.as_deref(),
            self.perm_vals.clone(),
            self.locs.clone(),
        )
    }

    /// Variant for FermionBasis (BitBasis inner).
    pub fn add_to_bit_basis(
        &self,
        basis: &mut quspin_core::basis::dispatch::BitBasis,
        grp_char: num_complex::Complex<f64>,
    ) -> Result<(), quspin_core::error::QuSpinError> {
        let n_sites = basis.n_sites();
        let locs = self.locs.clone().unwrap_or_else(|| (0..n_sites).collect());
        match (&self.perm, &self.perm_vals) {
            (Some(p), None) => basis.add_lattice(grp_char, p),
            (None, Some(v)) => basis.add_local(grp_char, v.clone(), locs),
            (Some(p), Some(v)) => basis.add_composite(grp_char, p, v.clone(), locs),
            (None, None) => Err(quspin_core::error::QuSpinError::ValueError(
                "add_to_bit_basis: empty element".into(),
            )),
        }
    }
}
```

**Step 4: Run, expect pass.** `cargo test -p quspin-py add_to_basis 2>&1 | tail -5`

(quspin-py has no standalone Rust tests by default — but this test under
`#[cfg(test)]` in a non-cdylib target should still compile via `cargo
test`. If maturin/abi3 makes that awkward, drop the Rust unit test and
rely on the Python-driven validation in Task 12.)

**Step 5: Commit.**

```sh
git commit -am "feat(py): _add_to_basis bridge from PySymElement to dispatch enums"
```

---

## Phase 3 — Pure-Python `SymmetryGroup`

### Task 8: `SymmetryGroup` skeleton (init + `add` + dunders)

**Files:**
- Create: `python/quspin_rs/symmetry.py`
- Modify: `python/quspin_rs/__init__.py` (re-export)
- Test: add `TestSymmetryGroupBasics` to `python/tests/test_symmetry_group.py`

**Step 1: Write failing tests**

```python
class TestSymmetryGroupBasics:
    def test_construct(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        assert g.n_sites == 4
        assert g.lhss == 2
        assert len(g) == 0

    def test_add_and_iter(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        T = Lattice([1, 2, 3, 0])
        g.add(T, 1.0 + 0j)
        assert len(g) == 1
        elems = list(g)
        assert elems[0][0] == T
        assert elems[0][1] == 1.0 + 0j

    def test_repr(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        assert "SymmetryGroup" in repr(g)
        assert "n_sites=4" in repr(g)
```

**Step 2: Run, expect fail.** ImportError.

**Step 3: Implement.**

`python/quspin_rs/symmetry.py`:

```python
"""User-facing SymmetryGroup for *Basis.symmetric(...)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quspin_rs._rs import SymElement


class SymmetryGroup:
    """Collection of (element, character) pairs describing a symmetry group.

    Construct with ``(n_sites, lhss)``, then add elements via :meth:`add`,
    :meth:`add_cyclic`, or :meth:`close`. Pass the result as the first
    positional argument to ``*Basis.symmetric(group, ham, seeds)``.
    """

    def __init__(self, n_sites: int, lhss: int) -> None:
        if n_sites < 1:
            raise ValueError(f"n_sites must be >= 1, got {n_sites}")
        if lhss < 2:
            raise ValueError(f"lhss must be >= 2, got {lhss}")
        self._n_sites = n_sites
        self._lhss = lhss
        self._elements: list[tuple["SymElement", complex]] = []

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def lhss(self) -> int:
        return self._lhss

    def add(self, element: "SymElement", character: complex) -> None:
        """Add a single non-identity element with its 1-D-rep character."""
        self._elements.append((element, complex(character)))

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[tuple["SymElement", complex]]:
        return iter(self._elements)

    def __repr__(self) -> str:
        return (
            f"SymmetryGroup(n_sites={self._n_sites}, lhss={self._lhss}, "
            f"|G|={1 + len(self._elements)})"
        )
```

In `python/quspin_rs/__init__.py`:

```python
from quspin_rs._rs import (
    Composite,
    Lattice,
    Local,
    SymElement,
    # ... existing imports
)
from quspin_rs.symmetry import SymmetryGroup

__all__ = [
    "Composite",
    "Lattice",
    "Local",
    "SymElement",
    "SymmetryGroup",
    # ... existing exports
]
```

**Step 4: Run, expect pass.**
```sh
uv run pytest python/tests/test_symmetry_group.py::TestSymmetryGroupBasics -v 2>&1 | tail -10
```

**Step 5: Commit.**
```sh
git add python/quspin_rs/symmetry.py python/quspin_rs/__init__.py python/tests/test_symmetry_group.py
git commit -m "feat(py): SymmetryGroup skeleton (init, add, dunders)"
```

---

### Task 9: `SymmetryGroup.add_cyclic`

**Files:**
- Modify: `python/quspin_rs/symmetry.py`
- Test: add `TestAddCyclic` to `test_symmetry_group.py`

**Step 1: Write failing tests**

```python
import cmath
from math import pi

class TestAddCyclic:
    def test_translation_k_equiv_eta_for_z2(self):
        from quspin_rs import SymmetryGroup
        # Z_2 lattice generator (swap)
        g_k = SymmetryGroup(n_sites=2, lhss=2)
        g_k.add_cyclic(Lattice([1, 0]), k=1)
        g_eta = SymmetryGroup(n_sites=2, lhss=2)
        g_eta.add_cyclic(Lattice([1, 0]), eta=-1)
        assert list(g_k) == list(g_eta)

    def test_translation_k1_z4(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        g.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        assert len(g) == 3  # T, T², T³
        # Characters: ω = exp(-2πi/4)
        omega = cmath.exp(-2j * pi / 4)
        for (_, chi), expected in zip(g, [omega, omega**2, omega**3]):
            assert abs(chi - expected) < 1e-12

    def test_eta_only_for_order_2(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        with pytest.raises(ValueError, match="order"):
            g.add_cyclic(Lattice([1, 2, 3, 0]), eta=-1)

    def test_exactly_one_of_k_eta_char(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        with pytest.raises(ValueError):
            g.add_cyclic(Lattice([1, 2, 3, 0]))
        with pytest.raises(ValueError):
            g.add_cyclic(Lattice([1, 2, 3, 0]), k=1, eta=1)

    def test_identity_generator_rejected(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=3, lhss=2)
        with pytest.raises(ValueError):
            g.add_cyclic(Lattice([0, 1, 2]), k=0)

    def test_k_out_of_range(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        with pytest.raises(ValueError):
            g.add_cyclic(Lattice([1, 2, 3, 0]), k=5)
```

**Step 2: Run, expect fail.** Method missing.

**Step 3: Implement.**

In `symmetry.py`:

```python
import cmath
from math import pi

from quspin_rs._rs import _compose, _order


class SymmetryGroup:
    # ... existing __init__, add, __len__, etc. ...

    def add_cyclic(
        self,
        generator: "SymElement",
        *,
        k: int | None = None,
        eta: int | None = None,
        char: complex | None = None,
    ) -> None:
        """Add g, g², …, g^(N-1) where N is the generator's computed order.

        Exactly one of ``{k, eta, char}`` must be supplied:

        - ``k=int``    χ(g^a) = exp(-2πi · k · a / N), any cyclic
        - ``eta=±1``   χ(g^a) = η^a, requires N == 2
        - ``char=z``   χ(g^a) = z^a (user picks any consistent rep)
        """
        supplied = sum(x is not None for x in (k, eta, char))
        if supplied != 1:
            raise ValueError(
                "add_cyclic requires exactly one of {k, eta, char}, "
                f"got {supplied}"
            )

        order = _order(generator, self._n_sites, self._lhss)
        if order < 2:
            raise ValueError(
                "add_cyclic generator has order < 2 (identity element)"
            )
        if eta is not None:
            if order != 2:
                raise ValueError(
                    f"eta=±1 requires order 2, got {order}"
                )
            if eta not in (1, -1):
                raise ValueError(f"eta must be ±1, got {eta}")
            base_char: complex = complex(eta)
        elif k is not None:
            if not (0 <= k < order):
                raise ValueError(f"k must be in [0, {order}), got {k}")
            base_char = cmath.exp(-2j * pi * k / order)
        else:
            assert char is not None
            base_char = complex(char)

        # Enumerate g, g², …, g^(N-1) via repeated _compose.
        g_pow: "SymElement" = generator
        for a in range(1, order):
            self._elements.append((g_pow, base_char ** a))
            if a + 1 < order:
                g_pow = _compose(g_pow, generator)
```

**Step 4: Run, expect pass.**
```sh
uv run pytest python/tests/test_symmetry_group.py::TestAddCyclic -v 2>&1 | tail -10
```

**Step 5: Commit.**
```sh
git commit -am "feat(py): SymmetryGroup.add_cyclic with k/eta/char selectors"
```

---

### Task 10: `SymmetryGroup.close`

**Files:**
- Modify: `python/quspin_rs/symmetry.py`
- Test: add `TestClose` to `test_symmetry_group.py`

**Step 1: Write failing tests**

```python
class TestClose:
    def test_close_dihedral_d4_trivial_rep(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        T = Lattice([1, 2, 3, 0])
        P = Lattice([3, 2, 1, 0])
        g.close(generators=[T, P], char=lambda elem: 1.0)
        # D_4 has 2*4 = 8 elements; 7 non-identity.
        assert len(g) == 7

    def test_close_just_T_z4(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        T = Lattice([1, 2, 3, 0])
        g.close(generators=[T], char=lambda elem: 1.0)
        assert len(g) == 3  # T, T², T³

    def test_close_empty_generators_no_op(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        g.close(generators=[], char=lambda elem: 1.0)
        assert len(g) == 0
```

**Step 2: Run, expect fail.**

**Step 3: Implement.**

```python
def close(
    self,
    generators: list["SymElement"],
    char: Callable[["SymElement"], complex],
) -> None:
    """BFS-close orbit under composition. ``char`` is called for each
    enumerated non-identity element. Caller is responsible for
    supplying a self-consistent 1-D rep — :meth:`validate` catches
    inconsistencies."""

    if not generators:
        return

    # BFS from generators; key elements by their (perm, perm_vals, locs)
    # via _compose's identity reduction. The set tracks which actions
    # we've already added so we don't double-count.
    seen: set["SymElement"] = set()
    frontier: list["SymElement"] = []
    for g in generators:
        if g not in seen:
            seen.add(g)
            frontier.append(g)
            self._elements.append((g, complex(char(g))))

    while frontier:
        next_frontier: list["SymElement"] = []
        for x in frontier:
            for g in generators:
                try:
                    composed = _compose(x, g)
                except ValueError:
                    # composition produced identity — skip
                    continue
                if composed in seen:
                    continue
                seen.add(composed)
                next_frontier.append(composed)
                self._elements.append((composed, complex(char(composed))))
        frontier = next_frontier
```

**Step 4: Run, expect pass.**

**Step 5: Commit.**
```sh
git commit -am "feat(py): SymmetryGroup.close BFS over generators"
```

---

### Task 11: `SymmetryGroup.product`

**Files:**
- Modify: `python/quspin_rs/symmetry.py`
- Test: add `TestProduct` to `test_symmetry_group.py`

**Step 1: Write failing tests**

```python
class TestProduct:
    def test_z4_x_z2_size(self):
        from quspin_rs import SymmetryGroup
        T = SymmetryGroup(n_sites=4, lhss=2)
        T.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        Z = SymmetryGroup(n_sites=4, lhss=2)
        Z.add_cyclic(Local([1, 0]), eta=-1)
        G = T.product(Z)
        # 4·2 - 1 = 7 non-identity elements
        assert len(G) == 7

    def test_product_does_not_mutate(self):
        from quspin_rs import SymmetryGroup
        T = SymmetryGroup(n_sites=4, lhss=2)
        T.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        n_before = len(T)
        Z = SymmetryGroup(n_sites=4, lhss=2)
        Z.add_cyclic(Local([1, 0]), eta=-1)
        _ = T.product(Z)
        assert len(T) == n_before  # T unchanged

    def test_product_lhss_mismatch_raises(self):
        from quspin_rs import SymmetryGroup
        a = SymmetryGroup(n_sites=4, lhss=2)
        b = SymmetryGroup(n_sites=4, lhss=3)
        with pytest.raises(ValueError):
            a.product(b)
```

**Step 2: Run, expect fail.**

**Step 3: Implement.**

```python
def product(self, other: "SymmetryGroup") -> "SymmetryGroup":
    """Out-of-place direct product. Both groups must share
    (n_sites, lhss). Caller asserts the factors commute;
    :meth:`validate` catches non-commutation at first build."""
    if (self._n_sites, self._lhss) != (other._n_sites, other._lhss):
        raise ValueError(
            f"product: factor groups must share (n_sites, lhss); "
            f"got {(self._n_sites, self._lhss)} vs {(other._n_sites, other._lhss)}"
        )
    out = SymmetryGroup(self._n_sites, self._lhss)
    # Self-only
    for elem, chi in self._elements:
        out._elements.append((elem, chi))
    # Other-only
    for elem, chi in other._elements:
        out._elements.append((elem, chi))
    # Cross terms
    for a_elem, chi_a in self._elements:
        for b_elem, chi_b in other._elements:
            try:
                composed = _compose(a_elem, b_elem)
            except ValueError:
                continue  # identity — skip
            out._elements.append((composed, chi_a * chi_b))
    return out
```

**Step 4: Run, expect pass.**

**Step 5: Commit.**
```sh
git commit -am "feat(py): SymmetryGroup.product (out-of-place)"
```

---

### Task 12: `SymmetryGroup.validate`

**Files:**
- Modify: `python/quspin_rs/symmetry.py`
- Modify: `crates/quspin-py/src/basis/sym_element.rs` — add `_validate_group(elements, n_sites, lhss)` `#[pyfunction]`
- Test: add `TestValidate` to `test_symmetry_group.py`

**Step 1: Write failing tests**

```python
class TestValidate:
    def test_validate_clean_group(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        g.add_cyclic(Lattice([1, 2, 3, 0]), k=1)
        g.validate()  # should not raise

    def test_validate_rejects_missing_closure(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        # Add T but not T² / T³ — closure broken.
        g.add(Lattice([1, 2, 3, 0]), 1.0 + 0j)
        with pytest.raises(ValueError, match="closed|closure"):
            g.validate()

    def test_validate_rejects_inconsistent_chars(self):
        from quspin_rs import SymmetryGroup
        g = SymmetryGroup(n_sites=4, lhss=2)
        # Cyclic group with wrong chars (not a 1-D rep).
        T = Lattice([1, 2, 3, 0])
        g.add(T, -1.0 + 0j)            # χ(T)
        # composes by hand…
        from quspin_rs._rs import _compose
        T2 = _compose(T, T)
        g.add(T2, 2.0 + 0j)            # χ(T²) should be (−1)² = 1, not 2
        T3 = _compose(T2, T)
        g.add(T3, -1.0 + 0j)            # χ(T³)
        with pytest.raises(ValueError, match="character"):
            g.validate()
```

**Step 2: Run, expect fail.**

**Step 3: Implement.**

In `sym_element.rs`, add:

```rust
#[pyfunction]
pub fn _validate_group(
    elements: Vec<(PySymElement, (f64, f64))>,
    n_sites: usize,
    lhss: usize,
) -> PyResult<()> {
    use quspin_core::basis::{GenericBasis, SpaceKind};
    use quspin_core::error::QuSpinError;
    use crate::error::Error;

    let fermionic = false;  // validate_group doesn't depend on fermion sign
    let mut basis = GenericBasis::new(n_sites, lhss, SpaceKind::Symm, fermionic)
        .map_err(|e: QuSpinError| Error::from(e))?;
    for (elem, (re, im)) in &elements {
        elem.add_to_basis(&mut basis, num_complex::Complex::new(*re, *im))
            .map_err(|e| Error::from(e))?;
    }
    // SymBasis::validate_group runs implicitly on first build with seeds=[].
    // Use a no-op build with a no-neighbours graph.
    struct NoNeighbors { lhss: usize }
    impl quspin_bitbasis::StateTransitions for NoNeighbors {
        fn lhss(&self) -> usize { self.lhss }
        fn neighbors<B: quspin_bitbasis::BitInt, F: FnMut(num_complex::Complex<f64>, B)>(
            &self, _state: B, _visit: F,
        ) {}
    }
    basis.build(&NoNeighbors { lhss }, &[])
        .map_err(|e| Error::from(e))?;
    Ok(())
}
```

(Note: `add_to_basis` from Task 7 takes `Complex`, so we pass `(re, im)` from
the Python tuple form. The Python side flattens the iter into this shape.)

Register `_validate_group` in `lib.rs`.

In `symmetry.py`:

```python
from quspin_rs._rs import _validate_group


class SymmetryGroup:
    # ... existing methods ...

    def validate(self) -> None:
        """Eagerly run SymBasis::validate_group: closure + 1-D character check."""
        elements_for_rust = [
            (elem, (chi.real, chi.imag)) for elem, chi in self._elements
        ]
        _validate_group(elements_for_rust, self._n_sites, self._lhss)
```

**Step 4: Run, expect pass.**

**Step 5: Commit.**
```sh
git commit -am "feat(py): SymmetryGroup.validate via Rust _validate_group bridge"
```

---

## Phase 4 — Migrate `*Basis.symmetric(...)` constructors

### Task 13: Update Rust PyO3 `symmetric` constructors

**Files:**
- Modify: `crates/quspin-py/src/basis/spin.rs`
- Modify: `crates/quspin-py/src/basis/fermion.rs`
- Modify: `crates/quspin-py/src/basis/boson.rs`
- Modify: `crates/quspin-py/src/basis/generic.rs`
- Modify: `crates/quspin-py/src/basis/mod.rs` (delete `apply_symmetries` helper if unused after this task — keep for now, delete in Task 15)
- Test: handled in Tasks 16-17 (Python integration).

**Step 1: Write end-to-end Python test**

Add to `test_symmetry_group.py`:

```python
class TestBasisSymmetricEndToEnd:
    def test_spin_symmetric_translation(self):
        from quspin_rs import PauliOperator, SpinBasis, SymmetryGroup

        n_sites = 4
        # XX chain
        bonds = [[1.0, i, (i + 1) % n_sites] for i in range(n_sites)]
        H = PauliOperator([("XX", bonds)])

        group = SymmetryGroup(n_sites=n_sites, lhss=2)
        group.add_cyclic(Lattice(list(range(1, n_sites)) + [0]), k=0)

        basis = SpinBasis.symmetric(group, H, ["0000"])
        assert basis.is_built
        assert basis.size > 0

    def test_fermion_symmetric_rejects_lhss_neq_2(self):
        from quspin_rs import FermionBasis, FermionOperator, SymmetryGroup

        group = SymmetryGroup(n_sites=4, lhss=3)  # bad
        H = FermionOperator([("+", [[1.0, 0]])])  # placeholder
        with pytest.raises(TypeError, match="lhss"):
            FermionBasis.symmetric(group, H, ["0000"])
```

**Step 2: Run, expect fail.** Old `symmetric(...)` signature doesn't accept a SymmetryGroup.

**Step 3: Update each `*Basis::symmetric` constructor.**

Pattern (apply to all four `crates/quspin-py/src/basis/{spin,fermion,boson,generic}.rs`):

```rust
#[classmethod]
#[pyo3(signature = (group, ham, seeds))]
fn symmetric(
    _cls: &Bound<'_, PyType>,
    py: Python<'_>,
    group: &Bound<'_, PyAny>,
    ham: &Bound<'_, PyAny>,  // or concrete Operator type
    seeds: Vec<String>,
) -> PyResult<Self> {
    // Pull n_sites and lhss from the group.
    let n_sites: usize = group.getattr("n_sites")?.extract()?;
    let lhss: usize = group.getattr("lhss")?.extract()?;

    // Compatibility check (FermionBasis only):
    // if lhss != 2 { return Err(PyTypeError::new_err(...)); }

    let byte_seeds = parse_seeds(&seeds, lhss)?;
    let mut basis = SpinBasis::new(n_sites, lhss, SpaceKind::Symm)
        .map_err(Error::from)?;

    // Iterate (element, character) tuples from group.
    for item in group.try_iter()? {
        let item = item?;
        let tup = item.downcast::<pyo3::types::PyTuple>()?;
        let elem: PyRef<PySymElement> =
            tup.get_item(0)?.downcast::<PySymElement>()?.borrow();
        let chi: Complex<f64> = tup.get_item(1)?.extract()?;
        elem.add_to_basis(&mut basis.inner, chi).map_err(Error::from)?;
    }

    build_spin_basis(&mut basis, ham, &byte_seeds)?;
    Ok(PySpinBasis { inner: basis })
}
```

For `FermionBasis::symmetric`, use `add_to_bit_basis` (since
`FermionBasis::inner` is `BitBasis`) and add the `lhss != 2` check.

For `GenericBasis::symmetric`, no `lhss` constraint beyond what
`GenericBasis::new` enforces.

**Step 4: Run, expect pass.**

```sh
just develop && uv run pytest python/tests/test_symmetry_group.py::TestBasisSymmetricEndToEnd -v 2>&1 | tail -15
```

**Step 5: Commit.**
```sh
git commit -am "feat(py): *Basis.symmetric(group, ham, seeds) — single SymmetryGroup arg"
```

---

### Task 14: Delete old `apply_symmetries` / `apply_local_symmetries` helpers

**Files:**
- Modify: `crates/quspin-py/src/basis/mod.rs` (remove `apply_symmetries`)
- Modify: `crates/quspin-py/src/basis/generic.rs` (remove `apply_local_symmetries`)

**Step 1: Verify no remaining callers.**

```sh
grep -rn "apply_symmetries\|apply_local_symmetries" crates/quspin-py/
```
Expected: only the definitions and the imports in the migrated `symmetric`
methods (which after Task 13 should not call them anymore).

**Step 2: Delete.**

Remove the function bodies and any unused imports in
`crates/quspin-py/src/basis/mod.rs` and
`crates/quspin-py/src/basis/generic.rs`.

**Step 3: Verify clean compile.**

```sh
cargo check --workspace 2>&1 | tail -5
```

**Step 4: Run all tests.**

```sh
cargo test --workspace 2>&1 | grep -E "test result|FAIL" | tail -20
just develop && uv run pytest python/tests/test_symmetry_group.py -v 2>&1 | tail -10
```

**Step 5: Commit.**
```sh
git commit -am "refactor(py): drop apply_symmetries / apply_local_symmetries helpers"
```

---

### Task 15: Migrate existing test files to the new API

**Files:**
- Modify: `python/tests/test_rs.py` — every `*Basis.symmetric(..., symmetries=[...])` call
- Modify: `python/tests/test_monomial_generic.py` — same

**Step 1: Find call sites.**

```sh
grep -rn "symmetries=\[" python/tests/
```

**Step 2: Rewrite each call site.**

For each call site, replace:

```python
# OLD
basis = SpinBasis.symmetric(
    n_sites, ham, seeds,
    symmetries=[(perm, (re, im))],
)
```

with:

```python
# NEW
group = SymmetryGroup(n_sites=n_sites, lhss=2)
group.add(Lattice(perm), complex(re, im))
basis = SpinBasis.symmetric(group, ham, seeds)
```

For tests using cyclic-translation groups, prefer `add_cyclic(..., k=...)` —
clearer than enumerating powers manually.

**Step 3: Run.**

```sh
just develop && uv run pytest python/tests/ -v 2>&1 | grep -E "test result|FAIL" | tail -10
```

**Step 4: Iterate** until all tests pass.

**Step 5: Commit.**
```sh
git commit -am "test: migrate existing pytest call sites to SymmetryGroup API"
```

---

### Task 16: Update type stubs

**Files:**
- Modify: `python/quspin_rs/_rs.pyi`

**Step 1:** Read existing stubs, identify the `*Basis.symmetric` signatures
and any helpers from the loose tuple API.

**Step 2:** Update the stubs:

- Add new entries for `SymElement` (opaque class, `__repr__` / `__eq__` / `__hash__`),
  `Lattice(perm: list[int]) -> SymElement`,
  `Local(perm_vals: list[int], locs: list[int] | None = None) -> SymElement`,
  `Composite(perm: list[int], perm_vals: list[int], locs: list[int] | None = None) -> SymElement`.
- Update each `*Basis.symmetric` to:
  ```python
  @classmethod
  def symmetric(cls, group: SymmetryGroup, ham: ..., seeds: list[str]) -> ...: ...
  ```
- Drop the old `symmetries=…` / `local_symmetries=…` parameters everywhere.
- Add `class SymmetryGroup` with init + methods (or import-only since the
  class is defined in `quspin_rs.symmetry`, not the `.pyi` for `_rs`).

**Step 3:** Run pyright (via pre-commit) to verify the stubs compile:

```sh
pre-commit run pyright-check --all-files 2>&1 | tail -20
```

**Step 4:** Run pytest to confirm no runtime regressions.

**Step 5: Commit.**
```sh
git commit -am "docs(py): update _rs.pyi for SymmetryGroup + SymElement API"
```

---

## Phase 5 — Polish

### Task 17: CLAUDE.md note + cross-reference

**Files:**
- Modify: `CLAUDE.md`

**Step 1:** Skim `CLAUDE.md` to find the right section. Probably under
"Python package" or near the symmetry-group memory pointer.

**Step 2:** Add a one-paragraph note describing the new Python API and
linking to the spec:

```markdown
- **Symmetry-group construction (Python).** `SymmetryGroup(n_sites, lhss)`
  in `quspin_rs.symmetry` is the user-facing handle. Construct elements via
  `Lattice(perm)`, `Local(perm_vals, locs=None)`, `Composite(perm, perm_vals,
  locs=None)`. Add via `.add(elem, char)`, `.add_cyclic(elem, k=…|eta=±1|char=…)`,
  `.close(generators, char=…)`, or `.product(other)`. Pass the result as the
  first positional argument to `*Basis.symmetric(group, ham, seeds)`. Spec:
  `docs/superpowers/specs/2026-04-26-python-symmetry-group-api-design.md`.
```

**Step 3 / 4:** N/A — docs change.

**Step 5: Commit.**
```sh
git commit -am "docs(claude): SymmetryGroup API note"
```

---

### Task 18: Final verification — full test suite + clippy

```sh
cargo test --workspace 2>&1 | grep -E "test result" | tail -20
cargo clippy --workspace -- -D warnings 2>&1 | tail -10
cargo clippy --workspace --features quspin-py/large-int -- -D warnings 2>&1 | tail -10
just develop && uv run pytest python/tests/ -v 2>&1 | grep -E "test result|FAIL" | tail -5
```

Push the branch and open a PR referencing the spec.

```sh
git push -u origin phil/python-symmetry-group-api
gh pr create --title "feat(py): first-class SymmetryGroup API for *Basis.symmetric(...)" --body "$(cat <<'EOF'
Implements docs/superpowers/specs/2026-04-26-python-symmetry-group-api-design.md.

## Summary
- New `SymmetryGroup` Python class + `Lattice` / `Local` / `Composite`
  element constructors. `add` / `add_cyclic` / `close` / `product` /
  `validate` methods.
- New Rust `add_symmetry_raw` + `add_composite` on dispatch enums.
- `*Basis.symmetric(group, ham, seeds)` — single positional `group` arg;
  `n_sites` / `lhss` pulled from group.
- Old `symmetries=…` / `local_symmetries=…` kwargs removed.

## Test plan
- [x] cargo test --workspace
- [x] cargo clippy (default + large-int)
- [x] pytest python/tests/

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Remember

- Each task is one TDD cycle: failing test → run → implement → run → commit.
- Frequent commits — one per task above is the granularity.
- DRY across the four basis wrappers (`spin.rs`/`fermion.rs`/`boson.rs`/`generic.rs`):
  if the migration in Task 13 starts duplicating logic, factor a `with_group`
  helper in `basis/mod.rs` that pulls `n_sites`/`lhss` and iterates
  elements once.
- YAGNI on `__mul__` / `__pow__` on `SymElement`. Defer per the spec's
  "Future work" section.
