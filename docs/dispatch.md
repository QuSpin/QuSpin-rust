# Dispatch flow: from NumPy arrays to type-erased Rust objects

This document traces how a Python call to `PyQMatrix.dot(coeff, input, output)`
travels through the two-crate architecture, and how `PyQMatrix.build_hardcore_hamiltonian`
assembles a matrix from NumPy dtype and basis information.

---

## 1. The problem: generic Rust at a Python boundary

The core algorithms in `quspin-core` are generic over several independent type
parameters that must be chosen at compile time:

| Parameter | Role | Possible types |
|-----------|------|----------------|
| `B: BitInt` | Basis state integer (one bit or dit per site) | `u32`, `u64`, `Uint<128,2>`, … `Uint<8192,128>` |
| `V: Primitive` | Matrix element value | `i8`, `i16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>` |
| `I: Index` | CSR row/column index | `i64` (fixed at the FFI boundary) |
| `C: CIndex` | Operator-string coefficient index | `u8`, `u16` |

Python cannot express these type parameters: a NumPy array has a dtype that is
only known at runtime, and the number of sites is also only known at runtime.
The library bridges this gap with **type-erased enums** and **dispatch macros**.

---

## 2. Type erasure: the `*Inner` enums

Each generic type has a corresponding enum that holds one of the concrete
instantiations. The enum replaces the type parameter with a runtime tag.

### `HardcoreHamiltonianInner` — erases `C`

```
HardcoreHamiltonianInner::Ham8  →  HardcoreHamiltonian<u8>   (≤255 cindices / sites)
HardcoreHamiltonianInner::Ham16 →  HardcoreHamiltonian<u16>  (larger)
```

The variant is chosen eagerly in `PyHardcoreHamiltonian::new` based on the
maximum `cindex` and site index seen in the `terms` list.

### `SymmetryGrpInner` — erases `B` for symmetry groups

```
SymmetryGrpInner::Sym32   →  SymmetryGrp<u32>
SymmetryGrpInner::Sym64   →  SymmetryGrp<u64>
SymmetryGrpInner::Sym128  →  SymmetryGrp<Uint<128,2>>
…
SymmetryGrpInner::Sym8192 →  SymmetryGrp<Uint<8192,128>>
```

Chosen eagerly in `PySymmetryGrp::new` from `n_sites` (≤32 → `u32`, ≤64 →
`u64`, then powers-of-two up to 8192).

### `HardcoreSpaceInner` — erases `B` for basis spaces

```
HardcoreSpaceInner::Full32   →  FullSpace<u32>
HardcoreSpaceInner::Full64   →  FullSpace<u64>
HardcoreSpaceInner::Sub32    →  Subspace<u32>
…
HardcoreSpaceInner::Sub8192  →  Subspace<Uint<8192,128>>
HardcoreSpaceInner::Sym32    →  SymmetricSubspace<u32>
…
HardcoreSpaceInner::Sym8192  →  SymmetricSubspace<Uint<8192,128>>
```

20 variants total: 2 Full (small `n_sites` only) + 9 Sub + 9 Sym.

### `QMatrixInner` — erases `V` and `C` together

```
QMatrixInner::QMf64U8   →  QMatrix<f64,  i64, u8>
QMatrixInner::QMf64U16  →  QMatrix<f64,  i64, u16>
QMatrixInner::QMc64U8   →  QMatrix<Complex<f64>, i64, u8>
…  (12 variants: 6 value types × 2 cindex types)
```

`I` is always `i64` at the FFI boundary — it is not erased.

---

## 3. Dispatch macros

Matching on an `*Inner` enum variant restores the concrete types, but the
match body must be monomorphic code that knows `B`, `V`, and `C`. Writing that
match by hand in every call site would be extremely repetitive. Instead,
`quspin-core` exports `#[macro_export]` macros that inject **local type aliases**
inside each match arm, making the body look like generic code without actually
being so:

```rust
// with_qmatrix! injects `V` and `C` and binds `mat` to the inner QMatrix ref.
with_qmatrix!(&self.inner, V, _C, mat, {
    // inside here, `V` and `mat` are concrete, monomorphic types.
    let c_slice: &[V] = c_ro.as_slice()?;
    mat.dot(overwrite, c_slice, inp_slice, out_slice)?;
});
```

The macro expands to a `match` with 12 arms, each setting `type V = …` and
`type _C = …` before entering the body block. The Rust compiler generates a
separate copy of the body for each arm — zero-overhead static dispatch.

The full set of dispatch macros exported from `quspin-core`:

| Macro | Erased types recovered | Variants covered |
|-------|------------------------|------------------|
| `with_qmatrix!` | `V`, `C` | 12 (all QMatrixInner) |
| `with_basis!` | `B` | 20 (all HardcoreSpaceInner) |
| `with_plain_basis!` | `B` | 11 (Full* + Sub*) |
| `with_sym_basis!` | `B` | 9 (Sym* only) |
| `with_sym_grp!` | `B` | 9 (all SymmetryGrpInner) |

`quspin-py` adds two more for the Python side:

| Macro | Erased types recovered | Source |
|-------|------------------------|--------|
| `with_value_dtype!` | `V` from `MatrixDType` enum | `quspin-py::macros` |
| `with_cindex_dtype!` | `C` from `CIndexDType` enum | `quspin-py::macros` |

---

## 4. Build flow: `PyQMatrix::build_hardcore_hamiltonian`

This is where all the type parameters come together for the first time.

```
Python call
───────────
PyQMatrix.build_hardcore_hamiltonian(ham, basis, dtype=np.dtype("float64"))

Step 1 — dtype → MatrixDType enum
──────────────────────────────────
MatrixDType::from_descr(py, dtype)
  inspects numpy PyArrayDescr via is_equiv_to()
  returns MatrixDType::Float64

Step 2 — MatrixDType → type alias V  (with_value_dtype!)
──────────────────────────────────────────────────────────
with_value_dtype!(v_dtype, V, {
    // type V = f64  injected here

    Step 3a — non-symmetric path (with_plain_basis!)
    ─────────────────────────────────────────────────
    with_plain_basis!(&basis.inner, B, plain_basis, {
        // type B = u32  (or u64, Uint<128,2>, …)  injected here

        Step 4 — HardcoreHamiltonianInner → concrete ham
        ──────────────────────────────────────────────────
        match &ham.inner {
            Ham8(h)  → build_from_basis::<B, V, i64, u8,  _>(h, plain_basis)
            Ham16(h) → build_from_basis::<B, V, i64, u16, _>(h, plain_basis)
        }
        // returns QMatrix<f64, i64, u8>  (or u16)

        Step 5 — QMatrix → QMatrixInner  (IntoQMatrixInner trait)
        ───────────────────────────────────────────────────────────
        .into_qmatrix_inner()
        // pattern-matches V=f64, C=u8 → QMatrixInner::QMf64U8(mat)
    })

    Step 3b — symmetric path (with_sym_basis!)
    ───────────────────────────────────────────
    with_sym_basis!(&basis.inner, B, sym_basis, {
        // same as 3a but calls build_from_symmetric(h, sym_basis)
        // which additionally applies group character scaling
    })
})

Result: PyQMatrix { inner: QMatrixInner::QMf64U8(…) }
```

The key insight is that `MatrixDType` resolves `V`, `HardcoreSpaceInner` resolves
`B`, and the `HardcoreHamiltonianInner` arm resolves `C`. All three are peeled
off in nested match expansions. By the time `build_from_basis` is called, all
four type parameters are concrete.

---

## 5. Dot flow: `PyQMatrix.dot`

After the matrix is built, `dot` recovers the concrete types going in the other
direction — from the type-erased `QMatrixInner` back to typed NumPy slices.

```
Python call
───────────
mat.dot(coeff, input, output, overwrite=True)
  coeff, input, output are PyAny (untyped at Rust boundary)

Step 1 — QMatrixInner → V and mat ref  (with_qmatrix!)
────────────────────────────────────────────────────────
with_qmatrix!(&self.inner, V, _C, mat, {
    // type V = f64  and  mat: &QMatrix<f64, i64, u8>  injected here

    Step 2 — PyAny → PyArray1<V>  (downcast)
    ─────────────────────────────────────────
    coeff.downcast::<PyArray1<V>>()   // fails with TypeError if dtype mismatch
    input.downcast::<PyArray1<V>>()
    output.downcast::<PyArray1<V>>()

    Step 3 — PyArray1<V> → &[V]
    ────────────────────────────
    c_ro.as_slice()      // &[f64]  (requires C-contiguous)
    inp_ro.as_slice()    // &[f64]
    out_slice = unsafe { out.as_slice_mut() }  // &mut [f64]

    Step 4 — typed Rust computation
    ────────────────────────────────
    mat.dot(overwrite, c_slice, inp_slice, out_slice)
    // output[row] = Σ_c coeff[c] * Σ_col M[c, row, col] * input[col]
})
```

The `downcast::<PyArray1<V>>()` call in step 2 is the critical safety check: if
the user passes an array whose NumPy dtype does not match `V`, PyO3 returns an
error before any unsafe memory access occurs. The dtype check is implicit in the
downcast — it succeeds only when the array's element size and kind match `V`.

---

## 6. The `Primitive` trait bridge

Inside `QMatrix::dot`, all arithmetic passes through `Complex<f64>` as an
intermediate representation. The `Primitive` trait provides two methods:

```rust
fn to_complex(self) -> Complex<f64>;   // widen: i8 → 0.0+0.0i, f64 → re+0.0i
fn from_complex(c: Complex<f64>) -> Self; // narrow: Complex<f64> → re as f32, etc.
```

This lets `build_from_basis` accumulate contributions into `Complex<f64>` and
then call `V::from_complex(sum)` to store the result, regardless of what `V` is.

---

## 7. Full picture

```
                  Python (quspin-py)
─────────────────────────────────────────────────────────────────────

 np.dtype("float64")
       │
       ▼
 MatrixDType::from_descr()         HardcoreHamiltonian
       │ MatrixDType::Float64              │
       │                         HardcoreHamiltonianInner
       │                          Ham8(H<u8>) │ Ham16(H<u16>)
       │                                      │
 with_value_dtype!                            │          PyHardcoreBasis
    type V = f64 ─────────────────────────────┼──────────────────┐
                                              │         HardcoreSpaceInner
                                    build path│         with_plain_basis! /
                                              │         with_sym_basis!
                                              │           type B = u32
                                              ▼
                         build_from_basis::<B=u32, V=f64, I=i64, C=u8>
                                              │
                                   QMatrix<f64, i64, u8>
                                              │
                              into_qmatrix_inner()
                                              │
─────────────────────────────────────────────┼─────────────────────
                  Stored in PyQMatrix         │
                                             ▼
                          QMatrixInner::QMf64U8(mat)

─────────────────────────────────────────────────────────────────────
 mat.dot(coeff, input, output)

       QMatrixInner::QMf64U8(mat)
              │
       with_qmatrix!
        type V = f64
        mat: &QMatrix<f64, i64, u8>
              │
       coeff.downcast::<PyArray1<f64>>()   ← TypeError if wrong dtype
       input.downcast::<PyArray1<f64>>()
       output.downcast::<PyArray1<f64>>()
              │
       c_ro.as_slice() → &[f64]
       inp_ro.as_slice() → &[f64]
       out.as_slice_mut() → &mut [f64]     ← unsafe, GIL held
              │
       mat.dot(overwrite, &[f64], &[f64], &mut [f64])
       // output[r] = Σ_c coeff[c] * Σ_col M[c,r,col] * input[col]
─────────────────────────────────────────────────────────────────────
```

---

## 8. Where each decision is made

| Decision | Where | How |
|----------|--------|-----|
| `C` (`u8` vs `u16`) | `PyHardcoreHamiltonian::new` | `max(cindex, site) > 255` check |
| `B` for symmetry group | `PySymmetryGrp::new` | `n_sites` ladder: ≤32→u32, ≤64→u64, … |
| `B` for basis | `PyHardcoreBasis::full/subspace/symmetric` | same `n_sites` ladder |
| `V` (element dtype) | `PyQMatrix::build_hardcore_hamiltonian` | `MatrixDType::from_descr` on numpy dtype |
| `I` (index type) | compile-time constant | always `i64` at FFI boundary |
| NumPy array typecheck | `PyQMatrix::dot/dot_transpose` | `PyAny::downcast::<PyArray1<V>>()` |
