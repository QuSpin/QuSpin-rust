# Rust Rewrite Plan

## Motivation

- Rust's build system has better first-class Python extension support than C++
- More modern, maintainable build toolchain long-term
- No existing downstream consumers — clean slate, no backwards compatibility constraints

## Strategy

- **Complete rewrite** — no incremental C++/Rust FFI bridging
- **Idiomatic Rust** throughout: enums instead of `std::variant`, traits instead of concepts/CRTP, `rayon` instead of OpenMP, macros + PyO3 for the dispatch layer
- **Cargo workspace** with three crates:
  - `bitbasis` — Benes network / bit manipulation primitives (standalone, crates.io candidate)
  - `quspin-core` — core library (no Python dependency)
  - `quspin-py` — PyO3 Python bindings (depends on `quspin-core`)
- **Bottom-up order**: core primitives → bitbasis → scalar/dtype → array → basis → qmatrix → dispatch + PyO3 bindings

## Key Decisions

| Topic | Decision |
|---|---|
| Translation style | Idiomatic Rust, not a literal C++ port |
| Python bindings | PyO3 |
| Rewrite order | Complete rewrite, no FFI shim |
| Existing Python consumers | Separate package, will be refactored later — out of scope for now |
| `bitbasis` layer | Rewrite in Rust; use existing C library to **generate test cases** that validate the new Rust implementation |
| Dispatch layer (macros) | Start with `macro_rules!`; upgrade to proc macros if expressiveness is insufficient |

---

## Submodule Walkthrough

Legend: ⬜ not started · 📝 described · 🦀 in progress · ✅ complete

| Submodule | C++ Location | Target Crate | Status |
|---|---|---|---|
| Bit manipulation primitives | `basis/detail/bitbasis/` | `bitbasis` | ⬜ |
| Data types | `dtype/` | `quspin-core` | ⬜ |
| Shared utilities | `detail/` | `quspin-core` | ⬜ |
| Scalar types | `scalar/` | `quspin-core` | ⬜ |
| Array types | `array/` | `quspin-core` | ⬜ |
| Quantum basis | `basis/` | `quspin-core` | ⬜ |
| Quantum sparse matrix | `qmatrix/` | `quspin-core` | ⬜ |
| Operators | `operator.hpp` | `quspin-core` | ⬜ |
| Dispatch layer + PyO3 | `src/` | `quspin-py` | ⬜ |

---

## Submodule Descriptions

### `bitbasis` — Bit Manipulation Primitives
**C++ location:** `include/quspin/basis/detail/bitbasis/`
**Target crate:** `bitbasis`
**Status:** ⬜

#### Description
TBD

#### Files
| File | Purpose |
|---|---|
| `benes.hpp` | TBD |
| `cast.hpp` | TBD |
| `dit_manip.hpp` | TBD |
| `dit_perm.hpp` | TBD |
| `info.hpp` | TBD |
| `utils.hpp` | TBD |

#### Rust Design Notes
TBD

#### Testing Strategy
Use existing C library to generate test vectors that validate the Rust implementation.

---

### `dtype/` — Data Type Abstraction
**C++ location:** `include/quspin/dtype/`, `src/dtype/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
TBD

#### Rust Design Notes
TBD

---

### `detail/` — Shared Utilities
**C++ location:** `include/quspin/detail/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
TBD

#### Files
| File | Purpose |
|---|---|
| `broadcast.hpp` | TBD |
| `cast.hpp` | TBD |
| `default_containers.hpp` | TBD |
| `dispatch.hpp` | TBD |
| `error.hpp` | TBD |
| `math.hpp` | TBD |
| `omp.hpp` | TBD |
| `operators.hpp` | TBD |
| `optional.hpp` | TBD |
| `pointer.hpp` | TBD |
| `select.hpp` | TBD |
| `threading.hpp` | TBD |
| `type_concepts.hpp` | TBD |
| `variant_container.hpp` | TBD |

#### Rust Design Notes
TBD

---

### `scalar/` — Scalar Types
**C++ location:** `include/quspin/scalar/`, `src/scalar/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
TBD

#### Rust Design Notes
TBD

---

### `array/` — Array Types
**C++ location:** `include/quspin/array/`, `src/array/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
TBD

#### Rust Design Notes
TBD

---

### `basis/` — Quantum Basis
**C++ location:** `include/quspin/basis/`, `src/basis/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
TBD

#### Subcomponents
| Subcomponent | C++ Location | Description |
|---|---|---|
| Group elements | `basis/grp/`, `basis/detail/symmetry/` | TBD |
| Space | `basis/detail/space.hpp` | TBD |
| Types | `basis/detail/types.hpp` | TBD |
| Basis generation | `basis/detail/generate.hpp` | TBD |
| Basis iterators | `basis/detail/iterators.hpp` | TBD |
| Basis operations | `basis/detail/basis_operations.hpp` | TBD |
| Hardcore basis | `basis/hardcore.hpp` | TBD |

#### Rust Design Notes
TBD

---

### `qmatrix/` — Quantum Sparse Matrix
**C++ location:** `include/quspin/qmatrix/`, `src/qmatrix/`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
TBD

#### Rust Design Notes
TBD

---

### `operator/` — Operators
**C++ location:** `include/quspin/operator.hpp`
**Target crate:** `quspin-core`
**Status:** ⬜

#### Description
TBD

#### Rust Design Notes
TBD

---

### Dispatch Layer + PyO3
**C++ location:** `src/`
**Target crate:** `quspin-py`
**Status:** ⬜

#### Description
TBD

#### Rust Design Notes
- `macro_rules!` for dtype dispatch over a fixed set of numeric types
- Upgrade to proc macros if `macro_rules!` proves insufficient

---

## Open Questions

- (none)
