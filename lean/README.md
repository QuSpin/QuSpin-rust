# QuSpinSymmetry — machine-checked proofs of the symmetry-sector formulas

A small Lean 4 / Mathlib development that proves the **mathematics** behind the
symmetry-adapted bases in `quspin-basis` and `quspin-matrix`.

It deliberately does **not** attempt to verify the Rust. The gap between "this
formula is correct" and "this code implements that formula" is bridged by the
test suite, as usual. What lives here is the half that the tests are bad at and
that does not rot: the derivations.

## Why

The motivating bug ([#128]): `ProjectState::project` applied a complex
conjugation to a group character that had already been conjugated. The code was
a faithful implementation of a formula that had been derived wrong. It passed
every test in the suite for two years, because every symmetry test in the repo
used a real character (`k = 0`, `k = L/2`, `η = ±1`), where the error is
invisible. For any other character the projector annihilates the entire sector
and returns exactly zero.

No amount of Rust-level verification would have caught that. The specification
was wrong. So this development targets the specification.

[#128]: https://github.com/QuSpin/QuSpin-rust/pull/128

## What is proved

All statements are fully proved — no `sorry`, and `#print axioms` reports only
Mathlib's standard three (`propext`, `Classical.choice`, `Quot.sound`).

### `Basic.lean` — the twisted permutation representation

The group elements carry two scalars: a character `χ : G → ℂ` on the unit
circle, and a Jordan–Wigner **sign** `η g x = ±1` that depends on the *state*
as well as the group element. `η` is not a character; it is a 1-cocycle.

| Theorem | Statement |
| --- | --- |
| `SignCocycle.sign_one` | `η 1 x = 1` — forced by the cocycle law, not assumed |
| `SignCocycle.sign_inv_smul` | `η g⁻¹ (g • x) = (η g x)⁻¹` — the sign undoes itself |
| `U.mul` | `Û (g·h) = Û g ∘ Û h` **iff** `η` is a 1-cocycle |
| `inner_U_U` | `⟪Û g v, Û g w⟫ = ⟪v, w⟫` — unitarity, needs `‖η‖ = 1` |
| `inner_U_left` | `⟪Û g v, w⟫ = ⟪v, Û g⁻¹ w⟫` — the adjoint relation |

`U.mul` is the theorem the cocycle condition exists to make true. Replace the
cocycle law with the naive "the sign is a character" and it fails for fermions.

### `Projector.lean` — the sector projector

`P = (1/|G|) Σ_g conj(χ g) • Û g`.

| Theorem | Statement |
| --- | --- |
| `Character.chi_inv` | `χ(g⁻¹) = conj (χ g)` — true *only* because `‖χ‖ = 1` |
| `Psum.apply_apply` | `Psum ∘ Psum = |G| • Psum` |
| `P.idem` | `P ∘ P = P` |
| `P.selfAdjoint` | `⟪P v, w⟫ = ⟪v, P w⟫` |

`P.selfAdjoint` is the one that governs #128. Its proof conjugates exactly
twice — once moving `Û g` across the inner product, once re-indexing `g ↦ g⁻¹`.
A formula that conjugates an odd number of times is not the adjoint of
anything, and the operator it defines kills every sector with a non-real
character.

### `Orbit.lean` — the stored orbit norm

`orbit.rs::check_refstate` computes `n_r = Σ_{k • r = r} χ(k) · η_k(r)` and
`collapse_norm` then asserts it is real, non-negative and integral before
rounding — but those are `debug_assert!`, so a release build checks nothing.

| Theorem | Statement |
| --- | --- |
| `coeff_mul_of_mem_stabilizer` | on the stabilizer the cocycle **collapses to a character** |
| `sum_eq_zero_or_card` | a character sum over a finite group is `0` or `|H|` |
| `orbit_norm_eq_zero_or_card` | `n_r = 0` or `n_r = |Stab(r)|` |

The middle column is the mechanism: off the stabilizer `coeff` is a genuine
cocycle and is *not* multiplicative, but for `k'` fixing `r` the `η_k(k' • r)`
in the cocycle identity becomes `η_k(r)` and the state-dependence disappears.
That is why a single integer suffices, and it is what the `debug_assert!`s are
asserting without proof.

## Not yet proved

The two formulas the Rust actually evaluates still need to be derived here.
They are the natural next step and the reason the scaffolding above exists:

1. **The projection coefficient.** For `h • s = r_j`,
   `⟪ψ_j, s⟫ = c_h(s) · √(n_j/|G|)` — un-conjugated. This is the literal
   content of `ProjectState::project`.
2. **The matrix element.** For `H` commuting with the action,
   `⟪ψ_c, H ψ_r⟩ = Σ amp · c_h(new) · √(n_c/n_r)`, which is
   `SymRows::row_into`.

Proving (1) and (2) from `P.selfAdjoint` and `orbit_norm_eq_zero_or_card` would
close the loop: the two formulas that must agree would be derived from a common
root rather than independently hand-checked.

## Building

Requires [elan](https://github.com/leanprover/elan). Mathlib's prebuilt cache
is fetched automatically on first build.

```sh
cd lean
lake exe cache get   # ~5 GB of prebuilt Mathlib oleans, once
lake build
```

The toolchain is pinned by `lean-toolchain` and Mathlib by `lakefile.toml`;
`lake-manifest.json` pins the exact revisions. `.lake/` is gitignored.
