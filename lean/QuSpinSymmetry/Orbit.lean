import QuSpinSymmetry.Projector

/-!
# Orbit norms

`quspin-basis` stores one representative per group orbit together with a
*norm* `n_r`, computed in `orbit.rs::check_refstate` as the character-weighted
stabilizer sum

    n_r = Σ_{k : k • r = r} χ(k) · η_k(r).

`collapse_norm` then asserts that this sum is real, non-negative and integral,
and rounds it — but those assertions are `debug_assert!`, so in a release build
nothing checks them. This file proves the fact they are asserting: the sum is
either `0` (the sector annihilates the orbit, and the representative is
dropped) or exactly the size of the stabilizer.

The mechanism is that the cocycle *collapses to a character* on the
stabilizer: for `k, k'` fixing `r` the `η_k(k' • r)` in the cocycle identity
becomes `η_k(r)`, so the state-dependence disappears. That is why `n_r` obeys
the dichotomy a character sum obeys, and it is the reason the Rust can get away
with storing a single integer.
-/

set_option linter.unusedSectionVars false

namespace QuSpinSymmetry

open Finset

/-! ## The orbit norm

The character-sum dichotomy this rests on (`sum_eq_zero_or_card`) lives in
`Projector.lean`, since sector orthogonality needs it too. -/

variable {G X : Type*} [Group G] [MulAction G X]

/-- The combined coefficient the implementation actually accumulates:
character times fermion sign. In `orbit.rs` this is what `OrbitImage::apply`
returns. -/
noncomputable def coeff (χ : Character G) (η : SignCocycle G X) (g : G) (x : X) : ℂ :=
  χ.chi g * η.sign g x

/-- **The cocycle collapses to a character on the stabilizer.**

Off the stabilizer, `coeff` is a genuine cocycle and *not* multiplicative — the
`h • x` in the cocycle identity sees a different point. On the stabilizer of
`r` that point is `r` again, the state-dependence vanishes, and what is left is
an honest 1-dimensional character of the stabilizer subgroup.

Everything in the next theorem rests on this. -/
theorem coeff_mul_of_mem_stabilizer (χ : Character G) (η : SignCocycle G X) (r : X)
    (k k' : MulAction.stabilizer G r) :
    coeff χ η (k * k' : G) r = coeff χ η (k : G) r * coeff χ η (k' : G) r := by
  unfold coeff
  have hk' : (k' : G) • r = r := k'.2
  rw [χ.mul, η.cocycle, hk']
  ring

variable [Fintype G] [DecidableEq X]

/-- **The orbit norm is either zero or the stabilizer size.**

This is the statement `collapse_norm` asserts and then rounds. In particular
the sum is real and a non-negative integer — neither of which is obvious from
the definition, since every summand is a complex number of modulus one. -/
theorem orbit_norm_eq_zero_or_card (χ : Character G) (η : SignCocycle G X) (r : X) :
    (∑ k : MulAction.stabilizer G r, coeff χ η (k : G) r) = 0 ∨
      (∑ k : MulAction.stabilizer G r, coeff χ η (k : G) r)
        = (Fintype.card (MulAction.stabilizer G r) : ℂ) :=
  sum_eq_zero_or_card (coeff_mul_of_mem_stabilizer χ η r)

end QuSpinSymmetry
