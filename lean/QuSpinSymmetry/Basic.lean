import Mathlib

/-!
# Twisted permutation modules

The symmetry-adapted bases in `quspin-basis` are built from a finite group `G`
acting on the computational basis states `X`, where each group element carries
**two** scalars:

* a 1-dimensional character `χ : G → ℂ`, supplied by the user per element, and
* a *fermion sign* `η g x = ±1`, produced by the Jordan-Wigner string, which
  depends on the state as well as the group element.

`η` is **not** a character: it is a 1-cocycle. That distinction is the whole
reason this file exists. Every derivation about the symmetry projector has a
step that survives only because `η` is real and satisfies the cocycle identity,
and those steps are exactly where the implementation went wrong once already:
the projection conjugated a character it should not have, silently zeroing
every symmetry sector whose character was not `±1`.

This file sets up the twisted action and proves it really is a unitary
representation. `Projector.lean` builds the symmetry projector on top of it.
-/

-- Some theorems below do not use every instance in scope (e.g. a statement
-- about `G` alone while `[Fintype X]` is in the section header). Harmless.
set_option linter.unusedSectionVars false

namespace QuSpinSymmetry

open Finset

variable {G X : Type*} [Group G] [MulAction G X]

/-- A **sign cocycle** on the action of `G` on `X`.

`sign g x` is the scalar the implementation attaches when group element `g`
maps basis state `x` to `g • x` — for spins it is identically `1`, for fermions
it is the Jordan-Wigner sign `±1`.

The `cocycle` field is the compatibility condition that makes the twisted
action a genuine group action; the `unit` field records that the sign has
modulus one. -/
structure SignCocycle (G X : Type*) [Group G] [MulAction G X] where
  /-- The scalar attached to `(g, x)`. -/
  sign : G → X → ℂ
  /-- The scalar has modulus one (for fermion signs it is literally `±1`). -/
  unit : ∀ g x, ‖sign g x‖ = 1
  /-- The 1-cocycle identity.  Note the `h • x`: the sign of a product is *not*
  the product of the signs at the same point, which is what makes `sign` a
  cocycle rather than a character. -/
  cocycle : ∀ g h x, sign (g * h) x = sign g (h • x) * sign h x

namespace SignCocycle

variable (η : SignCocycle G X)

/-- A sign is never zero. -/
theorem sign_ne_zero (g : G) (x : X) : η.sign g x ≠ 0 := by
  intro h
  have := η.unit g x
  rw [h, norm_zero] at this
  exact zero_ne_one this

/-- The identity carries sign `1`.  This is forced, not assumed: setting
`g = h = 1` in the cocycle gives `s = s * s` with `s ≠ 0`. -/
@[simp]
theorem sign_one (x : X) : η.sign 1 x = 1 := by
  have h := η.cocycle 1 1 x
  rw [one_mul, one_smul] at h
  -- `h : s = s * s`; rewrite the left side as `s * 1` and cancel.
  have h2 : η.sign 1 x * 1 = η.sign 1 x * η.sign 1 x := by rw [mul_one]; exact h
  exact (mul_left_cancel₀ (η.sign_ne_zero 1 x) h2).symm

/-- The sign of an inverse, evaluated at the image point.

This is the identity that makes the projection formula work: it says the sign
picked up going `x → g • x` is undone going back.  In the Rust derivation it
appears as `η_{h⁻¹}(h · s) = η_h(s)`. -/
theorem sign_inv_smul (g : G) (x : X) : η.sign g⁻¹ (g • x) = (η.sign g x)⁻¹ := by
  have h := η.cocycle g⁻¹ g x
  rw [inv_mul_cancel, sign_one] at h
  -- `h : 1 = η.sign g⁻¹ (g • x) * η.sign g x`
  have hne := η.sign_ne_zero g x
  apply mul_right_cancel₀ hne
  rw [inv_mul_cancel₀ hne]
  exact h.symm

end SignCocycle

/-! ## The twisted permutation representation -/

/-- A vector of amplitudes over the basis states. -/
abbrev State (X : Type*) := X → ℂ

/-- The twisted permutation operator.

On basis vectors this is `Û g |x⟩ = sign g x • |g • x⟩`; written on amplitude
functions the index moves the other way, hence the `g⁻¹ • y`. -/
noncomputable def U (η : SignCocycle G X) (g : G) (v : State X) : State X :=
  fun y => η.sign g (g⁻¹ • y) * v (g⁻¹ • y)

namespace U

variable (η : SignCocycle G X)

@[simp]
theorem apply_smul (g : G) (v : State X) (x : X) :
    U η g v (g • x) = η.sign g x * v x := by
  simp [U, inv_smul_smul]

/-- The identity element acts trivially. -/
@[simp]
theorem one (v : State X) : U η 1 v = v := by
  funext y
  simp [U]

/-- **The representation property.**  This is the theorem the cocycle condition
exists to make true: `Û` is a homomorphism `G → GL(State X)` precisely because
`sign` is a 1-cocycle.  Replace the cocycle identity by the naive
"`sign` is a character" condition and this fails for fermions. -/
theorem mul (g h : G) (v : State X) : U η (g * h) v = U η g (U η h v) := by
  funext y
  show η.sign (g * h) ((g * h)⁻¹ • y) * v ((g * h)⁻¹ • y) =
    η.sign g (g⁻¹ • y) * (η.sign h (h⁻¹ • g⁻¹ • y) * v (h⁻¹ • g⁻¹ • y))
  have hy : (g * h)⁻¹ • y = h⁻¹ • g⁻¹ • y := by
    rw [mul_inv_rev, mul_smul]
  rw [hy, η.cocycle g h (h⁻¹ • g⁻¹ • y), smul_inv_smul]
  ring

/-- `Û g` is invertible, with inverse `Û g⁻¹`. -/
@[simp]
theorem inv_left (g : G) (v : State X) : U η g⁻¹ (U η g v) = v := by
  rw [← U.mul, inv_mul_cancel, U.one]

@[simp]
theorem inv_right (g : G) (v : State X) : U η g (U η g⁻¹ v) = v := by
  rw [← U.mul, mul_inv_cancel, U.one]

/-- `Û g` is linear. -/
theorem add (g : G) (v w : State X) : U η g (v + w) = U η g v + U η g w := by
  funext y; simp [U]; ring

theorem smul (g : G) (c : ℂ) (v : State X) : U η g (c • v) = c • U η g v := by
  funext y; simp [U]; ring

end U

/-! ## Inner product

Worked out explicitly rather than through `EuclideanSpace`, to keep the
algebraic content visible.  Mathlib's convention (conjugate-linear in the first
argument) is followed. -/

variable [Fintype X]

/-- `⟪v, w⟫ = Σ_x conj (v x) * w x`. -/
noncomputable def inner' (v w : State X) : ℂ := ∑ x, star (v x) * w x

@[inherit_doc] notation "⟪" v ", " w "⟫" => inner' v w

/-- **The twisted representation is unitary.**

Needs `‖sign‖ = 1`; this is where a sign of modulus other than one would break
the norm bookkeeping that the orbit normalizations rely on. -/
theorem inner_U_U (η : SignCocycle G X) (g : G) (v w : State X) :
    ⟪U η g v, U η g w⟫ = ⟪v, w⟫ := by
  unfold inner'
  rw [← Equiv.sum_comp (MulAction.toPerm g)]
  refine Finset.sum_congr rfl fun x _ => ?_
  show star (U η g v (g • x)) * U η g w (g • x) = star (v x) * w x
  rw [U.apply_smul, U.apply_smul]
  have hnorm : star (η.sign g x) * η.sign g x = 1 := by
    rw [Complex.star_def, RCLike.conj_mul, η.unit g x]
    norm_num
  calc star (η.sign g x * v x) * (η.sign g x * w x)
      = (star (η.sign g x) * η.sign g x) * (star (v x) * w x) := by
        rw [star_mul]; ring
    _ = star (v x) * w x := by rw [hnorm, one_mul]

/-! ### Sesquilinearity

Conjugate-linear in the first slot, linear in the second. -/

theorem inner_smul_left (a : ℂ) (v w : State X) : ⟪a • v, w⟫ = star a * ⟪v, w⟫ := by
  unfold inner'
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  show star (a * v x) * w x = star a * (star (v x) * w x)
  rw [star_mul]; ring

theorem inner_smul_right (a : ℂ) (v w : State X) : ⟪v, a • w⟫ = a * ⟪v, w⟫ := by
  unfold inner'
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  show star (v x) * (a * w x) = a * (star (v x) * w x)
  ring

theorem inner_sum_left {ι : Type*} (s : Finset ι) (f : ι → State X) (w : State X) :
    ⟪∑ i ∈ s, f i, w⟫ = ∑ i ∈ s, ⟪f i, w⟫ := by
  unfold inner'
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.sum_apply, star_sum, Finset.sum_mul]

theorem inner_sum_right {ι : Type*} (s : Finset ι) (v : State X) (f : ι → State X) :
    ⟪v, ∑ i ∈ s, f i⟫ = ∑ i ∈ s, ⟪v, f i⟫ := by
  unfold inner'
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.sum_apply, Finset.mul_sum]

/-- Moving `Û g` across the inner product replaces it by `Û g⁻¹`.

This is the adjoint relation `(Û g)† = Û g⁻¹`, and it is the step that decides
whether the projection formula carries `χ` or `conj χ`. -/
theorem inner_U_left (η : SignCocycle G X) (g : G) (v w : State X) :
    ⟪U η g v, w⟫ = ⟪v, U η g⁻¹ w⟫ := by
  conv_lhs => rw [← U.inv_right η g w]
  rw [inner_U_U]

end QuSpinSymmetry
