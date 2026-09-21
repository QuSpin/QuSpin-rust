import QuSpinSymmetry.Basic

/-!
# The symmetry projector

`P = (1/|G|) Σ_g conj(χ g) • Û g` is the operator that projects onto the
symmetry sector labelled by the character `χ`.  `quspin-basis` never forms it
explicitly — it stores one representative per orbit together with a norm — but
every formula in `expand.rs`, `apply.rs` and `rowsource.rs` is a statement
about `P`, so this is where the conventions are actually pinned down.

The two theorems here are the ones that make "projector" the right word:
`P ∘ P = P` and `P† = P`.  Self-adjointness is the load-bearing one — it is
what forces the *un-conjugated* character in the projection formula, and
getting that backwards is precisely the bug that shipped.
-/

-- Some theorems below do not use every instance in scope (e.g. a statement
-- about `G` alone while `[Fintype X]` is in the section header). Harmless.
set_option linter.unusedSectionVars false

namespace QuSpinSymmetry

open Finset

variable {G X : Type*} [Group G] [MulAction G X]

/-- A 1-dimensional character of `G`: a multiplicative map to the unit circle.

This is what the user supplies per group element in `add_symmetry`.  Unlike the
fermion sign it does *not* depend on the state, and unlike the sign it is a
genuine homomorphism. -/
structure Character (G : Type*) [Group G] where
  /-- The character value. -/
  chi : G → ℂ
  /-- Characters take values on the unit circle. -/
  unit : ∀ g, ‖chi g‖ = 1
  /-- Multiplicativity. -/
  mul : ∀ g h, chi (g * h) = chi g * chi h

namespace Character

variable (χ : Character G)

theorem chi_ne_zero (g : G) : χ.chi g ≠ 0 := by
  intro h
  have := χ.unit g
  rw [h, norm_zero] at this
  exact zero_ne_one this

@[simp]
theorem chi_one : χ.chi 1 = 1 := by
  have h := χ.mul 1 1
  rw [one_mul] at h
  have h2 : χ.chi 1 * 1 = χ.chi 1 * χ.chi 1 := by rw [mul_one]; exact h
  exact (mul_left_cancel₀ (χ.chi_ne_zero 1) h2).symm

/-- `χ(g⁻¹) = conj (χ g)`.

True only because `|χ| = 1`: in general a character inverse is the reciprocal,
and it coincides with the conjugate exactly on the unit circle.  This is the
step that later lets the adjoint of `P` be `P` itself. -/
theorem chi_inv (g : G) : χ.chi g⁻¹ = star (χ.chi g) := by
  have hmul : χ.chi g⁻¹ * χ.chi g = 1 := by
    rw [← χ.mul, inv_mul_cancel, χ.chi_one]
  have hstar : star (χ.chi g) * χ.chi g = 1 := by
    rw [Complex.star_def, RCLike.conj_mul, χ.unit g]
    norm_num
  exact mul_right_cancel₀ (χ.chi_ne_zero g) (hmul.trans hstar.symm)

end Character

/-! ## The projector -/

variable [Fintype G] [Fintype X]

/-- The unnormalized symmetry projector `Σ_g conj(χ g) • Û g`. -/
noncomputable def Psum (χ : Character G) (η : SignCocycle G X) (v : State X) : State X :=
  ∑ g : G, star (χ.chi g) • U η g v

/-- The symmetry projector `P = (1/|G|) Σ_g conj(χ g) • Û g`. -/
noncomputable def P (χ : Character G) (η : SignCocycle G X) (v : State X) : State X :=
  (Fintype.card G : ℂ)⁻¹ • Psum χ η v

namespace Psum

variable (χ : Character G) (η : SignCocycle G X)

/-- `Û g` commutes with finite sums (it is linear). -/
theorem U_sum {ι : Type*} (s : Finset ι) (g : G) (f : ι → State X) :
    U η g (∑ i ∈ s, f i) = ∑ i ∈ s, U η g (f i) := by
  funext y
  simp only [U, Finset.sum_apply, Finset.mul_sum]

/-- **The key algebraic step.**  Applying the unnormalized projector twice
multiplies it by `|G|`.

The proof is the standard rearrangement: `Û g ∘ Û h = Û (g*h)` and
`χ(g)χ(h) = χ(g*h)` turn the double sum into `Σ_g Σ_h F(g*h)`, and for each
fixed `g` the map `h ↦ g*h` is a bijection of `G`, so the inner sum is
independent of `g`. -/
theorem apply_apply (v : State X) :
    Psum χ η (Psum χ η v) = (Fintype.card G : ℂ) • Psum χ η v := by
  unfold Psum
  -- Push `Û g` through the inner sum and the scalars.
  have step1 : ∀ g : G,
      star (χ.chi g) • U η g (∑ h : G, star (χ.chi h) • U η h v)
        = ∑ h : G, star (χ.chi (g * h)) • U η (g * h) v := by
    intro g
    rw [U_sum]
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl fun h _ => ?_
    rw [U.smul, smul_smul, U.mul, χ.mul, star_mul']
  rw [Finset.sum_congr rfl fun g _ => step1 g]
  -- For fixed `g`, reindex the inner sum by `h ↦ g * h`.
  have step2 : ∀ g : G,
      (∑ h : G, star (χ.chi (g * h)) • U η (g * h) v)
        = ∑ k : G, star (χ.chi k) • U η k v := by
    intro g
    exact Fintype.sum_equiv (Equiv.mulLeft g) _ _ fun h => rfl
  rw [Finset.sum_congr rfl fun g _ => step2 g]
  rw [Finset.sum_const, Finset.card_univ, Nat.cast_smul_eq_nsmul]

/-- The unnormalized projector is homogeneous. -/
theorem smul_apply (a : ℂ) (v : State X) :
    Psum χ η (a • v) = a • Psum χ η v := by
  unfold Psum
  rw [Finset.smul_sum]
  refine Finset.sum_congr rfl fun g _ => ?_
  rw [U.smul, smul_smul, smul_smul, mul_comm]

end Psum

/-- `|G|` is nonzero in `ℂ`. -/
theorem card_ne_zero : (Fintype.card G : ℂ) ≠ 0 := by
  have h : 0 < Fintype.card G := Fintype.card_pos_iff.mpr ⟨1⟩
  exact_mod_cast h.ne'

namespace P

variable (χ : Character G) (η : SignCocycle G X)

/-- **`P` is idempotent.**  With `Psum ∘ Psum = |G| • Psum`, the two factors of
`1/|G|` collapse to one. -/
theorem idem (v : State X) : P χ η (P χ η v) = P χ η v := by
  unfold P
  rw [Psum.smul_apply, Psum.apply_apply, smul_smul, smul_smul, mul_assoc,
    inv_mul_cancel₀ (card_ne_zero (G := G)), mul_one]

/-- **`P` is self-adjoint.**

This is the theorem that fixes the conjugation convention, and the reason the
projection in `apply.rs` must use the group character *un-conjugated*.

Read the proof backwards to see why.  Moving `Û g` across the inner product
turns it into `Û g⁻¹`; re-indexing the sum by `g ↦ g⁻¹` then turns the
coefficient `conj (conj (χ g)) = χ g` into `χ (g⁻¹) = conj (χ g)`, which is
exactly the coefficient `P` started with.  Both steps are needed, and each one
conjugates once — so a formula that conjugates an odd number of times is not
the adjoint of anything, and the resulting operator annihilates every sector
whose character is not real. -/
theorem selfAdjoint (v w : State X) : ⟪P χ η v, w⟫ = ⟪v, P χ η w⟫ := by
  unfold P Psum
  rw [inner_smul_left, inner_smul_right, inner_sum_left, inner_sum_right,
    Finset.mul_sum, Finset.mul_sum]
  -- Re-index the left-hand sum by `g ↦ g⁻¹`.
  refine Fintype.sum_equiv (Equiv.inv G) _ _ fun x => ?_
  show star ((Fintype.card G : ℂ)⁻¹) * ⟪star (χ.chi x) • U η x v, w⟫
      = (Fintype.card G : ℂ)⁻¹ * ⟪v, star (χ.chi x⁻¹) • U η x⁻¹ w⟫
  -- Left: the two conjugations on `χ` cancel, and `Û x` crosses as `Û x⁻¹`.
  -- Right: `χ(x⁻¹) = conj (χ x)` supplies the matching conjugation.
  rw [inner_smul_left, inner_smul_right, inner_U_left, χ.chi_inv]
  simp only [star_star]
  congr 1
  simp

end P

end QuSpinSymmetry
