import QuSpinSymmetry.Basic

/-!
# The symmetry projector

`P = (1/|G|) Σ_g conj(χ g) • Û g` is the operator that projects onto the
symmetry sector labelled by the character `χ`.  `quspin-basis` never forms it
explicitly — it stores one representative per orbit together with a norm — but
every formula in `expand.rs`, `apply.rs` and `rowsource.rs` is a statement
about `P`, so this is where the conventions are actually pinned down.

`P.idem` and `P.selfAdjoint` establish that `P` is an orthogonal projection.
That is all they establish — in particular they do **not** pin down the
conjugation convention, because `conj ∘ χ` is itself a legal unimodular
character, so the wrongly-conjugated operator is just `P` for the conjugate
character and is idempotent and self-adjoint too.

The theorem that actually reproduces issue #128 is `Psum_comp_of_ne` (sector
orthogonality) and its corollary `P_comp_conjChar_of_not_real`, at the bottom
of this file.
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

/-! ## The character-sum dichotomy

For a multiplicative map from a finite group to `ℂ`, the sum over the group is
either `|H|` (trivial map) or `0`.  Proved directly rather than pulled from
Mathlib's `MulChar` API, since the argument is three lines and the statement
wanted here is about a bare multiplicative function.

Used twice: for the orbit norm in `Orbit.lean`, and for sector orthogonality
below. -/

section CharacterSum

variable {H : Type*} [Group H] [Fintype H] {f : H → ℂ}

/-- A nontrivial multiplicative function sums to zero over a finite group.

`f g₀ * S = S` by re-indexing, so `(f g₀ - 1) * S = 0`. -/
theorem sum_eq_zero_of_ne_one (hmul : ∀ g h, f (g * h) = f g * f h) {g₀ : H}
    (hg₀ : f g₀ ≠ 1) : ∑ g : H, f g = 0 := by
  have key : f g₀ * ∑ g : H, f g = ∑ g : H, f g := by
    rw [Finset.mul_sum]
    have : ∀ g : H, f g₀ * f g = f (g₀ * g) := fun g => (hmul g₀ g).symm
    rw [Finset.sum_congr rfl fun g _ => this g]
    exact Fintype.sum_equiv (Equiv.mulLeft g₀) _ _ fun g => rfl
  have : (f g₀ - 1) * ∑ g : H, f g = 0 := by
    rw [sub_mul, one_mul, key, sub_self]
  rcases mul_eq_zero.mp this with h | h
  · exact absurd (sub_eq_zero.mp h) hg₀
  · exact h

/-- The trivial character sums to `|H|`. -/
theorem sum_eq_card_of_forall_eq_one (h : ∀ g : H, f g = 1) :
    ∑ g : H, f g = (Fintype.card H : ℂ) := by
  rw [Finset.sum_congr rfl fun g _ => h g, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul, mul_one]

/-- **The dichotomy.** -/
theorem sum_eq_zero_or_card (hmul : ∀ g h, f (g * h) = f g * f h) :
    (∑ g : H, f g) = 0 ∨ (∑ g : H, f g) = (Fintype.card H : ℂ) := by
  by_cases h : ∀ g : H, f g = 1
  · exact Or.inr (sum_eq_card_of_forall_eq_one h)
  · push Not at h
    obtain ⟨g₀, hg₀⟩ := h
    exact Or.inl (sum_eq_zero_of_ne_one hmul hg₀)

end CharacterSum

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

/-- The unnormalized projector is additive over finite sums. -/
theorem sum_apply {ι : Type*} (s : Finset ι) (f : ι → State X) :
    Psum χ η (∑ i ∈ s, f i) = ∑ i ∈ s, Psum χ η (f i) := by
  unfold Psum
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun g _ => ?_
  rw [U_sum, Finset.smul_sum]

/-- `Psum` eats a group element, paying a factor of `χ g`.

`Psum χ ∘ Û g = χ(g) · Psum χ`, by re-indexing the sum by `h ↦ h·g`.  This is
the one-line engine behind sector orthogonality below. -/
theorem U_apply (g : G) (v : State X) :
    Psum χ η (U η g v) = χ.chi g • Psum χ η v := by
  unfold Psum
  rw [Finset.smul_sum]
  refine Fintype.sum_equiv (Equiv.mulRight g) _ _ fun h => ?_
  show star (χ.chi h) • U η h (U η g v)
      = χ.chi g • (star (χ.chi (h * g)) • U η (h * g) v)
  rw [← U.mul, χ.mul, star_mul', smul_smul]
  congr 1
  -- `χ g * conj (χ g) = 1`, so the two factors of `χ g` cancel.
  have hgg : χ.chi g * star (χ.chi g) = 1 := by
    rw [Complex.star_def, RCLike.mul_conj, χ.unit g]
    norm_num
  calc star (χ.chi h)
      = (χ.chi g * star (χ.chi g)) * star (χ.chi h) := by rw [hgg, one_mul]
    _ = χ.chi g * (star (χ.chi h) * star (χ.chi g)) := by ring

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

Together with `idem` this says `P` is an orthogonal projection.  It does *not*
by itself fix the conjugation convention: see the note on sector orthogonality
below for why, and `P_comp_conjChar_of_not_real` for the statement that does.

Two conjugations appear in the proof and have to cancel.  Neither comes from
`inner_U_left` (which has no `star` in its statement at all) — one comes from
`inner_smul_left` pulling `conj` off the scalar, producing `conj (conj (χ g))`,
and the matching one comes from `Character.chi_inv` on the re-indexed side. -/
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

/-! ## Sector orthogonality — the theorem that governs issue #128

Self-adjointness is **not** enough to pin the conjugation convention, and it is
worth being precise about why: `conj ∘ χ` is itself a perfectly legal
unimodular character, so the "wrongly conjugated" operator is simply `P` for
the conjugate character — and is therefore self-adjoint and idempotent too.
Nothing about `P` alone can tell the two apart.

What distinguishes them is that they project onto *different sectors*, and
distinct sectors are orthogonal.  In the code, `expand_ref_state_iter` builds
a vector in the `χ` sector while the erroneous `project` measured it against
the `conj χ` sector; the composite is exactly zero unless the two sectors
coincide, i.e. unless `χ` is real-valued.  That is the reported symptom
exactly: correct for `χ ≡ ±1` (momentum `0`, `L/2`, parity `±1`), identically
zero for every other character. -/

/-- The conjugate character.

Its existence is the reason self-adjointness cannot detect a stray
conjugation. -/
noncomputable def conjChar (χ : Character G) : Character G where
  chi g := star (χ.chi g)
  unit g := by simpa using χ.unit g
  mul g h := by rw [χ.mul, star_mul']

@[simp]
theorem conjChar_chi (χ : Character G) (g : G) : (conjChar χ).chi g = star (χ.chi g) := rfl

/-- **Sector orthogonality.**  Projectors for distinct characters annihilate
one another.

The proof is two lines given `Psum.U_apply`: pushing `Psum χ` through the sum
defining `Psum χ'` leaves the scalar `Σ_g conj(χ' g) · χ(g)`, a character sum
that vanishes as soon as the two characters differ anywhere. -/
theorem Psum_comp_of_ne (χ χ' : Character G) (η : SignCocycle G X) {g₀ : G}
    (hne : χ.chi g₀ ≠ χ'.chi g₀) (v : State X) :
    Psum χ η (Psum χ' η v) = 0 := by
  have expand : Psum χ η (Psum χ' η v)
      = (∑ h : G, star (χ'.chi h) * χ.chi h) • Psum χ η v := by
    -- Unfold only the *inner* projector, leaving the outer one intact so
    -- `Psum.sum_apply` can push it through.
    have inner_eq : Psum χ' η v = ∑ h : G, star (χ'.chi h) • U η h v := rfl
    rw [inner_eq, Psum.sum_apply, Finset.sum_smul]
    refine Finset.sum_congr rfl fun h _ => ?_
    rw [Psum.smul_apply, Psum.U_apply, smul_smul]
  -- The scalar is a character sum, and the character is nontrivial at `g₀`.
  have hmul : ∀ a b : G, star (χ'.chi (a * b)) * χ.chi (a * b)
      = (star (χ'.chi a) * χ.chi a) * (star (χ'.chi b) * χ.chi b) := by
    intro a b
    rw [χ.mul, χ'.mul, star_mul']
    ring
  have hg₀ : star (χ'.chi g₀) * χ.chi g₀ ≠ 1 := by
    intro hcontra
    -- `conj(χ' g₀) · χ(g₀) = 1` with `|χ'| = 1` forces `χ g₀ = χ' g₀`.
    apply hne
    have hunit : χ'.chi g₀ * star (χ'.chi g₀) = 1 := by
      rw [Complex.star_def, RCLike.mul_conj, χ'.unit g₀]
      norm_num
    calc χ.chi g₀ = (χ'.chi g₀ * star (χ'.chi g₀)) * χ.chi g₀ := by rw [hunit, one_mul]
      _ = χ'.chi g₀ * (star (χ'.chi g₀) * χ.chi g₀) := by ring
      _ = χ'.chi g₀ := by rw [hcontra, mul_one]
  rw [expand,
    sum_eq_zero_of_ne_one (f := fun h : G => star (χ'.chi h) * χ.chi h) hmul hg₀,
    zero_smul]

/-- **The `#128` statement.**  Expanding in the `χ` sector and projecting with
the conjugated character gives identically zero, unless `χ` is real at every
group element — i.e. unless `χ ≡ ±1`, which for a unimodular character is the
only way `conj χ = χ`.

This is the theorem a formalization owed the codebase: it reproduces the exact
observed behaviour (right answer at momentum `0` and `L/2`, exact zero
everywhere else) rather than merely asserting that `P` is a projection. -/
theorem P_comp_conjChar_of_not_real (χ : Character G) (η : SignCocycle G X) {g₀ : G}
    (hne : χ.chi g₀ ≠ star (χ.chi g₀)) (v : State X) :
    P χ η (P (conjChar χ) η v) = 0 := by
  unfold P
  rw [Psum.smul_apply, Psum_comp_of_ne χ (conjChar χ) η (g₀ := g₀) (by simpa using hne),
    smul_zero, smul_zero]

end QuSpinSymmetry
