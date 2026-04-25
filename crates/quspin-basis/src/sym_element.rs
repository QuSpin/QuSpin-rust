//! [`SymElement<L>`]: explicit symmetry-group element for [`SymBasis`](super::SymBasis).
//!
//! Each user-supplied symmetry element carries at most a site permutation
//! and at most a local operator. The pair is applied atomically as one
//! group element — i.e. `g · state = local.apply(perm.apply(state))` — so
//! a user declaring `⟨P · Z⟩` (reflection composed with spin-inversion)
//! gets a group of order 2, not the cartesian product `⟨P⟩ × ⟨Z⟩` of
//! order 4.
//!
//! Elements are constructed via [`SymElement::lattice`],
//! [`SymElement::local`], or [`SymElement::composite`] — the struct's
//! fields are private so the illegal all-`None` state cannot be built.
//! The identity element is implicit in every [`SymBasis`](super::SymBasis)
//! and is never represented with a `SymElement`.

use quspin_bitbasis::Compose;

/// An explicit non-identity group element.
///
/// Type parameter `L` is the local-operator type (e.g. [`PermDitMask<B>`],
/// [`PermDitValues<N>`], [`DynamicPermDitValues`], [`SignedPermDitMask<B>`]).
/// `L` must impl [`Compose`] to enable
/// [`compose`](SymElement::compose) and the `add_cyclic` helper on
/// [`SymBasis`](super::SymBasis).
///
/// [`PermDitMask<B>`]: quspin_bitbasis::PermDitMask
/// [`PermDitValues<N>`]: quspin_bitbasis::PermDitValues
/// [`DynamicPermDitValues`]: quspin_bitbasis::DynamicPermDitValues
/// [`SignedPermDitMask<B>`]: quspin_bitbasis::SignedPermDitMask
#[derive(Clone, Debug)]
pub struct SymElement<L> {
    /// Raw site permutation: `perm[src] = dst`. `None` means identity
    /// on the lattice.
    perm: Option<Vec<usize>>,
    /// Local operator. `None` means identity on local degrees of freedom.
    local: Option<L>,
}

impl<L> SymElement<L> {
    /// Pure lattice element: site permutation only.
    ///
    /// `perm[src] = dst` moves the dit at site `src` to site `dst`.
    pub fn lattice(perm: &[usize]) -> Self {
        SymElement {
            perm: Some(perm.to_vec()),
            local: None,
        }
    }

    /// Pure local element: local operator only (no site permutation).
    pub fn local(op: L) -> Self {
        SymElement {
            perm: None,
            local: Some(op),
        }
    }

    /// Composite element: site permutation composed with a local operator.
    ///
    /// When applied, the permutation acts first and the local operator
    /// acts second: `element(state) = local.apply(perm.apply(state))`.
    /// Because lattice permutations and local operators act on
    /// orthogonal degrees of freedom, the two components commute;
    /// "perm first then local" and "local first then perm" produce the
    /// same resulting state.
    pub fn composite(perm: &[usize], op: L) -> Self {
        SymElement {
            perm: Some(perm.to_vec()),
            local: Some(op),
        }
    }

    /// Borrow the raw site permutation, if present.
    #[allow(dead_code)] // wired up in step 4 of the refactor (validation path)
    pub(crate) fn perm(&self) -> Option<&[usize]> {
        self.perm.as_deref()
    }

    /// Borrow the local operator, if present.
    #[allow(dead_code)] // wired up in step 4 of the refactor (validation path)
    pub(crate) fn local_op(&self) -> Option<&L> {
        self.local.as_ref()
    }

    /// Destructure into owned `(perm, local)` components. Used by
    /// `SymBasis::add_symmetry` to route each element into its storage
    /// vector.
    #[allow(dead_code)] // wired up in step 4 of the refactor
    pub(crate) fn into_parts(self) -> (Option<Vec<usize>>, Option<L>) {
        (self.perm, self.local)
    }
}

impl<L: Compose> SymElement<L> {
    /// Group composition: `self.compose(other)` returns the element
    /// representing `self · other` — when applied, the combined element
    /// first applies `other`, then `self`.
    ///
    /// Lattice and local components compose independently because they
    /// commute (site permutation acts on positions, local op acts on
    /// values at each site).
    ///
    /// # Precondition (panics on violation)
    ///
    /// Both operands must describe elements of the same symmetry group,
    /// so when both carry a permutation they must have identical
    /// `n_sites`. Likewise, if `L`'s [`Compose`] impl has its own
    /// preconditions (e.g. `PermDitValues`' identical-`locs` requirement)
    /// those apply here too. Composing elements from different bases is
    /// a programmer error, so we panic rather than returning a `Result`.
    pub fn compose(&self, other: &Self) -> Self {
        let perm = match (&self.perm, &other.perm) {
            (None, None) => None,
            (Some(p), None) | (None, Some(p)) => Some(p.clone()),
            (Some(a), Some(b)) => {
                // (a ∘ b)[src] = a[b[src]]  — apply b first, then a.
                let n = b.len();
                assert_eq!(
                    a.len(),
                    n,
                    "SymElement::compose: permutations have different site counts ({} vs {})",
                    a.len(),
                    n,
                );
                Some((0..n).map(|src| a[b[src]]).collect())
            }
        };
        let local = match (&self.local, &other.local) {
            (None, None) => None,
            (Some(l), None) | (None, Some(l)) => Some(l.clone()),
            (Some(a), Some(b)) => Some(a.compose(b)),
        };
        SymElement { perm, local }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use quspin_bitbasis::{BitStateOp, PermDitMask};

    type Mask = PermDitMask<u32>;

    fn mask(m: u32) -> Mask {
        PermDitMask::new(m)
    }

    #[test]
    fn constructors_populate_expected_fields() {
        let lat = SymElement::<Mask>::lattice(&[1, 0]);
        assert_eq!(lat.perm(), Some(&[1usize, 0][..]));
        assert!(lat.local_op().is_none());

        let loc = SymElement::local(mask(0b11));
        assert!(loc.perm().is_none());
        assert!(loc.local_op().is_some());

        let comp = SymElement::composite(&[1, 0], mask(0b11));
        assert_eq!(comp.perm(), Some(&[1usize, 0][..]));
        assert!(comp.local_op().is_some());
    }

    #[test]
    fn compose_pure_lattice() {
        // Swap (0 ↔ 1) twice is identity.
        let a = SymElement::<Mask>::lattice(&[1, 0]);
        let b = SymElement::<Mask>::lattice(&[1, 0]);
        let c = a.compose(&b);
        assert_eq!(c.perm(), Some(&[0usize, 1][..])); // identity
        assert!(c.local_op().is_none());
    }

    #[test]
    fn compose_pure_local() {
        // XOR masks XOR together.
        let a = SymElement::local(mask(0b1100));
        let b = SymElement::local(mask(0b1010));
        let c = a.compose(&b);
        assert!(c.perm().is_none());
        assert_eq!(c.local_op().unwrap().apply(0u32), 0b1100 ^ 0b1010);
    }

    #[test]
    fn compose_lattice_then_local_becomes_composite() {
        let lat = SymElement::<Mask>::lattice(&[1, 0]);
        let loc = SymElement::local(mask(0b11));
        let c = lat.compose(&loc);
        assert!(c.perm().is_some());
        assert!(c.local_op().is_some());
    }

    #[test]
    fn compose_composite_with_composite() {
        // (P_a, L_a) · (P_b, L_b) = (P_a · P_b, L_a · L_b)
        let a = SymElement::composite(&[2, 1, 0], mask(0b0011));
        let b = SymElement::composite(&[1, 0, 2], mask(0b1010));
        let c = a.compose(&b);
        // P_a(P_b(src)) for src=0: P_b(0)=1, P_a(1)=1. src=1: P_b(1)=0, P_a(0)=2. src=2: P_b(2)=2, P_a(2)=0.
        assert_eq!(c.perm(), Some(&[1usize, 2, 0][..]));
        assert_eq!(c.local_op().unwrap().apply(0u32), 0b0011 ^ 0b1010);
    }

    #[test]
    fn compose_is_associative() {
        let a = SymElement::composite(&[1, 2, 0], mask(0b001));
        let b = SymElement::composite(&[2, 0, 1], mask(0b010));
        let c = SymElement::composite(&[0, 2, 1], mask(0b100));

        let ab_c = a.compose(&b).compose(&c);
        let a_bc = a.compose(&b.compose(&c));

        assert_eq!(ab_c.perm(), a_bc.perm());
        assert_eq!(
            ab_c.local_op().unwrap().apply(0u32),
            a_bc.local_op().unwrap().apply(0u32)
        );
    }

    #[test]
    #[should_panic(expected = "different site counts")]
    fn compose_rejects_mismatched_perm_lengths() {
        let a = SymElement::<Mask>::lattice(&[1, 0]);
        let b = SymElement::<Mask>::lattice(&[1, 2, 0]);
        let _ = a.compose(&b);
    }
}
