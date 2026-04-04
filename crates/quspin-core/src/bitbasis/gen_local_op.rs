use super::int::BitInt;
use super::transform::{BitStateOp, DynamicPermDitValues, PermDitMask, PermDitValues};

/// Unified local-op type for `GenericBasis`.
///
/// Dispatches to static compiled paths for LHSS = 2, 3, 4 and falls back
/// to a runtime path for LHSS ≥ 5.  Uses existing types from `transform`:
///
/// | Variant   | Inner type              | LHSS |
/// |-----------|-------------------------|------|
/// | `Lhss2`   | `PermDitMask<B>`        | 2    |
/// | `Lhss3`   | `PermDitValues<3>`      | 3    |
/// | `Lhss4`   | `PermDitValues<4>`      | 4    |
/// | `Dynamic` | `DynamicPermDitValues`  | ≥ 5  |
///
/// Implements `BitStateOp<B>` so that `(Complex<f64>, GenLocalOp<B>)`
/// automatically satisfies the `LocalOpItem<B>` blanket impl in `lattice.rs`.
#[derive(Clone, Debug)]
pub enum GenLocalOp<B: BitInt> {
    Lhss2(PermDitMask<B>),
    Lhss3(PermDitValues<3>),
    Lhss4(PermDitValues<4>),
    Dynamic(DynamicPermDitValues),
}

impl<B: BitInt> BitStateOp<B> for GenLocalOp<B> {
    #[inline]
    fn apply(&self, state: B) -> B {
        match self {
            GenLocalOp::Lhss2(op) => op.apply(state),
            GenLocalOp::Lhss3(op) => op.apply(state),
            GenLocalOp::Lhss4(op) => op.apply(state),
            GenLocalOp::Dynamic(op) => op.apply(state),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitbasis::manip::DynamicDitManip;

    #[test]
    fn lhss2_flip_masked_sites() {
        // Flip sites 0 and 2 (mask = bitmask with bits 0 and 2 set for lhss=2).
        // For lhss=2, bit-position of site i = i.
        let mask: u32 = 0b0101; // sites 0 and 2
        let op: GenLocalOp<u32> = GenLocalOp::Lhss2(PermDitMask::new(mask));

        // state = 0b1111 -> XOR 0b0101 = 0b1010
        assert_eq!(op.apply(0b1111u32), 0b1010u32);
        // involution
        assert_eq!(op.apply(op.apply(0b1111u32)), 0b1111u32);
    }

    #[test]
    fn lhss3_cyclic_perm() {
        // Cyclic permutation [0,1,2] -> [1,2,0] applied at sites 0 and 1.
        let op: GenLocalOp<u64> = GenLocalOp::Lhss3(PermDitValues::<3>::new([1, 2, 0], vec![0, 1]));
        let manip = DynamicDitManip::new(3);

        let mut s: u64 = 0;
        s = manip.set_dit(s, 0, 0); // site 0 = 0 -> becomes 1
        s = manip.set_dit(s, 1, 1); // site 1 = 1 -> becomes 2
        s = manip.set_dit(s, 2, 2); // site 2 = 2 -> untouched (not in locs)

        let out = op.apply(s);
        assert_eq!(manip.get_dit(out, 0), 1);
        assert_eq!(manip.get_dit(out, 1), 2);
        assert_eq!(manip.get_dit(out, 2), 2); // unchanged
    }

    #[test]
    fn lhss4_swap() {
        // Swap values 1 and 2 at site 0 only.
        let op: GenLocalOp<u64> = GenLocalOp::Lhss4(PermDitValues::<4>::new([0, 2, 1, 3], vec![0]));
        let manip = DynamicDitManip::new(4);

        let mut s: u64 = 0;
        s = manip.set_dit(s, 1, 0); // site 0 = 1 -> becomes 2
        s = manip.set_dit(s, 3, 1); // site 1 = 3 -> untouched

        let out = op.apply(s);
        assert_eq!(manip.get_dit(out, 0), 2);
        assert_eq!(manip.get_dit(out, 1), 3);
    }

    #[test]
    fn dynamic_lhss5() {
        // Identity permutation on site 0 with lhss=5.
        let op: GenLocalOp<u64> =
            GenLocalOp::Dynamic(DynamicPermDitValues::new(5, vec![0, 1, 2, 3, 4], vec![0]));
        let manip = DynamicDitManip::new(5);

        let mut s: u64 = 0;
        s = manip.set_dit(s, 3, 0);
        assert_eq!(manip.get_dit(op.apply(s), 0), 3); // identity
    }
}
