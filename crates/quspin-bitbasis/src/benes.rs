/// Benes permutation network.
///
/// Port of `quspin/basis/detail/bitbasis/benes.hpp` (Jasper L. Neumann's algorithm,
/// adapted 2012-08-31 for "don't care" entries).
///
/// A [`BenesNetwork<B>`] is constructed by [`gen_benes`] from a target permutation
/// and applied via [`BenesNetwork::apply`] (or the [`benes_fwd`] free-function shim).
use quspin_types::BitInt;
use smallvec::SmallVec;

// Sentinel: index slot is empty / "don't care".
const EMPTY: i32 = -1;

// ---------------------------------------------------------------------------
// Butterfly network
// ---------------------------------------------------------------------------

/// One half of a Benes network: a sequence of butterfly stages.
///
/// `cfg[stage]` is the swap mask for butterfly stage `stage`
/// (length = `LD_BITS` = log₂(BITS)).
#[derive(Clone)]
struct Butterfly<B: BitInt> {
    cfg: SmallVec<[B; 10]>,
}

impl<B: BitInt> Butterfly<B> {
    /// Apply the butterfly forward (stages high → low).
    #[inline]
    fn apply(&self, mut x: B) -> B {
        for (stage, &m) in self.cfg.iter().enumerate().rev() {
            x = bit_permute_step(x, m, 1 << stage);
        }
        x
    }

    /// Apply the butterfly inverse (stages low → high).
    #[inline]
    fn apply_inv(&self, mut x: B) -> B {
        for (stage, &m) in self.cfg.iter().enumerate() {
            x = bit_permute_step(x, m, 1 << stage);
        }
        x
    }
}

// ---------------------------------------------------------------------------
// BenesNetwork
// ---------------------------------------------------------------------------

/// A Benes permutation network for `B`-wide integers.
///
/// Constructed via [`gen_benes`] from a bit-level target permutation slice.
/// Applied via [`BenesNetwork::apply`].
#[derive(Clone)]
pub struct BenesNetwork<B: BitInt> {
    b1: Butterfly<B>,
    b2: Butterfly<B>,
}

impl<B: BitInt> BenesNetwork<B> {
    /// Apply the Benes permutation to `x`.
    ///
    /// Given a network built from `gen_benes(c_tgt)`, output bit `dst`
    /// receives input bit `c_tgt[dst]`.
    #[inline]
    pub fn apply(&self, x: B) -> B {
        self.b2.apply_inv(self.b1.apply(x))
    }
}

// ---------------------------------------------------------------------------
// Free-function shim (backwards compatibility)
// ---------------------------------------------------------------------------

/// Apply the Benes permutation network to `x`.  Equivalent to [`BenesNetwork::apply`].
#[inline]
pub fn benes_fwd<B: BitInt>(net: &BenesNetwork<B>, x: B) -> B {
    net.apply(x)
}

// ---------------------------------------------------------------------------
// Core butterfly primitive
// ---------------------------------------------------------------------------

/// Conditional-swap butterfly step: for each bit-pair `(i, i+shift)`, swap
/// the two bits when the corresponding bit in `m` is set.
#[inline]
fn bit_permute_step<B: BitInt>(x: B, m: B, shift: usize) -> B {
    let t = ((x >> shift) ^ x) & m;
    x ^ t ^ (t << shift)
}

// ---------------------------------------------------------------------------
// Network construction helpers
// ---------------------------------------------------------------------------

/// Compute the inverse of a partial permutation.
///
/// Sets `inv[p[i]] = i` for every slot where `p[i] != EMPTY`.
fn invert_perm(p: &[i32], inv: &mut [i32]) {
    inv.fill(EMPTY);
    for (i, &v) in p.iter().enumerate() {
        if v != EMPTY {
            inv[v as usize] = i as i32;
        }
    }
}

// ---------------------------------------------------------------------------
// Public constructor
// ---------------------------------------------------------------------------

/// Generate a Benes network from a bit-level target permutation.
///
/// `c_tgt[dst]` = the source bit that should appear at output bit `dst`.
/// Use `None` for "don't care" slots (they default to the identity mapping).
///
/// The network is sized to `c_tgt.len()`, **not** to `B::BITS`. A shorter
/// slice yields a network with `log2(c_tgt.len())` butterfly stages instead
/// of `B::LD_BITS`, which is the difference between 6 stages and 12 for a
/// 64-site permutation held in a 4096-bit integer. Bits at or above
/// `c_tgt.len()` are left untouched by the resulting network: every stage
/// mask is confined to the low block and every shift is smaller than it, so
/// no butterfly step can move a bit across the boundary.
///
/// Prefer [`gen_benes_for`] unless the caller already has a power-of-two
/// slice.
///
/// # Panics
///
/// Panics unless `c_tgt.len()` is a power of two in `2..=B::BITS`, if any
/// source index is `>= c_tgt.len()`, or if any source appears more than
/// once. `None` slots are exempt — they are don't-cares, not sources.
pub fn gen_benes<B: BitInt>(c_tgt: &[Option<usize>]) -> BenesNetwork<B> {
    let bits = c_tgt.len();

    assert!(
        bits.is_power_of_two() && (2..=B::BITS as usize).contains(&bits),
        "c_tgt length must be a power of two in 2..=B::BITS, got {bits}"
    );

    // Convert to the internal i32 representation used by the routing algorithm.
    // Convention: c_int[dst] = src  (EMPTY = don't care).
    //
    // A repeated source is not a partial permutation: `gen_benes_inner`
    // assigns `src[c_tgt[s]] = s`, so the second occurrence overwrites the
    // first and the resulting bijective network cannot satisfy the request.
    // Reject it here rather than letting routing fail or silently misroute.
    let mut seen = vec![false; bits];
    let c_int: Vec<i32> = c_tgt
        .iter()
        .map(|e| {
            e.map_or(EMPTY, |v| {
                assert!(v < bits, "source index {v} outside network width {bits}");
                assert!(
                    !seen[v],
                    "source index {v} appears more than once in c_tgt; \
                     each source may feed at most one destination"
                );
                seen[v] = true;
                v as i32
            })
        })
        .collect();

    gen_benes_inner::<B>(&c_int, bits)
}

/// Generate a Benes network sized to `n_sites` rather than to `B::BITS`.
///
/// **`perm[src] = dst`** — the site at `src` moves to `dst`. This is the
/// forward convention used throughout the crate by
/// [`BenesPermDitLocations::new`](crate::BenesPermDitLocations) and by
/// `SymElement::lattice`, so a lattice permutation can be handed straight
/// to this function. (Note it is the *opposite* of `gen_benes`'s
/// `c_tgt[dst] = src`, which is a bit-level target map, not a site
/// permutation.)
///
/// The network is routed over `n_sites.next_power_of_two()` slots — the
/// smallest block a Benes network can address that still contains every
/// permuted site — and sites outside `perm` map to identity.
///
/// This is the entry point lattice symmetries should use: the stage count
/// follows the physics instead of the storage width.
///
/// # Panics
///
/// Panics if `perm` is empty, if `n_sites.next_power_of_two() > B::BITS`, if
/// any entry is `>= perm.len()`, or if any destination appears twice.
pub fn gen_benes_for<B: BitInt>(perm: &[usize]) -> BenesNetwork<B> {
    assert!(!perm.is_empty(), "permutation must be non-empty");
    // A Benes network addresses a power-of-two block, so round `n_sites` up.
    let bits = perm.len().next_power_of_two().max(2);
    assert!(
        bits <= B::BITS as usize,
        "{} sites need a {bits}-bit network, wider than B::BITS = {}",
        perm.len(),
        B::BITS
    );

    // Seed every slot with its identity route, *not* with `None`. A `None`
    // slot is a don't-care the router may use as scratch, which would let a
    // padding bit move: for `perm = [0, 1, 2, 4, 3]` in an 8-slot network,
    // don't-care padding routed input bit 5 to output bit 6. Identity
    // padding pins those slots so bits at or above `perm.len()` are fixed.
    let mut c_tgt: Vec<Option<usize>> = (0..bits).map(Some).collect();

    // `perm` is documented as a permutation; a repeated destination would
    // silently overwrite an earlier route and yield a network that cannot
    // satisfy the request, so reject it here rather than downstream.
    let mut seen = vec![false; perm.len()];
    for (src, &dst) in perm.iter().enumerate() {
        assert!(
            dst < perm.len(),
            "permutation entry {dst} outside 0..{}",
            perm.len()
        );
        assert!(
            !seen[dst],
            "destination {dst} appears more than once; `perm` must be a permutation"
        );
        seen[dst] = true;
        // `gen_benes` wants the inverse map: output bit `dst` reads `src`.
        c_tgt[dst] = Some(src);
    }
    gen_benes::<B>(&c_tgt)
}

/// Core routing algorithm (gen_benes_ex from the C++ reference).
///
/// Routes over `bits` slots using `log2(bits)` stages, in the standard
/// order `ld_bits-1, ld_bits-2, …, 0`.
fn gen_benes_inner<B: BitInt>(c_tgt: &[i32], bits: usize) -> BenesNetwork<B> {
    let ld_bits = bits.trailing_zeros() as usize;

    // Initialise src and tgt routing arrays.
    // src[s] = d  means: in the current routing, source slot s carries element
    //               destined for d.
    // tgt[s] = s  (identity) for every defined output slot.
    let mut src = vec![EMPTY; bits];
    let mut tgt = vec![EMPTY; bits];
    for s in 0..bits {
        if c_tgt[s] != EMPTY {
            tgt[s] = s as i32;
            src[c_tgt[s] as usize] = s as i32;
        }
    }

    let mut inv_src = vec![EMPTY; bits];
    let mut inv_tgt = vec![EMPTY; bits];
    invert_perm(&src, &mut inv_src);
    invert_perm(&tgt, &mut inv_tgt);

    // Stage configs, indexed by stage (0 = shift-by-1, ld_bits-1 = shift by BITS/2).
    let mut cfg_b1: SmallVec<[B; 10]> = smallvec::smallvec![B::from_u64(0); ld_bits];
    let mut cfg_b2: SmallVec<[B; 10]> = smallvec::smallvec![B::from_u64(0); ld_bits];

    // Manual bitset: tracks which src indices have been handled this stage.
    // A Vec<u64> handles B::BITS up to 64*words without generic complexity.
    let words = bits.div_ceil(64);
    let mut src_seen = vec![0u64; words];

    // Process stages in standard Benes order: ld_bits-1 down to 0.
    for stage in (0..ld_bits).rev() {
        // Clear the visited bitset for this stage.
        src_seen.fill(0);

        let mask = 1usize << stage;
        let mut cfg_src = B::from_u64(0);
        let mut cfg_tgt = B::from_u64(0);

        for main_idx in 0..bits {
            if (main_idx & mask) != 0 {
                continue; // only process the low element of each pair
            }
            // Process both elements of the pair: low (aux=0) and high (aux=1).
            for aux in 0..=1usize {
                let mut src_idx = main_idx | (aux << stage);

                // Skip if already handled or slot is empty.
                let seen = (src_seen[src_idx / 64] >> (src_idx % 64)) & 1 != 0;
                if seen || src[src_idx] == EMPTY {
                    continue;
                }

                // Trace the alternating-path routing loop.
                loop {
                    // Mark this src slot as handled.
                    src_seen[src_idx / 64] |= 1u64 << (src_idx % 64);

                    // --- Target-side step ---
                    let mut tgt_idx = inv_tgt[src[src_idx] as usize] as usize;
                    if tgt[tgt_idx] == EMPTY {
                        break;
                    }
                    if (src_idx ^ tgt_idx) & mask == 0 {
                        // Straight: route through the partner slot.
                        tgt_idx ^= mask;
                    } else {
                        // Cross: record swap in b2 and fix up tgt / inv_tgt.
                        cfg_tgt = cfg_tgt | (B::from_u64(1) << (tgt_idx & !mask));
                        let partner = tgt_idx ^ mask;
                        tgt.swap(tgt_idx, partner);
                        inv_tgt[tgt[partner] as usize] = partner as i32;
                        if tgt[tgt_idx] != EMPTY {
                            inv_tgt[tgt[tgt_idx] as usize] = tgt_idx as i32;
                        }
                    }
                    if tgt[tgt_idx] == EMPTY {
                        break;
                    }

                    // --- Source-side step ---
                    src_idx = inv_src[tgt[tgt_idx] as usize] as usize;
                    if (src_idx ^ tgt_idx) & mask == 0 {
                        // Straight: mark current src slot and move to partner.
                        src_seen[src_idx / 64] |= 1u64 << (src_idx % 64);
                        src_idx ^= mask;
                    } else {
                        // Cross: record swap in b1 and fix up src / inv_src.
                        cfg_src = cfg_src | (B::from_u64(1) << (src_idx & !mask));
                        let partner = src_idx ^ mask;
                        src_seen[partner / 64] |= 1u64 << (partner % 64);
                        src.swap(src_idx, partner);
                        inv_src[src[partner] as usize] = partner as i32;
                        if src[src_idx] != EMPTY {
                            inv_src[src[src_idx] as usize] = src_idx as i32;
                        }
                    }

                    // Stop if we've reached an open end or a visited node.
                    if src[src_idx] == EMPTY {
                        break;
                    }
                    let seen = (src_seen[src_idx / 64] >> (src_idx % 64)) & 1 != 0;
                    if seen {
                        break;
                    }
                }
            }
        }

        cfg_b1[stage] = cfg_src;
        cfg_b2[stage] = cfg_tgt;
    }

    BenesNetwork {
        b1: Butterfly { cfg: cfg_b1 },
        b2: Butterfly { cfg: cfg_b2 },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use ruint::Uint;

    use super::*;

    fn identity_ctgt<B: BitInt>() -> Vec<Option<usize>> {
        (0..B::BITS as usize).map(Some).collect()
    }

    /// Reference: apply permutation bit-by-bit.
    /// `c_tgt[dst] = Some(src)` means output bit `dst` comes from input bit `src`.
    fn naive_apply_perm(x: u32, c_tgt: &[Option<usize>]) -> u32 {
        let mut out = 0u32;
        for (dst, &entry) in c_tgt.iter().enumerate() {
            if let Some(src) = entry {
                out |= ((x >> src) & 1) << dst;
            }
        }
        out
    }

    #[test]
    fn identity_permutation() {
        let c_tgt = identity_ctgt::<u32>();
        let net = gen_benes::<u32>(&c_tgt);
        for x in [0u32, 1, 0b1010, 0xDEAD_BEEF, u32::MAX] {
            assert_eq!(net.apply(x), x, "identity failed for x={x:#010x}");
        }
    }

    #[test]
    fn swap_bits_0_and_1() {
        let mut c_tgt = identity_ctgt::<u32>();
        c_tgt[0] = Some(1); // output bit 0 ← input bit 1
        c_tgt[1] = Some(0); // output bit 1 ← input bit 0
        let net = gen_benes::<u32>(&c_tgt);
        assert_eq!(net.apply(0b01u32), 0b10u32);
        assert_eq!(net.apply(0b10u32), 0b01u32);
        assert_eq!(net.apply(0b11u32), 0b11u32);
        assert_eq!(net.apply(0b00u32), 0b00u32);
        for x in 0u32..16u32 {
            assert_eq!(net.apply(x), naive_apply_perm(x, &c_tgt));
        }
    }

    #[test]
    fn swap_bits_0_and_1_u64() {
        let mut c_tgt = identity_ctgt::<u64>();
        c_tgt[0] = Some(1);
        c_tgt[1] = Some(0);
        let net = gen_benes::<u64>(&c_tgt);
        assert_eq!(net.apply(0b01u64), 0b10u64);
        assert_eq!(net.apply(0b10u64), 0b01u64);
    }

    #[test]
    fn cyclic_shift_first_4_bits() {
        // Cyclic shift: output bit (i+1)%4 ← input bit i, for i in 0..4.
        // c_tgt[0]=Some(3), c_tgt[1]=Some(0), c_tgt[2]=Some(1), c_tgt[3]=Some(2).
        let mut c_tgt = identity_ctgt::<u32>();
        c_tgt[0] = Some(3);
        c_tgt[1] = Some(0);
        c_tgt[2] = Some(1);
        c_tgt[3] = Some(2);
        let net = gen_benes::<u32>(&c_tgt);
        for x in 0u32..16u32 {
            let expected = naive_apply_perm(x, &c_tgt);
            let got = net.apply(x);
            assert_eq!(
                got, expected,
                "cyclic shift failed for x={x:#06b}: got {got:#06b}, expected {expected:#06b}"
            );
        }
    }

    #[test]
    fn random_permutation_matches_naive() {
        // Fixed random permutation of all 32 bits (Fisher-Yates, seed 12345).
        let mut perm: Vec<Option<usize>> = (0..32usize).map(Some).collect();
        let mut rng: u64 = 12345;
        for i in (1..32usize).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }

        let net = gen_benes::<u32>(&perm);
        for x in [0u32, 1, 0b1010_1010, 0xDEAD_BEEF, u32::MAX, 0x1234_5678] {
            let expected = naive_apply_perm(x, &perm);
            let got = net.apply(x);
            assert_eq!(
                got, expected,
                "random perm failed for x={x:#010x}: got {got:#010x}, expected {expected:#010x}"
            );
        }
    }

    #[test]
    fn fuzz_random_permutations_and_inputs() {
        // 200 random Fisher-Yates shuffles of 32 bits × 500 random inputs each.
        // Uses a deterministic LCG for reproducibility.
        let mut rng: u64 = 0xDEAD_BEEF_1234_5678;
        let mut next = || -> u64 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng
        };

        const N_PERMS: usize = 200;
        const N_INPUTS_PER_PERM: usize = 500;

        for perm_idx in 0..N_PERMS {
            let mut perm: Vec<Option<usize>> = (0..32usize).map(Some).collect();
            for i in (1..32usize).rev() {
                let j = (next() >> 33) as usize % (i + 1);
                perm.swap(i, j);
            }
            let net = gen_benes::<u32>(&perm);
            for _ in 0..N_INPUTS_PER_PERM {
                let x = next() as u32;
                let expected = naive_apply_perm(x, &perm);
                let got = net.apply(x);
                assert_eq!(
                    got, expected,
                    "perm#{perm_idx} failed for x={x:#010x}: got {got:#010x}, expected {expected:#010x}"
                );
            }
        }
    }

    #[test]
    fn reverse_permutation() {
        // Reverse first 8 bits: output bit i ← input bit 7-i for i in 0..8.
        let mut c_tgt = identity_ctgt::<u32>();
        for (i, entry) in c_tgt.iter_mut().enumerate().take(8) {
            *entry = Some(7 - i);
        }
        let net = gen_benes::<u32>(&c_tgt);
        for x in 0u32..256u32 {
            let expected = naive_apply_perm(x, &c_tgt);
            let got = net.apply(x);
            assert_eq!(
                got, expected,
                "reverse perm failed for x={x:#010b}: got {got:#010b}, expected {expected:#010b}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Networks sized to the permutation rather than to B::BITS
    // -----------------------------------------------------------------

    /// Build the same permutation two ways — over `B::BITS` slots and over
    /// `n` slots — and require identical results on the low `n` bits.
    /// This is the property the whole optimisation rests on.
    fn short_matches_full<B: BitInt + std::fmt::Debug>(perm: &[usize], probes: &[u64]) {
        let n = perm.len();
        assert!(n.is_power_of_two());

        // `perm[src] = dst` (the crate convention); `gen_benes` wants the
        // inverse bit map, so invert while filling.
        let mut full: Vec<Option<usize>> = (0..B::BITS as usize).map(Some).collect();
        for (src, &dst) in perm.iter().enumerate() {
            full[dst] = Some(src);
        }
        let net_full = gen_benes::<B>(&full);
        let net_short = gen_benes_for::<B>(perm);

        let mask = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        for &p in probes {
            let x = B::from_u64(p & mask);
            assert_eq!(
                net_short.apply(x),
                net_full.apply(x),
                "short network disagrees with full for perm {perm:?} on {p:#x}"
            );
        }
    }

    /// `gen_benes_for` must use the crate's forward site-permutation
    /// convention, `perm[src] = dst`, the same as
    /// `BenesPermDitLocations::new` and `SymElement::lattice`.
    ///
    /// It originally took `perm[dst] = src`, so handing it a lattice
    /// permutation silently applied the inverse symmetry. A cyclic shift
    /// cannot catch that — it is its own inverse up to direction — so this
    /// uses a 3-cycle, where forward and inverse differ observably.
    #[test]
    fn uses_forward_site_permutation_convention() {
        // perm[src] = dst: 0 -> 1, 1 -> 2, 2 -> 0.
        let perm = [1usize, 2, 0];
        let net = gen_benes_for::<u64>(&perm);
        assert_eq!(net.apply(0b001u64), 0b010u64, "bit 0 must move to bit 1");
        assert_eq!(net.apply(0b010u64), 0b100u64, "bit 1 must move to bit 2");
        assert_eq!(net.apply(0b100u64), 0b001u64, "bit 2 must move to bit 0");
    }

    /// Stronger form of the above: agree with `BenesPermDitLocations`, the
    /// production site-permutation type, on the same input. If the two ever
    /// disagree, one of them is applying the inverse.
    #[test]
    fn agrees_with_perm_dit_locations() {
        use crate::transform::{BenesPermDitLocations, BitStateOp};
        for perm in [
            vec![1usize, 2, 0],
            vec![2usize, 0, 1],
            vec![3usize, 0, 1, 2],
            vec![1usize, 0, 3, 2, 5, 4],
            vec![4usize, 3, 0, 1, 2],
        ] {
            let n = perm.len();
            let direct = gen_benes_for::<u64>(&perm);
            // lhss = 2 -> one bit per site, so sites and bits coincide.
            let via_api = BenesPermDitLocations::<u64>::new(2, &perm, false);
            let mask = (1u64 << n) - 1;
            for x in 0..(1u64 << n) {
                let a = direct.apply(x & mask);
                let b = BitStateOp::apply(&via_api, x & mask);
                assert_eq!(a, b, "perm {perm:?}: x={x:#b} -> {a:#b} vs {b:#b}");
            }
        }
    }

    /// Regression: a non-power-of-two site count leaves padding slots that,
    /// if routed as don't-care, the router will happily use as scratch.
    /// With `perm = [0, 1, 2, 4, 3]` in an 8-slot network this moved input
    /// bit 5 to output bit 6. Every padding bit must be a fixed point.
    #[test]
    fn padding_bits_are_fixed_points() {
        for n in [3usize, 5, 6, 7, 9, 12, 20, 33] {
            // A permutation that is not the identity on the real sites.
            let perm: Vec<usize> = (0..n).map(|d| (d + n - 1) % n).collect();
            let net = gen_benes_for::<u64>(&perm);
            let bits = n.next_power_of_two().max(2);
            for bit in n..64 {
                let x = 1u64 << bit;
                assert_eq!(
                    net.apply(x),
                    x,
                    "n={n} (network {bits} slots): bit {bit} was moved"
                );
            }
        }
    }

    /// The same property under a full-width probe: bits outside the permuted
    /// block must survive untouched alongside live bits, not just alone.
    #[test]
    fn padding_bits_survive_alongside_live_bits() {
        let n = 20usize;
        let perm: Vec<usize> = (0..n).map(|d| (d + n - 1) % n).collect();
        let net = gen_benes_for::<u64>(&perm);
        let mut rng: u64 = 0xFEED_FACE_CAFE_BEEF;
        for _ in 0..500 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let out = net.apply(rng);
            assert_eq!(
                out >> n,
                rng >> n,
                "bits at or above {n} changed for {rng:#x}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "appears more than once")]
    fn rejects_duplicate_destination() {
        let _ = gen_benes_for::<u64>(&[0usize, 1, 1, 3]);
    }

    /// `gen_benes` itself must also reject a repeated source; `None` slots
    /// stay exempt because they are don't-cares rather than sources.
    #[test]
    #[should_panic(expected = "appears more than once")]
    fn gen_benes_rejects_duplicate_source() {
        let c_tgt = vec![Some(0usize), Some(1), Some(1), Some(3)];
        let _ = gen_benes::<u64>(&c_tgt);
    }

    #[test]
    fn gen_benes_allows_repeated_dont_cares() {
        let c_tgt = vec![Some(1usize), Some(0), None, None];
        let net = gen_benes::<u64>(&c_tgt);
        assert_eq!(net.apply(0b01u64), 0b10u64);
    }

    #[test]
    fn short_network_matches_full_width() {
        let probes: Vec<u64> = (0..64u64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .chain([0, 1, u64::MAX])
            .collect();

        // Cyclic shifts at several block sizes, in containers far wider.
        for &n in &[4usize, 8, 16, 32] {
            let perm: Vec<usize> = (0..n).map(|d| (d + n - 1) % n).collect();
            short_matches_full::<u64>(&perm, &probes);
            short_matches_full::<Uint<256, 4>>(&perm, &probes);
        }

        // Reversal, which routes every stage rather than just the top one.
        for &n in &[4usize, 8, 16] {
            let perm: Vec<usize> = (0..n).map(|d| n - 1 - d).collect();
            short_matches_full::<u64>(&perm, &probes);
            short_matches_full::<Uint<256, 4>>(&perm, &probes);
        }
    }

    #[test]
    fn short_network_leaves_high_bits_untouched() {
        // A 16-site permutation in a 256-bit container must not disturb any
        // bit at or above 16 — the guarantee that makes a short network
        // substitutable for a full-width one.
        let perm: Vec<usize> = (0..16usize).map(|d| (d + 15) % 16).collect();
        let net = gen_benes_for::<Uint<256, 4>>(&perm);
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        for _ in 0..200 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Set bits both inside and far outside the permuted block.
            let high: Uint<256, 4> = Uint::<256, 4>::from_u64(rng) << 64;
            let low = Uint::<256, 4>::from_u64(rng & 0xFFFF);
            let out = net.apply(low | high);
            assert_eq!(
                out >> 16,
                (low | high) >> 16,
                "bits above the permuted block were modified"
            );
        }
    }

    #[test]
    fn short_network_uses_fewer_stages() {
        // The point of the exercise: stage count follows the permutation,
        // not the container. 64 sites in a 4096-bit integer should cost 6
        // stages per butterfly, not 12.
        let perm: Vec<usize> = (0..64usize).map(|d| (d + 63) % 64).collect();
        let short = gen_benes_for::<Uint<4096, 64>>(&perm);
        assert_eq!(short.b1.cfg.len(), 6, "short network stage count");

        let full: Vec<Option<usize>> = (0..4096usize).map(Some).collect();
        let wide = gen_benes::<Uint<4096, 64>>(&full);
        assert_eq!(wide.b1.cfg.len(), 12, "full-width network stage count");
    }

    #[test]
    fn non_power_of_two_site_count_rounds_up() {
        // 20 sites route over a 32-slot network; the 12 padding slots are
        // identity and must not perturb the result.
        let perm: Vec<usize> = (0..20usize).map(|d| (d + 19) % 20).collect();
        let net = gen_benes_for::<u64>(&perm);
        assert_eq!(net.b1.cfg.len(), 5, "20 sites -> 32-slot network");

        let mut full: Vec<Option<usize>> = (0..64usize).map(Some).collect();
        for (src, &dst) in perm.iter().enumerate() {
            full[dst] = Some(src);
        }
        let reference = gen_benes::<u64>(&full);
        for i in 0..1000u64 {
            let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) & 0xF_FFFF;
            assert_eq!(
                net.apply(x),
                reference.apply(x),
                "padded network for {x:#x}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "wider than B::BITS")]
    fn rejects_permutation_wider_than_container() {
        let perm: Vec<usize> = (0..100usize).collect();
        let _ = gen_benes_for::<u64>(&perm);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two_slice() {
        let c_tgt: Vec<Option<usize>> = vec![None; 20];
        let _ = gen_benes::<u64>(&c_tgt);
    }
}
