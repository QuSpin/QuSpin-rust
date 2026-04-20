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
/// # Panics
///
/// Panics if `c_tgt.len() != B::BITS as usize`.
pub fn gen_benes<B: BitInt>(c_tgt: &[Option<usize>]) -> BenesNetwork<B> {
    let bits = B::BITS as usize;

    assert_eq!(c_tgt.len(), bits, "c_tgt must have length B::BITS");

    // Convert to the internal i32 representation used by the routing algorithm.
    // Convention: c_int[dst] = src  (EMPTY = don't care).
    let c_int: Vec<i32> = c_tgt
        .iter()
        .map(|e| e.map_or(EMPTY, |v| v as i32))
        .collect();

    gen_benes_inner::<B>(&c_int)
}

/// Core routing algorithm (gen_benes_ex from the C++ reference).
///
/// Uses the standard stage order: `LD_BITS-1, LD_BITS-2, …, 0`.
fn gen_benes_inner<B: BitInt>(c_tgt: &[i32]) -> BenesNetwork<B> {
    let bits = B::BITS as usize;
    let ld_bits = B::LD_BITS as usize;

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
}
