/// Benes permutation network.
///
/// Port of `quspin/basis/detail/bitbasis/benes.hpp` (Jasper L. Neumann's algorithm,
/// adapted 2012-08-31 for "don't care" entries).
///
/// A [`BenesNetwork<B>`] is constructed by [`gen_benes`] from a target permutation
/// and applied forward with [`benes_fwd`].
use crate::bitbasis::BitInt;
use smallvec::SmallVec;

const NO_INDEX: i32 = -1;

// ---------------------------------------------------------------------------
// Butterfly network
// ---------------------------------------------------------------------------

/// A butterfly network stage configuration.
///
/// `cfg[stage]` is the swap mask for stage `stage`.  Length = `LD_BITS` (= log₂(BITS)).
#[derive(Clone)]
struct Butterfly<B: BitInt> {
    cfg: SmallVec<[B; 10]>,
}

// ---------------------------------------------------------------------------
// BenesNetwork
// ---------------------------------------------------------------------------

/// A Benes permutation network for `B`-wide integers.
///
/// Constructed via [`gen_benes`] from a bit-level target permutation array.
/// Applied via [`benes_fwd`].
#[derive(Clone)]
pub struct BenesNetwork<B: BitInt> {
    b1: Butterfly<B>,
    b2: Butterfly<B>,
}

// ---------------------------------------------------------------------------
// Core primitives
// ---------------------------------------------------------------------------

/// Single butterfly step: conditionally swap bit `i` and bit `i + shift` for
/// all pairs controlled by `m`.
#[inline]
fn bit_permute_step<B: BitInt>(x: B, m: B, shift: usize) -> B {
    let t = ((x >> shift) ^ x) & m;
    let x = x ^ t;
    let t = t << shift;
    x ^ t
}

/// Apply butterfly network `bf` (stages processed from `ld_bits-1` down to 0).
fn bfly<B: BitInt>(bf: &Butterfly<B>, mut x: B) -> B {
    for stage in (0..bf.cfg.len()).rev() {
        x = bit_permute_step(x, bf.cfg[stage], 1 << stage);
    }
    x
}

/// Apply inverse butterfly network `bf` (stages processed from 0 to `ld_bits-1`).
fn ibfly<B: BitInt>(bf: &Butterfly<B>, mut x: B) -> B {
    for stage in 0..bf.cfg.len() {
        x = bit_permute_step(x, bf.cfg[stage], 1 << stage);
    }
    x
}

// ---------------------------------------------------------------------------
// Forward application
// ---------------------------------------------------------------------------

/// Apply the Benes permutation network to `x`.
///
/// After `gen_benes(c_tgt)`, `benes_fwd(net, x)` maps bit `i` of `x` to bit
/// `c_tgt[i]` of the result.
pub fn benes_fwd<B: BitInt>(net: &BenesNetwork<B>, x: B) -> B {
    ibfly(&net.b2, bfly(&net.b1, x))
}

// ---------------------------------------------------------------------------
// Network construction
// ---------------------------------------------------------------------------

/// Invert a partial permutation: `inv[src[i]] = i` for all `i` where `src[i] != NO_INDEX`.
fn invert_perm(src: &[i32], tgt: &mut [i32]) {
    for t in tgt.iter_mut() {
        *t = NO_INDEX;
    }
    for (i, &s) in src.iter().enumerate() {
        if s != NO_INDEX {
            tgt[s as usize] = i as i32;
        }
    }
}

/// Generate a Benes network configuration from a bit-level target permutation
/// with a specified stage order.
///
/// This is the core algorithm (gen_benes_ex from the C++ reference).
///
/// `c_tgt[i]` = destination of bit `i` (`NO_INDEX` = don't care / identity).
/// `a_stage` = stage visit order (length = `LD_BITS`).
fn gen_benes_ex<B: BitInt>(c_tgt: &[i32], a_stage: &[usize]) -> BenesNetwork<B> {
    let bits = B::BITS as usize;
    let ld_bits = B::LD_BITS as usize;

    // Initialise src and tgt arrays.
    let mut src = vec![NO_INDEX; bits];
    let mut tgt = vec![NO_INDEX; bits];

    for s in 0..bits {
        if c_tgt[s] != NO_INDEX {
            tgt[s] = s as i32;
            src[c_tgt[s] as usize] = s as i32;
        }
    }

    // Compute inverse permutations.
    let mut inv_src = vec![NO_INDEX; bits];
    let mut inv_tgt = vec![NO_INDEX; bits];
    invert_perm(&src, &mut inv_src);
    invert_perm(&tgt, &mut inv_tgt);

    // Build butterfly stage configs, starting with zero.
    let mut cfg_b1: Vec<B> = vec![B::from_u64(0); ld_bits];
    let mut cfg_b2: Vec<B> = vec![B::from_u64(0); ld_bits];

    // lo_bit is 1 as the integer type B.
    let lo_bit: usize = 1;

    // Track which src indices have been handled.
    // We need a bitset large enough for B::BITS bits. For B::BITS > 64 we use a
    // Vec<u64> here to avoid generic complexity.
    let words = bits.div_ceil(64);
    let mut src_set_words = vec![0u64; words];

    let bit_set = |words: &mut Vec<u64>, idx: usize| {
        words[idx / 64] |= 1u64 << (idx % 64);
    };
    let bit_test =
        |words: &Vec<u64>, idx: usize| -> bool { (words[idx / 64] >> (idx % 64)) & 1 != 0 };

    for &stage in a_stage.iter().take(ld_bits) {
        src_set_words.iter_mut().for_each(|w| *w = 0);
        let mask = lo_bit << stage; // = 1 << stage
        let mut cfg_src = B::from_u64(0);
        let mut cfg_tgt = B::from_u64(0);

        for main_idx in 0..bits {
            if (main_idx & mask) == 0 {
                // low half of each pair only
                for aux_idx in 0..=1usize {
                    let mut src_idx = main_idx + (aux_idx << stage);
                    if !bit_test(&src_set_words, src_idx) {
                        // yet unhandled
                        if src[src_idx] != NO_INDEX {
                            // not open
                            loop {
                                bit_set(&mut src_set_words, src_idx);

                                let mut tgt_idx = inv_tgt[src[src_idx] as usize] as usize;

                                if tgt[tgt_idx] == NO_INDEX {
                                    break; // open end
                                }

                                if (src_idx ^ tgt_idx) & mask == 0 {
                                    // straight
                                    tgt_idx ^= mask;
                                } else {
                                    // cross
                                    let low_tgt = tgt_idx & !mask;
                                    cfg_tgt = cfg_tgt | (B::from_u64(1) << low_tgt);
                                    let idx2 = tgt_idx ^ mask;
                                    tgt.swap(tgt_idx, idx2);
                                    inv_tgt[tgt[idx2] as usize] = idx2 as i32;
                                    if tgt[tgt_idx] != NO_INDEX {
                                        inv_tgt[tgt[tgt_idx] as usize] = tgt_idx as i32;
                                    }
                                }

                                if tgt[tgt_idx] == NO_INDEX {
                                    break; // open end
                                }

                                src_idx = inv_src[tgt[tgt_idx] as usize] as usize;

                                if (src_idx ^ tgt_idx) & mask == 0 {
                                    // straight
                                    bit_set(&mut src_set_words, src_idx);
                                    src_idx ^= mask;
                                } else {
                                    // cross
                                    let low_src = src_idx & !mask;
                                    cfg_src = cfg_src | (B::from_u64(1) << low_src);
                                    let idx2 = src_idx ^ mask;
                                    bit_set(&mut src_set_words, idx2);
                                    src.swap(src_idx, idx2);
                                    inv_src[src[idx2] as usize] = idx2 as i32;
                                    if src[src_idx] != NO_INDEX {
                                        inv_src[src[src_idx] as usize] = src_idx as i32;
                                    }
                                }

                                if src[src_idx] == NO_INDEX {
                                    break; // open end
                                }
                                if bit_test(&src_set_words, src_idx) {
                                    break; // already handled
                                }
                            }
                        }
                    }
                }
            }
        }

        cfg_b1[stage] = cfg_src;
        cfg_b2[stage] = cfg_tgt;
    }

    BenesNetwork {
        b1: Butterfly {
            cfg: cfg_b1.into_iter().collect(),
        },
        b2: Butterfly {
            cfg: cfg_b2.into_iter().collect(),
        },
    }
}

/// Generate a Benes network configuration for the standard stage order
/// (stages processed in backward order: `LD_BITS-1, LD_BITS-2, ..., 0`).
///
/// `c_tgt[i]` = destination of bit `i` (`NO_INDEX` = don't care, treated as identity).
///
/// # Panics
///
/// Panics if `c_tgt.len() != B::BITS as usize`.
pub fn gen_benes<B: BitInt>(c_tgt: &[i32]) -> BenesNetwork<B> {
    assert_eq!(
        c_tgt.len(),
        B::BITS as usize,
        "c_tgt must have length B::BITS"
    );
    let ld_bits = B::LD_BITS as usize;
    // Standard benes order: ld_bits-1, ld_bits-2, ..., 0
    let a_stage: Vec<usize> = (0..ld_bits).rev().collect();
    gen_benes_ex::<B>(c_tgt, &a_stage)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build identity c_tgt for B::BITS bits.
    fn identity_ctgt<B: BitInt>() -> Vec<i32> {
        (0..B::BITS as usize).map(|i| i as i32).collect()
    }

    // Helper: apply a naive permutation for cross-checking.
    // c_tgt[dst] = src: output bit `dst` comes from input bit c_tgt[dst].
    fn naive_apply_perm(x: u32, c_tgt: &[i32]) -> u32 {
        let mut out = 0u32;
        for (dst, &src) in c_tgt.iter().enumerate() {
            if src != NO_INDEX && dst < 32 {
                let bit = (x >> (src as usize)) & 1;
                out |= bit << dst;
            }
        }
        out
    }

    #[test]
    fn identity_permutation() {
        let c_tgt = identity_ctgt::<u32>();
        let net = gen_benes::<u32>(&c_tgt);
        for x in [0u32, 1, 0b1010, 0xDEAD_BEEF, u32::MAX] {
            assert_eq!(benes_fwd(&net, x), x, "identity failed for x={x:#010x}");
        }
    }

    #[test]
    fn swap_bits_0_and_1() {
        // c_tgt convention: c_tgt[dst] = src (output bit dst comes from input bit src).
        // Swap: output bit 0 comes from input bit 1, output bit 1 comes from input bit 0.
        let mut c_tgt = identity_ctgt::<u32>();
        c_tgt[0] = 1; // output bit 0 ← input bit 1
        c_tgt[1] = 0; // output bit 1 ← input bit 0
        let net = gen_benes::<u32>(&c_tgt);
        // input 0b01 (bit 0 set): output bit 1 ← input bit 0 = 1, so result = 0b10
        assert_eq!(benes_fwd(&net, 0b01u32), 0b10u32);
        assert_eq!(benes_fwd(&net, 0b10u32), 0b01u32);
        assert_eq!(benes_fwd(&net, 0b11u32), 0b11u32);
        assert_eq!(benes_fwd(&net, 0b00u32), 0b00u32);
        // Cross-check with naive
        for x in 0u32..16u32 {
            assert_eq!(benes_fwd(&net, x), naive_apply_perm(x, &c_tgt));
        }
    }

    #[test]
    fn swap_bits_0_and_1_u64() {
        let mut c_tgt = identity_ctgt::<u64>();
        c_tgt[0] = 1;
        c_tgt[1] = 0;
        let net = gen_benes::<u64>(&c_tgt);
        assert_eq!(benes_fwd(&net, 0b01u64), 0b10u64);
        assert_eq!(benes_fwd(&net, 0b10u64), 0b01u64);
    }

    #[test]
    fn cyclic_shift_first_4_bits() {
        // c_tgt convention: c_tgt[dst] = src.
        // Cyclic shift where input bit i appears at output bit (i+1)%4:
        // output bit 1 ← input bit 0, output bit 2 ← input bit 1, etc.
        // So: c_tgt[0]=3, c_tgt[1]=0, c_tgt[2]=1, c_tgt[3]=2
        let mut c_tgt = identity_ctgt::<u32>();
        c_tgt[0] = 3; // output bit 0 ← input bit 3
        c_tgt[1] = 0; // output bit 1 ← input bit 0
        c_tgt[2] = 1; // output bit 2 ← input bit 1
        c_tgt[3] = 2; // output bit 3 ← input bit 2

        let net = gen_benes::<u32>(&c_tgt);
        // Check a few values against naive implementation.
        for x in 0u32..16u32 {
            let expected = naive_apply_perm(x, &c_tgt);
            let got = benes_fwd(&net, x);
            assert_eq!(
                got, expected,
                "cyclic shift failed for x={x:#06b}: got {got:#06b}, expected {expected:#06b}"
            );
        }
    }

    #[test]
    fn random_permutation_matches_naive() {
        // Use a fixed "random" permutation of all 32 bits.
        // Generated from a simple LCG shuffle.
        let mut perm: Vec<i32> = (0i32..32).collect();
        // Simple Fisher-Yates with seed 12345.
        let mut rng: u64 = 12345;
        for i in (1..32usize).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }

        let net = gen_benes::<u32>(&perm);
        let test_vals = [0u32, 1, 0b1010_1010, 0xDEAD_BEEF, u32::MAX, 0x1234_5678];
        for &x in &test_vals {
            let expected = naive_apply_perm(x, &perm);
            let got = benes_fwd(&net, x);
            assert_eq!(
                got, expected,
                "random perm failed for x={x:#010x}: got {got:#010x}, expected {expected:#010x}"
            );
        }
    }

    #[test]
    fn reverse_permutation() {
        // Reverse first 8 bits. c_tgt convention: c_tgt[dst] = src.
        // Reverse means output bit i comes from input bit 7-i for i in 0..8.
        let mut c_tgt = identity_ctgt::<u32>();
        for (i, entry) in c_tgt.iter_mut().enumerate().take(8) {
            *entry = (7 - i) as i32;
        }

        let net = gen_benes::<u32>(&c_tgt);
        for x in 0u32..256u32 {
            let expected = naive_apply_perm(x, &c_tgt);
            let got = benes_fwd(&net, x);
            assert_eq!(
                got, expected,
                "reverse perm failed for x={x:#010b}: got {got:#010b}, expected {expected:#010b}"
            );
        }
    }
}
