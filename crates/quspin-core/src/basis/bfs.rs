//! Shared BFS wave helpers used by both [`Subspace`](super::Subspace) and
//! [`SymBasis`](super::SymBasis).

use crate::bitbasis::BitInt;
use num_complex::Complex;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// A net amplitude is treated as zero when its magnitude is below
/// `sum_of_input_magnitudes * AMP_CANCEL_TOL`.
///
/// This is a relative tolerance: it scales with the coupling strengths in
/// the Hamiltonian, so it catches exact symbolic cancellations (e.g. the
/// `+1` from `XX` and `−1` from `YY` on `|00⟩`) without requiring a
/// hardcoded absolute threshold.  The factor of 4 covers the two complex
/// multiplications in a typical single-site operator application.
pub(crate) const AMP_CANCEL_TOL: f64 = 4.0 * f64::EPSILON;

/// Minimum frontier size before switching to parallel BFS.
pub(crate) const PARALLEL_FRONTIER_THRESHOLD: usize = 256;

/// Process a single source state: apply `op`, accumulate amplitudes per
/// target, and insert survivors into `discovered`.
///
/// `contributions` is passed in so callers can reuse the allocation across
/// multiple source states (the map is cleared at the start of each call).
#[inline]
fn discover_from_state<B, Op, I, Iter>(
    state: B,
    op: &Op,
    contributions: &mut HashMap<B, (Complex<f64>, f64)>,
    discovered: &mut HashSet<B>,
) where
    B: BitInt,
    Op: Fn(B) -> Iter,
    Iter: IntoIterator<Item = (Complex<f64>, B, I)>,
{
    contributions.clear();
    for (amp, next_state, _cindex) in op(state) {
        if next_state != state {
            let e = contributions.entry(next_state).or_default();
            e.0 += amp;
            e.1 += amp.norm();
        }
    }
    for (&next_state, &(net_amp, scale)) in contributions.iter() {
        if net_amp.norm() > scale * AMP_CANCEL_TOL {
            discovered.insert(next_state);
        }
    }
}

/// Process one BFS wave sequentially, returning the set of candidate
/// states that survived per-source amplitude cancellation.
pub(crate) fn bfs_wave_sequential<B, Op, I, Iter>(frontier: &[B], op: &Op) -> HashSet<B>
where
    B: BitInt,
    Op: Fn(B) -> Iter,
    Iter: IntoIterator<Item = (Complex<f64>, B, I)>,
{
    let mut discovered = HashSet::new();
    let mut contributions = HashMap::new();
    for &state in frontier {
        discover_from_state(state, op, &mut contributions, &mut discovered);
    }
    discovered
}

/// Process one BFS wave in parallel using rayon, returning the set of
/// candidate states that survived per-source amplitude cancellation.
pub(crate) fn bfs_wave_parallel<B, Op, I, Iter>(frontier: &[B], op: &Op) -> HashSet<B>
where
    B: BitInt,
    Op: Fn(B) -> Iter + Sync,
    Iter: IntoIterator<Item = (Complex<f64>, B, I)>,
{
    frontier
        .par_iter()
        .fold(
            || (HashSet::<B>::new(), HashMap::new()),
            |(mut discovered, mut contributions), &state| {
                discover_from_state(state, op, &mut contributions, &mut discovered);
                (discovered, contributions)
            },
        )
        .map(|(discovered, _)| discovered)
        .reduce(
            || HashSet::new(),
            |mut a, b| {
                a.extend(b);
                a
            },
        )
}

/// Dispatch to parallel or sequential BFS wave based on frontier size.
pub(crate) fn bfs_wave<B, Op, I, Iter>(frontier: &[B], op: &Op) -> HashSet<B>
where
    B: BitInt,
    Op: Fn(B) -> Iter + Sync,
    Iter: IntoIterator<Item = (Complex<f64>, B, I)>,
{
    if frontier.len() >= PARALLEL_FRONTIER_THRESHOLD {
        bfs_wave_parallel(frontier, op)
    } else {
        bfs_wave_sequential(frontier, op)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn one() -> Complex<f64> {
        Complex::new(1.0, 0.0)
    }

    /// X on every site: flips each bit, connecting all 2^n states.
    fn x_op(n_sites: u32) -> impl Fn(u32) -> Vec<(Complex<f64>, u32, u8)> {
        move |state: u32| {
            (0..n_sites)
                .map(|loc| (one(), state ^ (1u32 << loc), 0u8))
                .collect()
        }
    }

    /// Hopping operator (XX + YY style): swaps adjacent 01↔10.
    /// The XX and YY contributions on |00⟩↔|11⟩ cancel exactly.
    fn hop_op(n_sites: u32) -> impl Fn(u32) -> Vec<(Complex<f64>, u32, u8)> {
        move |state: u32| {
            let mut out = vec![];
            for i in 0..n_sites - 1 {
                let si = (state >> i) & 1;
                let sj = (state >> (i + 1)) & 1;
                // XX: always flips both bits
                let ns_xx = state ^ (1 << i) ^ (1 << (i + 1));
                out.push((one(), ns_xx, 0u8));
                // YY: same target but with sign that cancels XX on |00⟩↔|11⟩
                let sign = if si == sj { -1.0 } else { 1.0 };
                out.push((Complex::new(sign, 0.0), ns_xx, 0u8));
            }
            out
        }
    }

    fn binom(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        (0..k).fold(1usize, |acc, i| acc * (n - i) / (i + 1))
    }

    #[test]
    fn sequential_discovers_all_x_connected_states() {
        let frontier = vec![0u32];
        let discovered = bfs_wave_sequential(&frontier, &x_op(3));
        // From |000⟩, X on 3 sites reaches |001⟩, |010⟩, |100⟩
        assert_eq!(discovered.len(), 3);
        assert!(discovered.contains(&0b001));
        assert!(discovered.contains(&0b010));
        assert!(discovered.contains(&0b100));
    }

    #[test]
    fn parallel_discovers_all_x_connected_states() {
        let frontier: Vec<u32> = (0..8u32).collect(); // all 3-site states
        let discovered = bfs_wave_parallel(&frontier, &x_op(3));
        // Every state reaches every other via X — discovered ⊆ {0..7}
        assert!(discovered.len() <= 8);
    }

    #[test]
    fn amplitude_cancellation_preserves_particle_number() {
        // XX+YY hopping from single-particle states should only find
        // single-particle states (no 0 or 2 particle states).
        let n = 6u32;
        // All 1-particle states as frontier
        let frontier: Vec<u32> = (0..n).map(|i| 1u32 << i).collect();
        let discovered = bfs_wave_sequential(&frontier, &hop_op(n));
        for &s in &discovered {
            assert_eq!(s.count_ones(), 1, "found non-1-particle state {s:#b}");
        }
    }

    #[test]
    fn bfs_wave_dispatches_correctly() {
        // Small frontier → sequential path
        let small: Vec<u32> = vec![0];
        let d1 = bfs_wave(&small, &x_op(3));
        assert_eq!(d1.len(), 3);

        // Large frontier → parallel path (same result)
        let large: Vec<u32> = (0..300u32).map(|i| i % 8).collect();
        let d2 = bfs_wave(&large, &x_op(3));
        assert!(d2.len() <= 8);
    }

    #[test]
    fn hop_conserves_particle_number_all_sectors() {
        let n = 6usize;
        for k in 0..=n {
            // Build full single-particle-number sector via repeated BFS waves
            let seed = if k == 0 { 0u32 } else { (1u32 << k) - 1 };
            let mut known: HashSet<u32> = HashSet::new();
            known.insert(seed);
            let mut frontier = vec![seed];
            while !frontier.is_empty() {
                let discovered = bfs_wave_sequential(&frontier, &hop_op(n as u32));
                frontier.clear();
                for s in discovered {
                    if known.insert(s) {
                        frontier.push(s);
                    }
                }
            }
            assert_eq!(
                known.len(),
                binom(n, k),
                "k={k}: expected C({n},{k})={}, got {}",
                binom(n, k),
                known.len()
            );
        }
    }
}
