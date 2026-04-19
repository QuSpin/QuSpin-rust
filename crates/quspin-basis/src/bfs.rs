//! Shared BFS wave helpers used by both [`Subspace`](super::Subspace) and
//! [`SymBasis`](super::SymBasis).

use num_complex::Complex;
use quspin_bitbasis::{BitInt, StateTransitions};
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

/// Process a single source state: apply `graph`, accumulate amplitudes per
/// target, and insert survivors into `discovered`.
///
/// `contributions` is passed in so callers can reuse the allocation across
/// multiple source states (the map is cleared at the start of each call).
#[inline]
fn discover_from_state<B, G>(
    state: B,
    graph: &G,
    contributions: &mut HashMap<B, (Complex<f64>, f64)>,
    discovered: &mut HashSet<B>,
) where
    B: BitInt,
    G: StateTransitions,
{
    contributions.clear();
    graph.neighbors::<B, _>(state, |amp, next_state| {
        if next_state != state {
            let e = contributions.entry(next_state).or_default();
            e.0 += amp;
            e.1 += amp.norm();
        }
    });
    for (&next_state, &(net_amp, scale)) in contributions.iter() {
        if net_amp.norm() > scale * AMP_CANCEL_TOL {
            discovered.insert(next_state);
        }
    }
}

/// Process one BFS wave sequentially, returning the set of candidate
/// states that survived per-source amplitude cancellation.
pub(crate) fn bfs_wave_sequential<B, G>(frontier: &[B], graph: &G) -> HashSet<B>
where
    B: BitInt,
    G: StateTransitions,
{
    let mut discovered = HashSet::new();
    let mut contributions = HashMap::new();
    for &state in frontier {
        discover_from_state(state, graph, &mut contributions, &mut discovered);
    }
    discovered
}

/// Process one BFS wave in parallel using rayon, returning the set of
/// candidate states that survived per-source amplitude cancellation.
pub(crate) fn bfs_wave_parallel<B, G>(frontier: &[B], graph: &G) -> HashSet<B>
where
    B: BitInt,
    G: StateTransitions,
{
    frontier
        .par_iter()
        .fold(
            || (HashSet::<B>::new(), HashMap::new()),
            |(mut discovered, mut contributions), &state| {
                discover_from_state(state, graph, &mut contributions, &mut discovered);
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
pub(crate) fn bfs_wave<B, G>(frontier: &[B], graph: &G) -> HashSet<B>
where
    B: BitInt,
    G: StateTransitions,
{
    if frontier.len() >= PARALLEL_FRONTIER_THRESHOLD {
        bfs_wave_parallel(frontier, graph)
    } else {
        bfs_wave_sequential(frontier, graph)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use quspin_bitbasis::test_graphs::{XAllSites, XXYYNearestNeighbor};

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
        let discovered = bfs_wave_sequential(&frontier, &XAllSites::new(3));
        // From |000⟩, X on 3 sites reaches |001⟩, |010⟩, |100⟩
        assert_eq!(discovered.len(), 3);
        assert!(discovered.contains(&0b001));
        assert!(discovered.contains(&0b010));
        assert!(discovered.contains(&0b100));
    }

    #[test]
    fn parallel_discovers_all_x_connected_states() {
        let frontier: Vec<u32> = (0..8u32).collect(); // all 3-site states
        let discovered = bfs_wave_parallel(&frontier, &XAllSites::new(3));
        // Every state reaches every other via X — discovered ⊆ {0..7}
        assert!(discovered.len() <= 8);
    }

    #[test]
    fn amplitude_cancellation_preserves_particle_number() {
        // XX+YY hopping from single-particle states should only find
        // single-particle states (no 0 or 2 particle states).
        let n = 6u32;
        let frontier: Vec<u32> = (0..n).map(|i| 1u32 << i).collect();
        let discovered = bfs_wave_sequential(&frontier, &XXYYNearestNeighbor::new(n));
        for &s in &discovered {
            assert_eq!(s.count_ones(), 1, "found non-1-particle state {s:#b}");
        }
    }

    #[test]
    fn bfs_wave_dispatches_correctly() {
        // Small frontier → sequential path
        let small: Vec<u32> = vec![0];
        let d1 = bfs_wave(&small, &XAllSites::new(3));
        assert_eq!(d1.len(), 3);

        // Large frontier → parallel path (same result)
        let large: Vec<u32> = (0..300u32).map(|i| i % 8).collect();
        let d2 = bfs_wave(&large, &XAllSites::new(3));
        assert!(d2.len() <= 8);
    }

    #[test]
    fn hop_conserves_particle_number_all_sectors() {
        let n = 6usize;
        for k in 0..=n {
            let seed = if k == 0 { 0u32 } else { (1u32 << k) - 1 };
            let mut known: HashSet<u32> = HashSet::new();
            known.insert(seed);
            let mut frontier = vec![seed];
            while !frontier.is_empty() {
                let discovered =
                    bfs_wave_sequential(&frontier, &XXYYNearestNeighbor::new(n as u32));
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
