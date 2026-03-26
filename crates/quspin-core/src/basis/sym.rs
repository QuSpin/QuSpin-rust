use super::{BasisSpace, traits::SymGrp};
use num_complex::Complex;
use std::collections::HashMap;

/// See `space::AMP_CANCEL_TOL` for the rationale.
const AMP_CANCEL_TOL: f64 = 4.0 * f64::EPSILON;

// ---------------------------------------------------------------------------
// SymmetricSubspace
// ---------------------------------------------------------------------------

/// A symmetry-reduced subspace of basis states.
///
/// Each entry stores the representative state (largest in its orbit) and its
/// orbit norm (number of distinct group images).  States are sorted in
/// ascending order by representative; a `HashMap` provides O(1) lookup.
///
/// Construction is currently sequential (matching the C++ single-thread
/// branch).  Rayon-parallel BFS is a deferred optimisation.
///
/// Mirrors `symmetric_subspace<grp_t, bitset_t, norm_t>` from `space.hpp`.
pub struct SymmetricSubspace<G: SymGrp> {
    /// `(representative_state, orbit_norm)` pairs, sorted by state ascending.
    states: Vec<(G::State, f64)>,
    /// Maps representative state → index in `states`.
    index_map: HashMap<G::State, usize>,
    grp: G,
}

impl<G: SymGrp> SymmetricSubspace<G> {
    pub fn new(grp: G) -> Self {
        SymmetricSubspace {
            states: Vec::new(),
            index_map: HashMap::new(),
            grp,
        }
    }

    /// Build the symmetric subspace reachable from `seed` under `op`, using
    /// the stored symmetry group to identify representative states.
    ///
    /// A state is included if and only if it is its own representative (i.e.
    /// no group image is larger) and its norm > 0 (non-zero orbit).
    ///
    /// Mirrors `symmetric_subspace::build` from `space.hpp` (single-thread branch).
    pub fn build<Op, I, Iter>(&mut self, seed: G::State, op: Op)
    where
        Op: Fn(G::State) -> Iter,
        Iter: IntoIterator<Item = (Complex<f64>, G::State, I)>,
    {
        // Initialise from seed.
        let (ref_seed, _coeff) = self.grp.get_refstate(seed);
        let (_ref2, norm_seed) = self.grp.check_refstate(ref_seed);

        if norm_seed > 0.0
            && let std::collections::hash_map::Entry::Vacant(e) = self.index_map.entry(ref_seed)
        {
            e.insert(self.states.len());
            self.states.push((ref_seed, norm_seed));
        }

        let mut stack: Vec<G::State> = vec![ref_seed];

        while let Some(state) = stack.pop() {
            // Accumulate (net_amp, sum_of_magnitudes) per output state.
            let mut contributions: HashMap<G::State, (Complex<f64>, f64)> = HashMap::new();
            for (amp, next_state, _cindex) in op(state) {
                if next_state != state {
                    let e = contributions.entry(next_state).or_default();
                    e.0 += amp;
                    e.1 += amp.norm();
                }
            }
            for (next_state, (net_amp, scale)) in contributions {
                if net_amp.norm() <= scale * AMP_CANCEL_TOL {
                    continue;
                }
                let (next_ref, _coeff) = self.grp.get_refstate(next_state);
                let (_ref2, next_norm) = self.grp.check_refstate(next_ref);

                if next_norm > 0.0
                    && let std::collections::hash_map::Entry::Vacant(e) =
                        self.index_map.entry(next_ref)
                {
                    e.insert(self.states.len());
                    self.states.push((next_ref, next_norm));
                    stack.push(next_ref);
                }
            }
        }

        self.sort();
    }

    fn sort(&mut self) {
        self.states.sort_unstable_by_key(|&(s, _)| s);
        self.index_map.clear();
        for (i, &(s, _)) in self.states.iter().enumerate() {
            self.index_map.insert(s, i);
        }
    }

    /// Return the representative state and orbit norm for index `i`.
    pub fn entry(&self, i: usize) -> (G::State, f64) {
        self.states[i]
    }

    /// Find the representative and accumulated coefficient for `state`.
    pub fn get_refstate(&self, state: G::State) -> (G::State, Complex<f64>) {
        self.grp.get_refstate(state)
    }

    /// Check whether `state` is a representative and return its norm.
    pub fn check_refstate(&self, state: G::State) -> (G::State, f64) {
        self.grp.check_refstate(state)
    }
}

impl<G: SymGrp> BasisSpace<G::State> for SymmetricSubspace<G> {
    #[inline]
    fn n_sites(&self) -> usize {
        self.grp.n_sites()
    }

    #[inline]
    fn size(&self) -> usize {
        self.states.len()
    }

    /// The `i`-th representative state (ascending order).
    #[inline]
    fn state_at(&self, i: usize) -> G::State {
        self.states[i].0
    }

    #[inline]
    fn index(&self, state: G::State) -> Option<usize> {
        self.index_map.get(&state).copied()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::symmetry::group::{HardcoreSymmetryGrp, LatticeElement};
    use crate::bitbasis::PermDitLocations;
    use num_complex::Complex;

    /// X operator on all sites of an N-site chain.
    fn x_op(n_sites: u32) -> impl Fn(u32) -> Vec<(Complex<f64>, u32, u8)> {
        move |state: u32| {
            (0..n_sites)
                .map(|loc| (Complex::new(1.0, 0.0), state ^ (1 << loc), 0u8))
                .collect()
        }
    }

    /// Identity lattice element for an N-site spin-1/2 chain.
    fn id_lattice(n_sites: usize) -> LatticeElement {
        let locs: Vec<usize> = (0..n_sites).collect();
        LatticeElement::new(
            Complex::new(1.0, 0.0),
            PermDitLocations::new(2, &locs),
            n_sites,
        )
    }

    /// Z₂ bit-flip group on the full N-site chain.
    fn bitflip_grp(n_sites: u32) -> HardcoreSymmetryGrp<u32> {
        let n = n_sites as usize;
        let mut grp = HardcoreSymmetryGrp::<u32>::new_empty(n);
        grp.push_lattice(id_lattice(n));
        grp.push_local_inv(Complex::new(1.0, 0.0), &(0..n).collect::<Vec<_>>());
        grp
    }

    #[test]
    fn symmetric_subspace_bitflip_2site() {
        // 2-site chain, Z₂ bitflip symmetry.
        // Orbits: {0↔3}, {1↔2}. Representatives (largest): 3, 2.
        let mut grp = HardcoreSymmetryGrp::<u32>::new_empty(2);
        grp.push_lattice(id_lattice(2));
        grp.push_local_inv(Complex::new(1.0, 0.0), &[0, 1]);
        let mut sym = SymmetricSubspace::new(grp);
        sym.build(0u32, x_op(2));

        assert_eq!(sym.size(), 2);
        assert!(sym.index(3u32).is_some());
        assert!(sym.index(2u32).is_some());
    }

    #[test]
    fn symmetric_subspace_no_symmetry_matches_subspace() {
        // With a trivial group (identity lattice only, no local ops), every
        // state is its own representative with norm = 1.
        let mut grp = HardcoreSymmetryGrp::<u32>::new_empty(3);
        grp.push_lattice(id_lattice(3));
        let mut sym = SymmetricSubspace::new(grp);
        sym.build(0u32, x_op(3));

        // All 2^3 = 8 states should be present.
        assert_eq!(sym.size(), 8);
    }

    #[test]
    fn symmetric_subspace_sorted_ascending() {
        let mut grp = HardcoreSymmetryGrp::<u32>::new_empty(3);
        grp.push_lattice(id_lattice(3));
        let mut sym = SymmetricSubspace::new(grp);
        sym.build(0u32, x_op(3));

        for i in 1..sym.size() {
            assert!(sym.state_at(i) > sym.state_at(i - 1));
        }
    }

    #[test]
    fn symmetric_subspace_bitflip_grp_helper() {
        let _ = bitflip_grp(3); // verify construction doesn't panic
    }
}
