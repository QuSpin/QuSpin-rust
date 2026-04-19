//! Reusable [`StateTransitions`] implementations for BFS and basis tests.
//!
//! Gated behind the `test-graphs` feature so release builds don't carry
//! this code. Downstream crates enable the feature on their
//! `[dev-dependencies] quspin-bitbasis` entry to pull the helpers into
//! their test modules.

use num_complex::Complex;

use crate::int::BitInt;
use crate::state_transitions::StateTransitions;

fn one() -> Complex<f64> {
    Complex::new(1.0, 0.0)
}

/// `X` on every site: flips each bit, emitting amplitude 1 for every
/// single-site flip from `state`.
///
/// For a system with `n_sites` sites the graph connects every state to
/// `n_sites` neighbours. Combined BFS reaches all 2^n_sites states.
pub struct XAllSites {
    pub n_sites: u32,
}

impl XAllSites {
    pub fn new(n_sites: u32) -> Self {
        Self { n_sites }
    }
}

impl StateTransitions for XAllSites {
    fn lhss(&self) -> usize {
        2
    }

    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
        for loc in 0..self.n_sites {
            let mask = B::from_u64(1u64 << loc);
            visit(one(), state ^ mask);
        }
    }
}

/// `XX + YY` nearest-neighbour hopping: emits two contributions per pair,
/// with signs chosen so they cancel exactly on `|00⟩ ↔ |11⟩`.
///
/// Particle number is preserved: BFS from a 1-particle seed reaches only
/// 1-particle states because the 0- and 2-particle images cancel.
pub struct XXYYNearestNeighbor {
    pub n_sites: u32,
}

impl XXYYNearestNeighbor {
    pub fn new(n_sites: u32) -> Self {
        Self { n_sites }
    }
}

impl StateTransitions for XXYYNearestNeighbor {
    fn lhss(&self) -> usize {
        2
    }

    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
        for i in 0..self.n_sites.saturating_sub(1) {
            let mi = B::from_u64(1u64 << i);
            let mj = B::from_u64(1u64 << (i + 1));
            let si = (state & mi) != B::from_u64(0);
            let sj = (state & mj) != B::from_u64(0);
            let ns_xx = state ^ mi ^ mj;
            // XX: always flip both bits, amplitude +1.
            visit(one(), ns_xx);
            // YY: same target, sign cancels on like-bits.
            let sign = if si == sj { -1.0 } else { 1.0 };
            visit(Complex::new(sign, 0.0), ns_xx);
        }
    }
}

/// Nearest-neighbour swap `|01⟩ ↔ |10⟩`: emits neighbours only when the
/// adjacent bits differ. Particle-number-preserving by construction.
pub struct NearestNeighborSwap {
    pub n_sites: u32,
}

impl NearestNeighborSwap {
    pub fn new(n_sites: u32) -> Self {
        Self { n_sites }
    }
}

impl StateTransitions for NearestNeighborSwap {
    fn lhss(&self) -> usize {
        2
    }

    fn neighbors<B: BitInt, F: FnMut(Complex<f64>, B)>(&self, state: B, mut visit: F) {
        let one_b = B::from_u64(1);
        for i in 0..self.n_sites.saturating_sub(1) {
            let mi = one_b << (i as usize);
            let mj = one_b << ((i + 1) as usize);
            let si = state & mi;
            let sj = state & mj;
            if si != sj {
                visit(one(), state ^ mi ^ mj);
            }
        }
    }
}
