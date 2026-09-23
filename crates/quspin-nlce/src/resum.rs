//! Resummation of the bare partial sums.
//!
//! Bare NLCE partial sums often converge slowly or alternate from order to
//! order at low temperature, e.g. the even/odd swings of the bond expansion.
//! Sequence accelerators estimate the limit from the last few partial sums
//! (Tang, Khatami & Rigol, arXiv:1207.3366, Sec. 4):
//!
//! - [`Bare`]: the highest-order partial sum (no acceleration).
//! - [`Wynn`]: Wynn's ε algorithm, a nonlinear accelerator that sums
//!   geometric-like tails exactly; the workhorse for NLCE.
//! - [`Euler`]: the Euler transformation, for tails whose terms alternate in
//!   sign.
//!
//! The nonlinear methods act on each component separately
//! ([`Componentwise`]). Agreement between several methods and parameters is
//! the usual convergence criterion: resummation extrapolates, so any single
//! resummed value can be confidently wrong.

use crate::error::NlceError;
use crate::property::{Componentwise, Property};

/// Accelerates convergence of a sequence of partial sums (ascending order).
pub trait Resummation<P: Property> {
    /// Estimate of the limit of `partial_sums`.
    ///
    /// # Errors
    /// `InvalidInput` if the sequence is too short for the method.
    fn resum(&self, partial_sums: &[P]) -> Result<P, NlceError>;
}

/// No resummation: returns the highest-order partial sum.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bare;

impl<P: Property> Resummation<P> for Bare {
    fn resum(&self, partial_sums: &[P]) -> Result<P, NlceError> {
        partial_sums
            .last()
            .cloned()
            .ok_or_else(|| NlceError::InvalidInput("no partial sums to resum".into()))
    }
}

/// Wynn's ε algorithm with `cycles` cycles, applied to the last
/// `2 · cycles + 1` partial sums.
///
/// With `ε_{−1}⁽ⁿ⁾ = 0` and `ε_0⁽ⁿ⁾ = S_n`, the table
/// `ε_{k+1}⁽ⁿ⁾ = ε_{k−1}⁽ⁿ⁺¹⁾ + 1 / (ε_k⁽ⁿ⁺¹⁾ − ε_k⁽ⁿ⁾)` is built and
/// `ε_{2·cycles}` from the latest partial sums is returned; the odd columns
/// are auxiliary. `cycles = 1` is Aitken's Δ² process.
///
/// If a difference vanishes (to within a few ulps) for some component, that
/// component has converged and its most recent even-column estimate is
/// returned. Sequences with exactly repeated partial sums (e.g. the Ising
/// bond expansion, whose odd orders vanish) should be thinned to the
/// non-trivial orders first.
#[derive(Clone, Copy, Debug)]
pub struct Wynn {
    /// Number of cycles (`>= 1`).
    pub cycles: usize,
}

/// Scalar Wynn ε on `s` (length `2 · cycles + 1`).
fn wynn_scalar(s: &[f64]) -> f64 {
    let mut prev: Vec<f64> = vec![0.0; s.len() + 1];
    let mut cur: Vec<f64> = s.to_vec();
    let mut best = *s.last().unwrap();
    let mut k = 0;
    while cur.len() > 1 {
        let mut next = Vec::with_capacity(cur.len() - 1);
        for i in 0..cur.len() - 1 {
            let d = cur[i + 1] - cur[i];
            let scale = cur[i + 1].abs().max(cur[i].abs());
            if k % 2 == 0 && d.abs() <= 4.0 * f64::EPSILON * scale {
                // Even column stalled: this component has converged.
                return best;
            }
            if d == 0.0 || !d.is_finite() {
                return best;
            }
            next.push(prev[i + 1] + 1.0 / d);
        }
        prev = cur;
        cur = next;
        k += 1;
        if k % 2 == 0 {
            best = *cur.last().unwrap();
        }
    }
    best
}

impl<P: Componentwise> Resummation<P> for Wynn {
    fn resum(&self, partial_sums: &[P]) -> Result<P, NlceError> {
        let need = 2 * self.cycles + 1;
        if self.cycles == 0 || partial_sums.len() < need {
            return Err(NlceError::InvalidInput(format!(
                "Wynn with {} cycles needs at least {need} partial sums (and cycles >= 1), got {}",
                self.cycles,
                partial_sums.len()
            )));
        }
        let tail = &partial_sums[partial_sums.len() - need..];
        let comps: Vec<Vec<f64>> = tail.iter().map(Componentwise::to_components).collect();
        let n = comps[0].len();
        let out: Vec<f64> = (0..n)
            .map(|c| {
                let seq: Vec<f64> = comps.iter().map(|v| v[c]).collect();
                wynn_scalar(&seq)
            })
            .collect();
        Ok(tail[need - 1].with_components(&out))
    }
}

/// Euler transformation of the terms from position `start` on.
///
/// With terms `t_0 = S_0`, `t_n = S_n − S_{n−1}`, the partial sum
/// `S_{start−1}` is kept as is and the tail `t_{start+j} = (−1)^j u_j` is
/// replaced by `Σ_k (−1)^k Δ^k u_0 / 2^{k+1}` over all available terms
/// (`Δ` the forward difference). `start` is a position in the partial-sum
/// slice, not an order label. Only useful for alternating tails.
#[derive(Clone, Copy, Debug)]
pub struct Euler {
    /// Position of the first transformed term.
    pub start: usize,
}

/// Scalar Euler transformation of the partial sums `s` from `start`.
fn euler_scalar(s: &[f64], start: usize) -> f64 {
    let term = |n: usize| if n == 0 { s[0] } else { s[n] - s[n - 1] };
    let head = if start == 0 { 0.0 } else { s[start - 1] };
    let u: Vec<f64> = (start..s.len())
        .enumerate()
        .map(|(j, n)| if j % 2 == 0 { term(n) } else { -term(n) })
        .collect();
    // Forward differences Δ^k u_0 by repeated differencing.
    let mut diffs = u;
    let mut total = head;
    let mut k: i32 = 0;
    while !diffs.is_empty() {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        total += sign * diffs[0] / 2f64.powi(k + 1);
        diffs = diffs.windows(2).map(|w| w[1] - w[0]).collect();
        k += 1;
    }
    total
}

impl<P: Componentwise> Resummation<P> for Euler {
    fn resum(&self, partial_sums: &[P]) -> Result<P, NlceError> {
        if self.start >= partial_sums.len() {
            return Err(NlceError::InvalidInput(format!(
                "Euler start {} is beyond the {} partial sums",
                self.start,
                partial_sums.len()
            )));
        }
        let comps: Vec<Vec<f64>> = partial_sums
            .iter()
            .map(Componentwise::to_components)
            .collect();
        let n = comps[0].len();
        let out: Vec<f64> = (0..n)
            .map(|c| {
                let seq: Vec<f64> = comps.iter().map(|v| v[c]).collect();
                euler_scalar(&seq, self.start)
            })
            .collect();
        Ok(partial_sums.last().unwrap().with_components(&out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Thermo;
    use std::sync::Arc;

    /// Partial sums of `Σ_{k>=1} (−1)^{k+1} / k = ln 2`.
    fn ln2_sums(n: usize) -> Vec<Vec<f64>> {
        let mut s = 0.0;
        (1..=n)
            .map(|k| {
                s += if k % 2 == 1 { 1.0 } else { -1.0 } / k as f64;
                vec![s]
            })
            .collect()
    }

    #[test]
    fn bare_is_last() {
        let s = ln2_sums(5);
        assert_eq!(Bare.resum(&s).unwrap(), s[4]);
        assert!(Resummation::<Vec<f64>>::resum(&Bare, &[]).is_err());
    }

    #[test]
    fn wynn_sums_geometric_series_exactly() {
        // Aitken (one cycle) is exact for S_n = a + b q^n.
        let q: f64 = -0.7;
        let sums: Vec<Vec<f64>> = (1..=3)
            .map(|n| vec![(1.0 - q.powi(n)) / (1.0 - q)])
            .collect();
        let r = Wynn { cycles: 1 }.resum(&sums).unwrap();
        assert!((r[0] - 1.0 / (1.0 - q)).abs() < 1e-14);
    }

    #[test]
    fn wynn_and_euler_accelerate_ln2() {
        let ln2 = 2f64.ln();
        let s = ln2_sums(12);
        let bare = (s[11][0] - ln2).abs();
        assert!(bare > 0.03);
        let w = (Wynn { cycles: 5 }.resum(&s).unwrap()[0] - ln2).abs();
        assert!(w < 1e-8, "Wynn error {w:e}"); // ≈1e-9 from 12 terms
        let e = (Euler { start: 0 }.resum(&ln2_sums(20)).unwrap()[0] - ln2).abs();
        assert!(e < 1e-6, "Euler error {e:e}");
        // A later start keeps the head exactly and transforms the rest.
        let e3 = (Euler { start: 3 }.resum(&ln2_sums(20)).unwrap()[0] - ln2).abs();
        assert!(e3 < 1e-6, "Euler(start 3) error {e3:e}");
    }

    #[test]
    fn converged_sequences_stay_finite() {
        let s = vec![vec![0.5, 1.0]; 7];
        assert_eq!(Wynn { cycles: 3 }.resum(&s).unwrap(), vec![0.5, 1.0]);
        assert_eq!(Euler { start: 2 }.resum(&s).unwrap(), vec![0.5, 1.0]);
    }

    #[test]
    fn rejects_short_sequences() {
        let s = ln2_sums(4);
        assert!(Wynn { cycles: 2 }.resum(&s).is_err());
        assert!(Wynn { cycles: 0 }.resum(&s).is_err());
        assert!(Euler { start: 4 }.resum(&s).is_err());
    }

    #[test]
    fn thermo_is_resummed_per_component() {
        let temps: Arc<[f64]> = vec![1.0, 2.0].into();
        let s = ln2_sums(9);
        let mk = |x: f64| Thermo {
            temps: temps.clone(),
            ln_z: vec![x, 2.0 * x],
            energy: vec![-x, x],
            entropy: vec![x, x],
            specific_heat: vec![3.0 * x, x],
            magnetization: None,
            susceptibility: Some(vec![x, -x]),
        };
        let sums: Vec<Thermo> = s.iter().map(|v| mk(v[0])).collect();
        let got = Wynn { cycles: 4 }.resum(&sums).unwrap();
        let want = Wynn { cycles: 4 }.resum(&s).unwrap()[0];
        let expect = mk(want);
        assert!(got.max_abs_diff(&expect) < 1e-12);
        assert_eq!(sums[0].with_components(&sums[0].to_components()), sums[0]);
    }
}
