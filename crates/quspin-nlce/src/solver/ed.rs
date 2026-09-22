//! Full exact diagonalisation, block-diagonalised with QuSpin-rust symmetry
//! sectors.

use super::ClusterSolver;
use crate::error::NlceError;
use crate::graph::{ClusterGraph, compose};
use crate::model::Model;
use crate::property::Thermo;
use num_complex::Complex;
use quspin_core::basis::SpaceKind;
use quspin_core::basis::dispatch::GenericBasis;
use quspin_core::dtype::ValueDType;
use quspin_core::{QMatrixInner, eigvalsh, with_qmatrix};
use rayon::prelude::*;
use std::sync::Arc;

type C64 = Complex<f64>;

/// Imaginary parts of a symmetry block below this (absolute) are treated as
/// round-off and the block is diagonalised as a real symmetric matrix.
const IMAG_TOL: f64 = 1e-12;

/// Largest cluster the solver accepts (the S^z seeds enumerate `2^N` states).
pub const MAX_ED_SITES: usize = 24;

/// Thermodynamics by full exact diagonalisation of every symmetry block.
///
/// Block structure, from QuSpin-rust:
/// - **S^z sectors** when the model conserves S^z. Every configuration of a
///   sector is passed as a BFS seed, so sectors are complete even when the
///   Hamiltonian does not connect them (e.g. classical Ising).
/// - **Lattice symmetry**: the largest elementary-abelian 2-subgroup of the
///   cluster's automorphisms found greedily (C2v for rectangles), with all
///   of its ±1 characters. Higher-dimensional irreps are never needed.
/// - **Spin flip**: when the model is spin-flip symmetric, sectors `±S^z`
///   are isospectral so only `S^z >= 0` is diagonalised; at `S^z = 0`
///   the flip is added to the group.
///
/// The block dimensions must add up to `2^N`; otherwise the model declared a
/// symmetry it does not have and the solver returns an error.
#[derive(Clone, Debug)]
pub struct ExactDiagSolver {
    /// Temperature grid.
    pub temps: Arc<[f64]>,
    /// Also compute magnetisation and susceptibility (needs S^z conservation).
    pub magnetization: bool,
}

impl ExactDiagSolver {
    /// Solver on the given temperature grid.
    ///
    /// # Errors
    /// `InvalidInput` if the grid is empty or contains a non-positive or
    /// non-finite temperature.
    pub fn new(temps: impl Into<Arc<[f64]>>) -> Result<Self, NlceError> {
        let temps = temps.into();
        if temps.is_empty() {
            return Err(NlceError::InvalidInput("empty temperature grid".into()));
        }
        if let Some(t) = temps.iter().find(|t| !(t.is_finite() && **t > 0.0)) {
            return Err(NlceError::InvalidInput(format!(
                "temperatures must be positive and finite, got {t}"
            )));
        }
        Ok(Self {
            temps,
            magnetization: false,
        })
    }

    /// Also compute magnetisation and susceptibility.
    pub fn with_magnetization(mut self) -> Self {
        self.magnetization = true;
        self
    }

    /// Full spectrum of `model` on `g`, organised by symmetry block.
    ///
    /// # Errors
    /// - `InvalidInput` if `g` has more than [`MAX_ED_SITES`] sites.
    /// - `InconsistentClusters` if the block dimensions do not add up to
    ///   `2^N`.
    /// - Errors from basis construction, matrix build, or `eigvalsh`.
    pub fn spectrum<M: Model>(&self, g: &ClusterGraph, model: &M) -> Result<Spectrum, NlceError> {
        let n = g.n_sites;
        if n == 0 || n > MAX_ED_SITES {
            return Err(NlceError::InvalidInput(format!(
                "exact diagonalisation supports 1..={MAX_ED_SITES} sites, got {n}"
            )));
        }
        let op = model.operator(g)?;
        let lattice = elementary_abelian_subgroup(&g.automorphisms);
        let n_lat_chars = 1usize << lattice.n_generators;

        // (n_down or None for "all states", lattice character, flip character, degeneracy)
        let mut jobs: Vec<(Option<usize>, usize, Option<f64>, f64)> = Vec::new();
        if model.conserves_sz() {
            let flip = model.spin_flip_symmetric();
            for k in 0..=n {
                let (deg, flip_chars): (f64, Vec<Option<f64>>) = if !flip {
                    (1.0, vec![None])
                } else if 2 * k > n {
                    continue;
                } else if 2 * k < n {
                    (2.0, vec![None])
                } else {
                    (1.0, vec![Some(1.0), Some(-1.0)])
                };
                for c in 0..n_lat_chars {
                    for &eta in &flip_chars {
                        jobs.push((Some(k), c, eta, deg));
                    }
                }
            }
        } else {
            for c in 0..n_lat_chars {
                jobs.push((None, c, None, 1.0));
            }
        }

        let blocks: Vec<SpectrumBlock> = jobs
            .par_iter()
            .map(|&(k, c, eta, deg)| {
                let seeds = sector_seeds(n, k);
                let mut basis = if lattice.elements.is_empty() && eta.is_none() {
                    GenericBasis::new(n, 2, SpaceKind::Sub, false)?
                } else {
                    let mut b = GenericBasis::new(n, 2, SpaceKind::Symm, false)?;
                    let all: Vec<usize> = (0..n).collect();
                    for (perm, mask) in &lattice.elements {
                        let chi = lattice_character(c, *mask);
                        b.add_lattice(C64::new(chi, 0.0), perm.clone())?;
                        if let Some(eta) = eta {
                            b.add_symmetry_raw(
                                C64::new(chi * eta, 0.0),
                                Some(perm),
                                Some(vec![1, 0]),
                                Some(all.clone()),
                            )?;
                        }
                    }
                    if let Some(eta) = eta {
                        b.add_inv(C64::new(eta, 0.0), all)?;
                    }
                    b
                };
                basis.build(&op, &seeds)?;
                let eigenvalues = diagonalize(&op, &basis)?;
                Ok(SpectrumBlock {
                    eigenvalues,
                    sz: k.map(|k| 0.5 * n as f64 - k as f64),
                    degeneracy: deg,
                })
            })
            .collect::<Result<_, NlceError>>()?;

        let total: f64 = blocks
            .iter()
            .map(|b| b.degeneracy * b.eigenvalues.len() as f64)
            .sum();
        let full = (1u64 << n) as f64;
        if total != full {
            return Err(NlceError::InconsistentClusters(format!(
                "symmetry blocks span {total} states but the {n}-site Hilbert space has {full}; \
                 the model probably declares a symmetry it does not have"
            )));
        }
        Ok(Spectrum { blocks })
    }
}

impl<M: Model> ClusterSolver<M> for ExactDiagSolver {
    type Output = Thermo;

    fn solve(&self, g: &ClusterGraph, model: &M) -> Result<Thermo, NlceError> {
        if self.magnetization && !model.conserves_sz() {
            return Err(NlceError::Unsupported(
                "magnetisation from exact diagonalisation requires an S^z-conserving model".into(),
            ));
        }
        Ok(self
            .spectrum(g, model)?
            .thermo(&self.temps, self.magnetization))
    }
}

/// Eigenvalues of one symmetry block, with its quantum numbers.
#[derive(Clone, Debug)]
pub struct SpectrumBlock {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Total S^z of the block (`None` if S^z is not conserved).
    pub sz: Option<f64>,
    /// `2.0` if the block also stands for its spin-flipped partner at
    /// `−S^z` (identical spectrum), else `1.0`.
    pub degeneracy: f64,
}

/// Complete spectrum of a cluster Hamiltonian as a list of symmetry blocks.
#[derive(Clone, Debug)]
pub struct Spectrum {
    /// The blocks; `Σ degeneracy · len = 2^N`.
    pub blocks: Vec<SpectrumBlock>,
}

impl Spectrum {
    /// Ground-state energy.
    pub fn ground_state_energy(&self) -> f64 {
        self.blocks
            .iter()
            .flat_map(|b| b.eigenvalues.first().copied())
            .fold(f64::INFINITY, f64::min)
    }

    /// All eigenvalues with multiplicity (spin-flip partners expanded),
    /// sorted ascending.
    pub fn all_eigenvalues(&self) -> Vec<f64> {
        let mut v: Vec<f64> = self
            .blocks
            .iter()
            .flat_map(|b| {
                let reps = b.degeneracy as usize;
                b.eigenvalues
                    .iter()
                    .flat_map(move |&e| std::iter::repeat_n(e, reps))
            })
            .collect();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Canonical-ensemble thermodynamics on `temps`. Boltzmann weights are
    /// taken relative to the ground-state energy so they never overflow.
    /// Magnetic quantities are only filled if `magnetization` is set and
    /// every block carries an S^z.
    pub fn thermo(&self, temps: &Arc<[f64]>, magnetization: bool) -> Thermo {
        let e0 = self.ground_state_energy();
        let magnetization = magnetization && self.blocks.iter().all(|b| b.sz.is_some());
        let nt = temps.len();
        let mut out = Thermo {
            temps: temps.clone(),
            ln_z: vec![0.0; nt],
            energy: vec![0.0; nt],
            entropy: vec![0.0; nt],
            specific_heat: vec![0.0; nt],
            magnetization: magnetization.then(|| vec![0.0; nt]),
            susceptibility: magnetization.then(|| vec![0.0; nt]),
        };
        for (it, &t) in temps.iter().enumerate() {
            let beta = 1.0 / t;
            let (mut z, mut e1, mut m1, mut m2) = (0.0, 0.0, 0.0, 0.0);
            for b in &self.blocks {
                let sz = b.sz.unwrap_or(0.0);
                for &e in &b.eigenvalues {
                    let w = b.degeneracy * (-beta * (e - e0)).exp();
                    z += w;
                    e1 += w * e;
                    // A degeneracy-2 block pairs S^z with −S^z: ⟨S^z⟩ cancels.
                    if b.degeneracy == 1.0 {
                        m1 += w * sz;
                    }
                    m2 += w * sz * sz;
                }
            }
            let mean_e = e1 / z;
            let mut var_e = 0.0;
            for b in &self.blocks {
                for &e in &b.eigenvalues {
                    let w = b.degeneracy * (-beta * (e - e0)).exp();
                    var_e += w * (e - mean_e) * (e - mean_e);
                }
            }
            var_e /= z;
            let ln_z = -beta * e0 + z.ln();
            out.ln_z[it] = ln_z;
            out.energy[it] = mean_e;
            out.entropy[it] = ln_z + beta * mean_e;
            out.specific_heat[it] = beta * beta * var_e;
            if let (Some(m), Some(chi)) = (out.magnetization.as_mut(), out.susceptibility.as_mut())
            {
                let mean_m = m1 / z;
                m[it] = mean_m;
                chi[it] = beta * (m2 / z - mean_m * mean_m);
            }
        }
        out
    }
}

/// Non-identity elements of an elementary-abelian 2-group, each with the
/// bitmask of generators whose product it is.
struct AbelianGroup {
    elements: Vec<(Vec<usize>, u32)>,
    n_generators: u32,
}

/// Greedily pick commuting involutions from `autos` and close them into an
/// elementary-abelian 2-group (all of whose irreps are ±1 characters).
fn elementary_abelian_subgroup(autos: &[Vec<usize>]) -> AbelianGroup {
    let mut gens: Vec<&Vec<usize>> = Vec::new();
    let mut elements: Vec<(Vec<usize>, u32)> = Vec::new();
    for a in autos {
        let n = a.len();
        let identity: Vec<usize> = (0..n).collect();
        if compose(a, a) != identity || elements.iter().any(|(e, _)| e == a) {
            continue;
        }
        if gens.iter().any(|g| compose(g, a) != compose(a, g)) {
            continue;
        }
        let bit = 1u32 << gens.len();
        gens.push(a);
        let mut new = vec![(a.clone(), bit)];
        for (e, mask) in &elements {
            new.push((compose(a, e), mask | bit));
        }
        elements.extend(new);
    }
    AbelianGroup {
        elements,
        n_generators: gens.len() as u32,
    }
}

/// Character `c` (a bitmask choosing the generators with χ = −1) evaluated
/// on the element with generator mask `mask`.
fn lattice_character(c: usize, mask: u32) -> f64 {
    if (c as u32 & mask).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Every configuration with `k` down spins (byte `1`), or all `2^n`
/// configurations for `k = None`.
fn sector_seeds(n: usize, k: Option<usize>) -> Vec<Vec<u8>> {
    (0u64..1 << n)
        .filter(|s| k.is_none_or(|k| s.count_ones() as usize == k))
        .map(|s| (0..n).map(|i| ((s >> i) & 1) as u8).collect())
        .collect()
}

/// Dense eigenvalues of `op` restricted to `basis`.
fn diagonalize(
    op: &quspin_core::SpinOperatorInner,
    basis: &GenericBasis,
) -> Result<Vec<f64>, NlceError> {
    let dim = basis.size();
    if dim == 0 {
        return Ok(Vec::new());
    }
    let qm = QMatrixInner::build_spin(op, basis, ValueDType::Complex128);
    let dense: Vec<C64> = with_qmatrix!(&qm, _M, _C, mat, {
        mat.to_dense::<C64>(&vec![C64::new(1.0, 0.0); mat.num_coeff()])?
    });
    let max_imag = dense.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
    let vals = if max_imag <= IMAG_TOL {
        let real: Vec<f64> = dense.iter().map(|z| z.re).collect();
        eigvalsh(&real, dim)?
    } else {
        eigvalsh(&dense, dim)?
    };
    Ok(vals)
}
