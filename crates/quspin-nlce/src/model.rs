//! Hamiltonian specifications evaluated on a cluster.

use crate::error::NlceError;
use crate::graph::ClusterGraph;
use num_complex::Complex;
use quspin_core::operator::spin::{SpinOp, SpinOpEntry, SpinOperator, SpinOperatorInner};

/// A spin-1/2 Hamiltonian family that can be instantiated on any cluster.
///
/// The model also declares the symmetries the solver may exploit. Declaring
/// a symmetry the Hamiltonian does not have is caught by the solver's
/// dimension check only for S^z conservation, so implementors must be
/// accurate.
pub trait Model: Sync {
    /// The Hamiltonian on cluster `g` as a spin-1/2 QuSpin operator, with
    /// every term on cindex `0` (solvers evaluate it with coefficient `1`). Site `i` of `g` is QuSpin site `i`.
    ///
    /// # Errors
    /// Model-specific (e.g. unknown bond label).
    fn operator(&self, g: &ClusterGraph) -> Result<SpinOperatorInner, NlceError>;

    /// `true` if total S^z commutes with the Hamiltonian.
    fn conserves_sz(&self) -> bool;

    /// `true` if the global spin flip (all `S^z → −S^z`) commutes with the
    /// Hamiltonian.
    fn spin_flip_symmetric(&self) -> bool;
}

/// Nearest-neighbour XXZ model in a longitudinal field,
///
/// `H = Σ_⟨ij⟩ [ Jxy (S^x_i S^x_j + S^y_i S^y_j) + Jz S^z_i S^z_j ] − hz Σ_i S^z_i`,
///
/// with spin-1/2 operators (`S = σ/2`). Bond labels are ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xxz {
    /// Transverse exchange `Jxy`.
    pub jxy: f64,
    /// Longitudinal exchange `Jz`.
    pub jz: f64,
    /// Longitudinal field `hz`.
    pub hz: f64,
}

impl Xxz {
    /// Isotropic Heisenberg model `J Σ S_i · S_j`.
    pub fn heisenberg(j: f64) -> Self {
        Self {
            jxy: j,
            jz: j,
            hz: 0.0,
        }
    }

    /// XX model `J Σ (S^x S^x + S^y S^y)`.
    pub fn xx(j: f64) -> Self {
        Self {
            jxy: j,
            jz: 0.0,
            hz: 0.0,
        }
    }

    /// Classical Ising model `J Σ S^z_i S^z_j`.
    pub fn ising(j: f64) -> Self {
        Self {
            jxy: 0.0,
            jz: j,
            hz: 0.0,
        }
    }
}

impl Model for Xxz {
    fn operator(&self, g: &ClusterGraph) -> Result<SpinOperatorInner, NlceError> {
        let c = |x: f64| Complex::new(x, 0.0);
        let mut terms = Vec::new();
        for b in &g.bonds {
            if self.jxy != 0.0 {
                // Jxy (SxSx + SySy) = Jxy/2 (S+S- + S-S+)
                terms.push(SpinOpEntry::new(
                    0u8,
                    c(0.5 * self.jxy),
                    [(SpinOp::Plus, b.i), (SpinOp::Minus, b.j)]
                        .into_iter()
                        .collect(),
                ));
                terms.push(SpinOpEntry::new(
                    0u8,
                    c(0.5 * self.jxy),
                    [(SpinOp::Minus, b.i), (SpinOp::Plus, b.j)]
                        .into_iter()
                        .collect(),
                ));
            }
            if self.jz != 0.0 {
                terms.push(SpinOpEntry::new(
                    0u8,
                    c(self.jz),
                    [(SpinOp::Z, b.i), (SpinOp::Z, b.j)].into_iter().collect(),
                ));
            }
        }
        if self.hz != 0.0 {
            for i in 0..g.n_sites as u32 {
                terms.push(SpinOpEntry::new(
                    0u8,
                    c(-self.hz),
                    [(SpinOp::Z, i)].into_iter().collect(),
                ));
            }
        }
        Ok(SpinOperatorInner::Ham8(SpinOperator::new(terms, 2)))
    }

    fn conserves_sz(&self) -> bool {
        true
    }

    fn spin_flip_symmetric(&self) -> bool {
        self.hz == 0.0
    }
}
