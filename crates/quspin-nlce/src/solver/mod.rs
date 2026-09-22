//! Cluster solvers: `(cluster, model) → property`.
//!
//! [`ExactDiagSolver`] is the Phase 1 implementation. Other solvers
//! (Lanczos/FTLM for larger clusters, time evolution for dynamics) implement
//! the same trait and plug into [`run_nlce`](crate::combiner::run_nlce)
//! unchanged, as long as their output implements
//! [`Property`].

mod ed;

pub use ed::{ExactDiagSolver, MAX_ED_SITES, Spectrum, SpectrumBlock};

use crate::error::NlceError;
use crate::graph::ClusterGraph;
use crate::model::Model;
use crate::property::Property;

/// Computes an extensive property of a single open-boundary cluster.
pub trait ClusterSolver<M: Model>: Sync {
    /// The property type produced (e.g. [`Thermo`](crate::property::Thermo)).
    type Output: Property;

    /// Solve `model` on cluster `g`.
    ///
    /// # Errors
    /// Solver-specific; errors from QuSpin-rust are wrapped in
    /// [`NlceError::QuSpin`].
    fn solve(&self, g: &ClusterGraph, model: &M) -> Result<Self::Output, NlceError>;
}
