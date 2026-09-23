//! Cluster generators: the cluster DAG an expansion sums over.
//!
//! A [`ClusterGenerator`] yields every cluster type `c` with its lattice
//! constant `L(c)` and the multiplicities `M(s, c)` of its proper
//! sub-clusters. This is the only information the [`combine`] step needs,
//! so any generator (rectangles, topological bond clusters, …) slots in
//! without changes elsewhere.
//!
//! [`combine`]: crate::combiner::combine

mod bond;
mod rectangle;

pub use bond::{BondGenerator, EmbeddingCensus};
pub use rectangle::{RectangleGenerator, RectangleOrder};

use crate::error::NlceError;
use crate::graph::{ClusterGraph, ClusterKey};

/// One cluster type of the expansion.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterType {
    /// Cache key (topology + labels).
    pub key: ClusterKey,
    /// Expansion order this cluster belongs to. Partial sums are reported
    /// per order.
    pub order: usize,
    /// A concrete open-boundary realisation handed to the solver.
    pub graph: ClusterGraph,
    /// Number of embeddings per lattice site, `L(c)`.
    pub lattice_constant: f64,
    /// Proper sub-cluster types `s` of `c` with multiplicities `M(s, c)`
    /// (the number of distinct embeddings of `s` in `c`). Only types that
    /// are themselves part of the expansion are listed.
    pub subclusters: Vec<(ClusterKey, u64)>,
}

/// Source of the cluster DAG.
pub trait ClusterGenerator {
    /// All cluster types, sorted by non-decreasing
    /// [`order`](ClusterType::order). Every key listed in a
    /// [`subclusters`](ClusterType::subclusters) entry must belong to a
    /// cluster that appears earlier in the list.
    ///
    /// # Errors
    /// Generator-specific (invalid parameters, failed realisation).
    fn clusters(&self) -> Result<Vec<ClusterType>, NlceError>;
}
