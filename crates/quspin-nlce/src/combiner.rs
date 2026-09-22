//! Inclusion–exclusion over the cluster DAG, and the end-to-end driver.

use crate::error::NlceError;
use crate::generator::{ClusterGenerator, ClusterType};
use crate::graph::ClusterKey;
use crate::model::Model;
use crate::property::Property;
use crate::solver::ClusterSolver;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Per-cluster properties `P(c)` and weights `W(c)`, keyed by
/// [`ClusterKey`].
///
/// A cache is only valid for one model and solver configuration; reuse it
/// across runs of the same problem (e.g. to extend to a higher order
/// without re-solving the smaller clusters).
#[derive(Clone, Debug)]
pub struct ClusterCache<P> {
    /// Cluster properties `P(c)`.
    pub properties: BTreeMap<ClusterKey, P>,
    /// Cluster weights `W(c) = P(c) − Σ_s M(s, c) W(s)`.
    pub weights: BTreeMap<ClusterKey, P>,
}

impl<P> Default for ClusterCache<P> {
    fn default() -> Self {
        Self {
            properties: BTreeMap::new(),
            weights: BTreeMap::new(),
        }
    }
}

/// Output of an expansion.
#[derive(Clone, Debug)]
pub struct NlceResult<P> {
    /// Properties and weights of every cluster.
    pub cache: ClusterCache<P>,
    /// `(n, Σ_{order(c) = n} L(c) W(c))` for each order present, ascending.
    pub order_contributions: Vec<(usize, P)>,
    /// Bare partial sums per site, `(n, Σ_{order(c) <= n} L(c) W(c))`,
    /// ascending in `n`. Input to [`Resummation`](crate::resum::Resummation).
    pub partial_sums: Vec<(usize, P)>,
}

impl<P: Property> NlceResult<P> {
    /// The highest-order bare partial sum.
    pub fn last_partial_sum(&self) -> Option<&P> {
        self.partial_sums.last().map(|(_, p)| p)
    }
}

/// Weight of one cluster from its property and the (already known) weights
/// of its sub-clusters.
fn weight<P: Property>(
    c: &ClusterType,
    p: &P,
    weights: &BTreeMap<ClusterKey, P>,
) -> Result<P, NlceError> {
    let mut w = p.clone();
    for (s, m) in &c.subclusters {
        let ws = weights.get(s).ok_or_else(|| {
            NlceError::InconsistentClusters(format!(
                "sub-cluster {s} of {} has no weight yet (clusters out of order?)",
                c.key
            ))
        })?;
        w.axpy(-(*m as f64), ws);
    }
    Ok(w)
}

/// Inclusion–exclusion: compute every weight, the per-order contributions,
/// and the bare partial sums from known cluster properties.
///
/// `clusters` must satisfy the [`ClusterGenerator::clusters`] ordering
/// contract, and `properties` must contain every cluster's key.
///
/// # Errors
/// `InconsistentClusters` if a property is missing, a sub-cluster is
/// referenced before its weight is known, or `clusters` is empty.
pub fn combine<P: Property>(
    clusters: &[ClusterType],
    properties: &BTreeMap<ClusterKey, P>,
) -> Result<NlceResult<P>, NlceError> {
    let mut weights: BTreeMap<ClusterKey, P> = BTreeMap::new();
    let mut order_contributions: Vec<(usize, P)> = Vec::new();
    for c in clusters {
        let p = properties.get(&c.key).ok_or_else(|| {
            NlceError::InconsistentClusters(format!("no property for cluster {}", c.key))
        })?;
        let w = weight(c, p, &weights)?;
        match order_contributions.last_mut() {
            Some((n, acc)) if *n == c.order => acc.axpy(c.lattice_constant, &w),
            Some((n, _)) if *n > c.order => {
                return Err(NlceError::InconsistentClusters(format!(
                    "cluster {} of order {} listed after order {n}",
                    c.key, c.order
                )));
            }
            _ => {
                let mut acc = w.zeros_like();
                acc.axpy(c.lattice_constant, &w);
                order_contributions.push((c.order, acc));
            }
        }
        weights.insert(c.key.clone(), w);
    }
    let mut partial_sums: Vec<(usize, P)> = Vec::with_capacity(order_contributions.len());
    for (n, contrib) in &order_contributions {
        let mut s = match partial_sums.last() {
            Some((_, prev)) => prev.clone(),
            None => contrib.zeros_like(),
        };
        s.axpy(1.0, contrib);
        partial_sums.push((*n, s));
    }
    if partial_sums.is_empty() {
        return Err(NlceError::InconsistentClusters("no clusters".into()));
    }
    Ok(NlceResult {
        cache: ClusterCache {
            properties: properties.clone(),
            weights,
        },
        order_contributions,
        partial_sums,
    })
}

/// Generate clusters, solve each one, and combine: the full expansion.
///
/// Clusters are processed order by order (increasing size); clusters within
/// one order are solved in parallel.
///
/// # Errors
/// Any error from the generator, the solver, or [`combine`].
pub fn run_nlce<G, M, S>(
    generator: &G,
    solver: &S,
    model: &M,
) -> Result<NlceResult<S::Output>, NlceError>
where
    G: ClusterGenerator,
    M: Model,
    S: ClusterSolver<M>,
{
    run_nlce_cached(generator, solver, model, ClusterCache::default())
}

/// Like [`run_nlce`], but clusters whose property is already in `cache` are
/// not re-solved. The cache must come from the same model and solver.
///
/// # Errors
/// Same as [`run_nlce`].
pub fn run_nlce_cached<G, M, S>(
    generator: &G,
    solver: &S,
    model: &M,
    mut cache: ClusterCache<S::Output>,
) -> Result<NlceResult<S::Output>, NlceError>
where
    G: ClusterGenerator,
    M: Model,
    S: ClusterSolver<M>,
{
    let clusters = generator.clusters()?;
    let mut start = 0;
    while start < clusters.len() {
        let order = clusters[start].order;
        let end = start
            + clusters[start..]
                .iter()
                .take_while(|c| c.order == order)
                .count();
        let solved: Vec<(ClusterKey, S::Output)> = clusters[start..end]
            .par_iter()
            .filter(|c| !cache.properties.contains_key(&c.key))
            .map(|c| Ok((c.key.clone(), solver.solve(&c.graph, model)?)))
            .collect::<Result<_, NlceError>>()?;
        cache.properties.extend(solved);
        start = end;
    }
    combine(&clusters, &cache.properties)
}
