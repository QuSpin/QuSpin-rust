//! Topological cluster expansions over a pluggable notion of *node*.
//!
//! A cluster is a connected set of **nodes** of the lattice; what a node is
//! decides the expansion ([`NodeKind`]):
//!
//! | kind | node | nodes adjacent when | cluster bonds | order |
//! |---|---|---|---|---|
//! | [`Bonds`] | a lattice bond | they share a site | the chosen bonds | # bonds |
//! | [`Sites`] | a lattice site | a lattice bond joins them | all bonds among the sites (induced) | # sites |
//!
//! Other kinds (square plaquettes, kagome triangles, …) implement the same
//! trait. Embeddings whose (labelled) graphs are isomorphic are merged into
//! one **topological** cluster type (Rigol, Bryant & Singh; Tang, Khatami &
//! Rigol, arXiv:1207.3366).
//!
//! # How the cluster types are built
//!
//! 1. **Enumerate embeddings once each (Redelmeier).** Nodes of a finite
//!    lattice patch are ordered by their sorted site lists (a
//!    translation-invariant total order, given one on sites). Every
//!    translation class of connected `n`-node clusters has exactly one
//!    representative whose smallest node's first site lies in the unit cell;
//!    Redelmeier's algorithm, run on the node adjacency graph from each such
//!    root while forbidding nodes below the root, visits each of those
//!    representatives exactly once. It is a depth-first backtracking search:
//!    each stack frame adds one node popped from an "untried" list of
//!    candidates; after the subtree *with* that node is explored, the node
//!    stays marked for the rest of the frame so no sibling can add it again.
//!    No hash set of seen clusters is needed. (This implementation copies the
//!    untried list per frame, `O(n²)` words; Redelmeier's in-place version is
//!    `O(n)`.)
//! 2. **Canonicalise.** Each embedding's graph, with site and bond labels,
//!    is canonically labelled ([`crate::canon`]); the code is its topology.
//! 3. **Lattice constants.** `L(c)` is the number of embeddings that produced
//!    code `c`, divided by the number of sites per unit cell.
//! 4. **Multiplicities.** The same search on each topology's own graph
//!    (its nodes recovered by [`NodeKind::nodes_of_graph`]) enumerates its
//!    connected proper sub-node-sets once each; canonicalising them gives
//!    `M(s, c)`.
//!
//! Step 1 is where the work is: every embedding has to be visited to count
//! `L(c)`, so the gain over "grow by one node and deduplicate" is that no
//! embedding is produced twice and nothing needs to be stored.

use super::{ClusterGenerator, ClusterType};
use crate::canon::{SmallGraph, canonical_form, from_code};
use crate::error::NlceError;
use crate::graph::{Bond, ClusterGraph, ClusterKey, Topology};
use crate::lattice::Lattice;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Node kinds
// ---------------------------------------------------------------------------

/// A node: a finite set of sites and the bonds it contributes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node<S> {
    /// Sites, sorted ascending.
    pub sites: Vec<S>,
    /// Bonds `(a, b, label)` with `a < b`, sorted.
    pub bonds: Vec<(S, S, u32)>,
}

/// How nodes connect into clusters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Adjacency {
    /// Nodes are adjacent when they share a site (bonds, plaquettes).
    SharedSite,
    /// Nodes are adjacent when they share a site or a lattice bond joins
    /// them (sites).
    SiteOrBond,
}

/// What the nodes of a topological expansion are.
///
/// Implementations must be consistent between the lattice
/// ([`nodes_at`](Self::nodes_at)) and an abstract cluster graph
/// ([`nodes_of_graph`](Self::nodes_of_graph)): the nodes of a cluster found
/// on the lattice must be exactly the nodes recovered from its graph.
pub trait NodeKind: Clone + Debug + Send + Sync {
    /// Short name for descriptions, e.g. `"bond"`.
    fn name(&self) -> &'static str;

    /// Every node that contains `site`.
    fn nodes_at<L: Lattice>(&self, lattice: &L, site: L::Site) -> Vec<Node<L::Site>>;

    /// The nodes of an abstract cluster graph (vertices `0..n`, edges with
    /// labels), used to enumerate sub-clusters.
    ///
    /// # Errors
    /// `Unsupported` if the node structure cannot be recovered from the graph.
    fn nodes_of_graph(&self, g: &SmallGraph) -> Result<Vec<Node<usize>>, NlceError>;

    /// Node adjacency rule.
    fn adjacency(&self) -> Adjacency;

    /// `true` if a cluster contains every lattice bond among its sites,
    /// `false` if only the bonds of its nodes.
    fn induced(&self) -> bool;

    /// Largest lattice distance from a node's first site to any site of an
    /// `order`-node cluster containing it (sizes the finite patch).
    fn reach(&self, order: usize) -> usize;

    /// Largest number of sites of an `order`-node cluster.
    fn max_sites(&self, order: usize) -> usize;

    /// `true` if the expansion needs a separate order-0 single-site term
    /// (clusters with nodes always have more than one site).
    fn single_site_term(&self) -> bool;
}

/// Bond expansion: nodes are lattice bonds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bonds;

impl NodeKind for Bonds {
    fn name(&self) -> &'static str {
        "bond"
    }

    fn nodes_at<L: Lattice>(&self, lattice: &L, site: L::Site) -> Vec<Node<L::Site>> {
        lattice
            .neighbors(site)
            .into_iter()
            .map(|(t, label)| {
                let (a, b) = if site < t { (site, t) } else { (t, site) };
                Node {
                    sites: vec![a, b],
                    bonds: vec![(a, b, label)],
                }
            })
            .collect()
    }

    fn nodes_of_graph(&self, g: &SmallGraph) -> Result<Vec<Node<usize>>, NlceError> {
        Ok(g.labeled_edges()
            .into_iter()
            .map(|(a, b, l)| Node {
                sites: vec![a, b],
                bonds: vec![(a, b, l)],
            })
            .collect())
    }

    fn adjacency(&self) -> Adjacency {
        Adjacency::SharedSite
    }

    fn induced(&self) -> bool {
        false
    }

    fn reach(&self, order: usize) -> usize {
        order
    }

    fn max_sites(&self, order: usize) -> usize {
        order + 1
    }

    fn single_site_term(&self) -> bool {
        true
    }
}

/// Site expansion: nodes are lattice sites; clusters are induced subgraphs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sites;

impl NodeKind for Sites {
    fn name(&self) -> &'static str {
        "site"
    }

    fn nodes_at<L: Lattice>(&self, _lattice: &L, site: L::Site) -> Vec<Node<L::Site>> {
        vec![Node {
            sites: vec![site],
            bonds: Vec::new(),
        }]
    }

    fn nodes_of_graph(&self, g: &SmallGraph) -> Result<Vec<Node<usize>>, NlceError> {
        Ok((0..g.n_vertices())
            .map(|v| Node {
                sites: vec![v],
                bonds: Vec::new(),
            })
            .collect())
    }

    fn adjacency(&self) -> Adjacency {
        Adjacency::SiteOrBond
    }

    fn induced(&self) -> bool {
        true
    }

    fn reach(&self, order: usize) -> usize {
        order.saturating_sub(1)
    }

    fn max_sites(&self, order: usize) -> usize {
        order
    }

    fn single_site_term(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// Topological expansion with node kind `K` up to `max_order` nodes.
#[derive(Clone, Debug)]
pub struct TopologicalGenerator<L, K> {
    /// The infinite lattice. Its `Site` order must be translation invariant
    /// (true for lexicographic coordinates), and `unit_cell_sites` must hold
    /// exactly one site per translation class.
    pub lattice: L,
    /// What a node is.
    pub kind: K,
    /// Largest number of nodes per cluster.
    pub max_order: usize,
}

/// Topological bond expansion ([`Bonds`]).
pub type BondGenerator<L> = TopologicalGenerator<L, Bonds>;

/// Topological site expansion ([`Sites`]).
pub type SiteGenerator<L> = TopologicalGenerator<L, Sites>;

impl<L: Lattice> TopologicalGenerator<L, Bonds> {
    /// Bond expansion with clusters of up to `max_order` bonds.
    pub fn new(lattice: L, max_order: usize) -> Self {
        Self::with_kind(lattice, Bonds, max_order)
    }
}

impl<L: Lattice> TopologicalGenerator<L, Sites> {
    /// Site expansion with clusters of up to `max_order` sites.
    pub fn new(lattice: L, max_order: usize) -> Self {
        Self::with_kind(lattice, Sites, max_order)
    }
}

/// Embedding statistics of every topology up to the maximum order.
#[derive(Clone, Debug)]
pub struct EmbeddingCensus {
    /// `fixed_clusters[n]`: number of connected `n`-node clusters per unit
    /// cell, counted up to translation (index 0 unused).
    pub fixed_clusters: Vec<u64>,
    /// Embeddings per unit cell for each topology, keyed by canonical code.
    pub embeddings: BTreeMap<Vec<u64>, u64>,
}

impl<L: Lattice, K: NodeKind> TopologicalGenerator<L, K> {
    /// Expansion over nodes of kind `kind` with up to `max_order` nodes.
    pub fn with_kind(lattice: L, kind: K, max_order: usize) -> Self {
        Self {
            lattice,
            kind,
            max_order,
        }
    }

    /// Enumerate every embedding up to `max_order` nodes (steps 1–3 above).
    ///
    /// # Errors
    /// `InvalidInput` if the unit cell is empty or a cluster would exceed the
    /// canonicaliser's 32-vertex limit.
    pub fn census(&self) -> Result<EmbeddingCensus, NlceError> {
        let n = self.max_order;
        if self.kind.max_sites(n) > SmallGraph::MAX_VERTICES {
            return Err(NlceError::InvalidInput(format!(
                "max_order {n} exceeds the {}-vertex canonicaliser limit",
                SmallGraph::MAX_VERTICES
            )));
        }
        let (space, roots) = lattice_patch(&self.lattice, &self.kind, n)?;
        // Split the search tree at a shallow depth for parallelism.
        let split = n.min(4);
        let mut tasks: Vec<Task> = Vec::new();
        let mut shallow = Walker::new(&space, n);
        for &root in &roots {
            shallow.run_root(root, Some((split, &mut tasks)));
        }
        let partials: Vec<Walker> = tasks
            .into_par_iter()
            .map(|t| {
                let mut w = Walker::new(&space, n);
                w.resume(t);
                w
            })
            .collect();
        let mut fixed = shallow.fixed;
        let mut embeddings = shallow.embeddings;
        for w in partials {
            for (k, v) in w.fixed.iter().enumerate() {
                fixed[k] += v;
            }
            for (code, c) in w.embeddings {
                *embeddings.entry(code).or_default() += c;
            }
        }
        Ok(EmbeddingCensus {
            fixed_clusters: fixed,
            embeddings: embeddings.into_iter().collect(),
        })
    }

    /// Build the cluster DAG (step 4 above) from a precomputed
    /// [`census`](Self::census), e.g. to reuse one census for statistics and
    /// for the expansion.
    ///
    /// # Errors
    /// `InconsistentClusters` if a sub-cluster is missing from the census
    /// (the census came from a smaller `max_order` or another lattice);
    /// errors from [`NodeKind::nodes_of_graph`].
    pub fn clusters_from_census(
        &self,
        census: &EmbeddingCensus,
    ) -> Result<Vec<ClusterType>, NlceError> {
        let per_cell = self.lattice.unit_cell_sites().len() as f64;
        let key = |code: &[u64]| ClusterKey {
            topology: Topology::Canonical(code.to_vec()),
            bond_labels: Vec::new(),
            site_labels: Vec::new(),
        };
        // Order-0 single-site terms, one per site label in the unit cell.
        let mut site_terms: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
        let mut out: Vec<ClusterType> = Vec::new();
        if self.kind.single_site_term() {
            let mut per_label: BTreeMap<u32, f64> = BTreeMap::new();
            for s in self.lattice.unit_cell_sites() {
                *per_label.entry(self.lattice.site_label(s)).or_default() += 1.0;
            }
            for (l, count) in per_label {
                let code = canonical_form(&SmallGraph::from_labeled_edges(&[l], &[]), false).code;
                let mut graph = ClusterGraph::new(1, Vec::new())?;
                graph.site_labels = vec![l];
                out.push(ClusterType {
                    key: key(&code),
                    order: 0,
                    graph,
                    lattice_constant: count / per_cell,
                    subclusters: Vec::new(),
                });
                site_terms.insert(l, code);
            }
        }
        let built: Vec<ClusterType> = census
            .embeddings
            .par_iter()
            .map(|(code, &count)| {
                let g = from_code(code);
                let nodes = self.kind.nodes_of_graph(&g)?;
                let order = nodes.len();
                let autos = canonical_form(&g, true).automorphisms;
                let bonds: Vec<Bond> = g
                    .labeled_edges()
                    .into_iter()
                    .map(|(a, b, l)| Bond {
                        i: a as u32,
                        j: b as u32,
                        label: l,
                    })
                    .collect();
                let mut subclusters: Vec<(ClusterKey, u64)> = Vec::new();
                if self.kind.single_site_term() {
                    let mut per_label: BTreeMap<u32, u64> = BTreeMap::new();
                    for v in 0..g.n_vertices() {
                        *per_label.entry(g.vertex_label(v)).or_default() += 1;
                    }
                    for (l, m) in per_label {
                        let code = site_terms.get(&l).ok_or_else(|| {
                            NlceError::InconsistentClusters(format!(
                                "site label {l} of {} is not in the unit cell",
                                key(code)
                            ))
                        })?;
                        subclusters.push((key(code), m));
                    }
                }
                for (sub, m) in sub_node_sets(&self.kind, &g, &nodes)? {
                    if !census.embeddings.contains_key(&sub) {
                        return Err(NlceError::InconsistentClusters(format!(
                            "sub-cluster of {} does not embed in the lattice",
                            key(code)
                        )));
                    }
                    subclusters.push((key(&sub), m));
                }
                let mut graph = ClusterGraph::new(g.n_vertices(), bonds)?;
                graph.site_labels = g.vertex_labels();
                Ok(ClusterType {
                    key: key(code),
                    order,
                    graph: graph.with_automorphisms(autos)?,
                    lattice_constant: count as f64 / per_cell,
                    subclusters,
                })
            })
            .collect::<Result<_, NlceError>>()?;
        out.extend(built);
        out.sort_by(|a, b| {
            (a.order, a.graph.n_sites, &a.key).cmp(&(b.order, b.graph.n_sites, &b.key))
        });
        Ok(out)
    }
}

impl<L: Lattice, K: NodeKind> ClusterGenerator for TopologicalGenerator<L, K> {
    fn clusters(&self) -> Result<Vec<ClusterType>, NlceError> {
        self.clusters_from_census(&self.census()?)
    }
}

/// Canonical codes of all connected proper sub-node-sets of the cluster
/// graph `g` (with nodes `nodes`) and their multiplicities.
fn sub_node_sets<K: NodeKind>(
    kind: &K,
    g: &SmallGraph,
    nodes: &[Node<usize>],
) -> Result<BTreeMap<Vec<u64>, u64>, NlceError> {
    let mut nodes: Vec<Node<u32>> = nodes.iter().map(to_u32).collect();
    nodes.sort();
    let bonds: Vec<(u32, u32, u32)> = g
        .labeled_edges()
        .into_iter()
        .map(|(a, b, l)| (a as u32, b as u32, l))
        .collect();
    let space = NodeSpace::build(
        g.vertex_labels(),
        bonds,
        nodes,
        kind.adjacency(),
        kind.induced(),
    );
    let n = space.node_sites.len();
    let mut w = Walker::new(&space, n.saturating_sub(1));
    for root in 0..n as u32 {
        w.run_root(root, None);
    }
    Ok(w.embeddings.into_iter().collect())
}

fn to_u32(n: &Node<usize>) -> Node<u32> {
    Node {
        sites: n.sites.iter().map(|&s| s as u32).collect(),
        bonds: n
            .bonds
            .iter()
            .map(|&(a, b, l)| (a as u32, b as u32, l))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Node spaces: a finite lattice patch or an abstract cluster graph
// ---------------------------------------------------------------------------

/// Nodes with integer ids (in the translation-invariant order), their
/// adjacency, and what each contributes to a cluster graph.
struct NodeSpace {
    adj: Vec<Vec<u32>>,
    node_sites: Vec<Vec<u32>>,
    node_bonds: Vec<Vec<(u32, u32, u32)>>,
    site_labels: Vec<u32>,
    /// Per site: `(neighbour, label)` of every bond (used when `induced`).
    site_bonds: Vec<Vec<(u32, u32)>>,
    induced: bool,
    /// Some site or bond label is non-zero.
    labelled: bool,
}

impl NodeSpace {
    /// `nodes` must already be sorted.
    fn build(
        site_labels: Vec<u32>,
        bonds: Vec<(u32, u32, u32)>,
        nodes: Vec<Node<u32>>,
        adjacency: Adjacency,
        induced: bool,
    ) -> Self {
        let n_sites = site_labels.len();
        let mut site_bonds: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_sites];
        for &(a, b, l) in &bonds {
            site_bonds[a as usize].push((b, l));
            site_bonds[b as usize].push((a, l));
        }
        let mut at_site: Vec<Vec<u32>> = vec![Vec::new(); n_sites];
        for (k, node) in nodes.iter().enumerate() {
            for &s in &node.sites {
                at_site[s as usize].push(k as u32);
            }
        }
        let adj: Vec<Vec<u32>> = nodes
            .iter()
            .enumerate()
            .map(|(k, node)| {
                let mut nb: BTreeSet<u32> = BTreeSet::new();
                for &s in &node.sites {
                    nb.extend(&at_site[s as usize]);
                    if adjacency == Adjacency::SiteOrBond {
                        for &(t, _) in &site_bonds[s as usize] {
                            nb.extend(&at_site[t as usize]);
                        }
                    }
                }
                nb.remove(&(k as u32));
                nb.into_iter().collect()
            })
            .collect();
        let labelled = site_labels.iter().any(|&l| l != 0) || bonds.iter().any(|b| b.2 != 0);
        let (node_sites, node_bonds) = nodes.into_iter().map(|n| (n.sites, n.bonds)).unzip();
        Self {
            labelled,
            adj,
            node_sites,
            node_bonds,
            site_labels,
            site_bonds,
            induced,
        }
    }
}

/// The finite patch of `lattice` that holds every `n`-node cluster rooted in
/// the unit cell, with the root node ids.
fn lattice_patch<L: Lattice, K: NodeKind>(
    lattice: &L,
    kind: &K,
    n: usize,
) -> Result<(NodeSpace, Vec<u32>), NlceError> {
    let cell = lattice.unit_cell_sites();
    if cell.is_empty() {
        return Err(NlceError::InvalidInput(
            "lattice has an empty unit cell".into(),
        ));
    }
    let depth = kind.reach(n);
    let mut dist: HashMap<L::Site, usize> = HashMap::new();
    let mut queue: VecDeque<L::Site> = VecDeque::new();
    for &s in &cell {
        dist.insert(s, 0);
        queue.push_back(s);
    }
    while let Some(s) = queue.pop_front() {
        let d = dist[&s];
        if d == depth {
            continue;
        }
        for (t, _) in lattice.neighbors(s) {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(t) {
                e.insert(d + 1);
                queue.push_back(t);
            }
        }
    }
    // Sites are indexed in sorted order, so index order is site order and
    // sorted index lists order nodes translation-invariantly.
    let mut sites: Vec<L::Site> = dist.keys().copied().collect();
    sites.sort();
    let index: HashMap<L::Site, u32> = sites
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as u32))
        .collect();
    let mut bonds: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    for (i, &s) in sites.iter().enumerate() {
        for (t, label) in lattice.neighbors(s) {
            if let Some(&j) = index.get(&t)
                && (i as u32) < j
            {
                bonds.insert((i as u32, j, label));
            }
        }
    }
    let mut nodes: BTreeSet<Node<u32>> = BTreeSet::new();
    for &s in &sites {
        'node: for node in kind.nodes_at(lattice, s) {
            let mut ns = Vec::with_capacity(node.sites.len());
            for t in &node.sites {
                match index.get(t) {
                    Some(&i) => ns.push(i),
                    None => continue 'node,
                }
            }
            let mut nb = Vec::with_capacity(node.bonds.len());
            for (a, b, l) in &node.bonds {
                match (index.get(a), index.get(b)) {
                    (Some(&i), Some(&j)) => nb.push((i.min(j), i.max(j), *l)),
                    _ => continue 'node,
                }
            }
            ns.sort_unstable();
            nb.sort_unstable();
            nodes.insert(Node {
                sites: ns,
                bonds: nb,
            });
        }
    }
    let nodes: Vec<Node<u32>> = nodes.into_iter().collect();
    let cell_idx: Vec<u32> = cell.iter().map(|s| index[s]).collect();
    let roots: Vec<u32> = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| cell_idx.contains(&node.sites[0]))
        .map(|(k, _)| k as u32)
        .collect();
    let site_labels: Vec<u32> = sites.iter().map(|&s| lattice.site_label(s)).collect();
    let space = NodeSpace::build(
        site_labels,
        bonds.into_iter().collect(),
        nodes,
        kind.adjacency(),
        kind.induced(),
    );
    Ok((space, roots))
}

// ---------------------------------------------------------------------------
// Redelmeier enumeration
// ---------------------------------------------------------------------------

/// A search subtree to finish in parallel: the state right after the prefix
/// cluster was recorded, before its children are explored.
struct Task {
    root: u32,
    cluster: Vec<u32>,
    untried: Vec<u32>,
    reached: Vec<u32>,
}

struct Walker<'a> {
    space: &'a NodeSpace,
    max: usize,
    root: u32,
    reached: Vec<bool>,
    cluster: Vec<u32>,
    fixed: Vec<u64>,
    embeddings: HashMap<Vec<u64>, u64>,
}

impl<'a> Walker<'a> {
    fn new(space: &'a NodeSpace, max: usize) -> Self {
        Self {
            space,
            max,
            root: 0,
            reached: vec![false; space.node_sites.len()],
            cluster: Vec::with_capacity(max),
            fixed: vec![0; max + 1],
            embeddings: HashMap::new(),
        }
    }

    /// All clusters whose smallest node is `root`. With `split = Some((d, _))`
    /// the subtrees below depth `d` are handed out as tasks instead.
    fn run_root(&mut self, root: u32, mut split: Option<(usize, &mut Vec<Task>)>) {
        if self.max == 0 {
            return;
        }
        self.root = root;
        self.reached[root as usize] = true;
        let mut untried = vec![root];
        self.extend(&mut untried, &mut split);
        self.reached[root as usize] = false;
    }

    fn resume(&mut self, t: Task) {
        self.root = t.root;
        for &b in &t.reached {
            self.reached[b as usize] = true;
        }
        self.cluster = t.cluster;
        let mut untried = t.untried;
        self.extend(&mut untried, &mut None);
    }

    /// Redelmeier's step: pop each untried node in turn, add it, record the
    /// cluster, then recurse with the untried set enlarged by its new
    /// neighbours (only nodes above the root, each added at most once).
    fn extend(&mut self, untried: &mut Vec<u32>, split: &mut Option<(usize, &mut Vec<Task>)>) {
        while let Some(b) = untried.pop() {
            self.cluster.push(b);
            self.record();
            if self.cluster.len() < self.max {
                let mut next = untried.clone();
                let mut added: Vec<u32> = Vec::new();
                for &c in &self.space.adj[b as usize] {
                    if c > self.root && !self.reached[c as usize] {
                        self.reached[c as usize] = true;
                        next.push(c);
                        added.push(c);
                    }
                }
                match split {
                    Some((depth, tasks)) if self.cluster.len() == *depth => {
                        tasks.push(Task {
                            root: self.root,
                            cluster: self.cluster.clone(),
                            untried: next,
                            reached: (0..self.reached.len() as u32)
                                .filter(|&c| self.reached[c as usize])
                                .collect(),
                        });
                    }
                    _ => self.extend(&mut next, split),
                }
                for c in added {
                    self.reached[c as usize] = false;
                }
            }
            self.cluster.pop();
        }
    }

    /// Build the current cluster's labelled graph, canonicalise it, count it.
    fn record(&mut self) {
        let space = self.space;
        self.fixed[self.cluster.len()] += 1;
        let mut local: Vec<u32> = Vec::with_capacity(self.cluster.len() + 1);
        for &k in &self.cluster {
            for &s in &space.node_sites[k as usize] {
                if !local.contains(&s) {
                    local.push(s);
                }
            }
        }
        let pos = |s: u32| local.iter().position(|&x| x == s);
        let mut edges: Vec<(usize, usize, u32)> = Vec::with_capacity(local.len() + 4);
        if space.induced {
            for (i, &s) in local.iter().enumerate() {
                for &(t, l) in &space.site_bonds[s as usize] {
                    if let Some(j) = pos(t)
                        && i < j
                    {
                        edges.push((i, j, l));
                    }
                }
            }
        } else {
            for &k in &self.cluster {
                for &(a, b, l) in &space.node_bonds[k as usize] {
                    let e = (pos(a).unwrap(), pos(b).unwrap(), l);
                    if !edges.contains(&e) {
                        edges.push(e);
                    }
                }
            }
        }
        let labels: Vec<u32> = if space.labelled {
            local
                .iter()
                .map(|&s| space.site_labels[s as usize])
                .collect()
        } else {
            vec![0; local.len()]
        };
        let code = canonical_form(&SmallGraph::from_labeled_edges(&labels, &edges), false).code;
        *self.embeddings.entry(code).or_default() += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{ChainLattice, SquareLattice};
    use std::collections::HashSet;

    /// Brute force: grow every cluster by one bond and deduplicate by
    /// translation-normalised bond sets.
    fn brute_force_fixed(n: usize) -> Vec<u64> {
        type B = ([i32; 2], [i32; 2]);
        let norm = |set: &[B]| -> Vec<B> {
            let mx = set.iter().map(|b| b.0[0].min(b.1[0])).min().unwrap();
            let my = set.iter().map(|b| b.0[1].min(b.1[1])).min().unwrap();
            let mut v: Vec<B> = set
                .iter()
                .map(|&(a, b)| ([a[0] - mx, a[1] - my], [b[0] - mx, b[1] - my]))
                .collect();
            v.sort();
            v
        };
        let mut level: HashSet<Vec<B>> = [vec![([0, 0], [1, 0])], vec![([0, 0], [0, 1])]]
            .into_iter()
            .collect();
        let mut out = vec![0, level.len() as u64];
        for _ in 2..=n {
            let mut next = HashSet::new();
            for c in &level {
                for &(a, b) in c {
                    for s in [a, b] {
                        for t in [
                            [s[0] + 1, s[1]],
                            [s[0] - 1, s[1]],
                            [s[0], s[1] + 1],
                            [s[0], s[1] - 1],
                        ] {
                            let nb = if s < t { (s, t) } else { (t, s) };
                            if !c.contains(&nb) {
                                let mut g = c.clone();
                                g.push(nb);
                                next.insert(norm(&g));
                            }
                        }
                    }
                }
            }
            out.push(next.len() as u64);
            level = next;
        }
        out
    }

    #[test]
    fn redelmeier_matches_brute_force() {
        let census = BondGenerator::new(SquareLattice, 7).census().unwrap();
        assert_eq!(census.fixed_clusters, brute_force_fixed(7));
        // Every embedding belongs to exactly one topology.
        let total: u64 = census.embeddings.values().sum();
        assert_eq!(total, census.fixed_clusters.iter().sum::<u64>());
    }

    #[test]
    fn chain_has_one_path_per_order() {
        let census = BondGenerator::new(ChainLattice, 6).census().unwrap();
        assert_eq!(census.fixed_clusters, vec![0, 1, 1, 1, 1, 1, 1]);
        assert_eq!(census.embeddings.len(), 6);
    }

    fn sub_bond_sets(g: &SmallGraph) -> BTreeMap<Vec<u64>, u64> {
        sub_node_sets(&Bonds, g, &Bonds.nodes_of_graph(g).unwrap()).unwrap()
    }

    fn cluster<'a>(cs: &'a [ClusterType], edges: &[(usize, usize)], n: usize) -> &'a ClusterType {
        let code = canonical_form(&SmallGraph::from_edges(n, edges), false).code;
        cs.iter()
            .find(|c| c.key.topology == Topology::Canonical(code.clone()))
            .unwrap()
    }

    /// All connected proper edge subsets of `g` by exhaustive enumeration.
    fn brute_force_subsets(g: &SmallGraph) -> BTreeMap<Vec<u64>, u64> {
        let edges = g.edges();
        let e = edges.len();
        let mut out = BTreeMap::new();
        for mask in 1u32..(1 << e) - 1 {
            let chosen: Vec<(usize, usize)> = (0..e)
                .filter(|i| mask >> i & 1 == 1)
                .map(|i| edges[i])
                .collect();
            let mut verts: Vec<usize> = chosen.iter().flat_map(|&(a, b)| [a, b]).collect();
            verts.sort_unstable();
            verts.dedup();
            let pos = |v: usize| verts.iter().position(|&x| x == v).unwrap();
            let local: Vec<(usize, usize)> =
                chosen.iter().map(|&(a, b)| (pos(a), pos(b))).collect();
            let h = SmallGraph::from_edges(verts.len(), &local);
            let bonds = local
                .iter()
                .map(|&(a, b)| Bond::new(a as u32, b as u32))
                .collect();
            if ClusterGraph::new(verts.len(), bonds)
                .unwrap()
                .is_connected()
            {
                *out.entry(canonical_form(&h, false).code).or_default() += 1;
            }
        }
        out
    }

    #[test]
    fn sub_bond_sets_match_brute_force() {
        let grid = |w: usize, h: usize| {
            let mut e = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    if x + 1 < w {
                        e.push((x + w * y, x + 1 + w * y));
                    }
                    if y + 1 < h {
                        e.push((x + w * y, x + w * (y + 1)));
                    }
                }
            }
            SmallGraph::from_edges(w * h, &e)
        };
        let tree =
            SmallGraph::from_edges(8, &[(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (4, 6), (6, 7)]);
        for g in [grid(2, 2), grid(3, 2), grid(3, 3), tree] {
            assert_eq!(sub_bond_sets(&g), brute_force_subsets(&g));
        }
    }

    #[test]
    fn small_lattice_constants_and_multiplicities() {
        let cs = BondGenerator::new(SquareLattice, 4).clusters().unwrap();
        let bond = cluster(&cs, &[(0, 1)], 2);
        let p3 = cluster(&cs, &[(0, 1), (1, 2)], 3);
        let p4 = cluster(&cs, &[(0, 1), (1, 2), (2, 3)], 4);
        let star = cluster(&cs, &[(0, 1), (0, 2), (0, 3)], 4);
        let square = cluster(&cs, &[(0, 1), (1, 2), (2, 3), (3, 0)], 4);
        assert_eq!(cs[0].order, 0);
        assert_eq!(bond.lattice_constant, 2.0);
        assert_eq!(p3.lattice_constant, 6.0); // 2 straight + 4 bent
        assert_eq!(p4.lattice_constant, 18.0);
        assert_eq!(star.lattice_constant, 4.0); // T shapes
        assert_eq!(square.lattice_constant, 1.0);
        let m = |c: &ClusterType, s: &ClusterType| {
            c.subclusters
                .iter()
                .find(|(k, _)| *k == s.key)
                .map_or(0, |(_, v)| *v)
        };
        assert_eq!(m(square, &cs[0]), 4);
        assert_eq!(m(square, bond), 4);
        assert_eq!(m(square, p3), 4);
        assert_eq!(m(square, p4), 4);
        assert_eq!(m(star, p3), 3);
        assert_eq!(m(star, bond), 3);
        assert_eq!(square.graph.automorphisms.len(), 7);
    }
}
