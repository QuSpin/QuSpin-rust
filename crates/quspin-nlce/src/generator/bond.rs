//! Bond-based topological expansion (Rigol, Bryant & Singh; Tang, Khatami &
//! Rigol, arXiv:1207.3366).
//!
//! Clusters are connected sets of lattice **bonds** (the cluster Hamiltonian
//! contains exactly those bonds, not every lattice bond between its sites),
//! the order is the number of bonds, and embeddings with isomorphic graphs
//! are merged into one **topological** cluster type.
//!
//! # How the cluster types are built
//!
//! 1. **Enumerate embeddings once each (Redelmeier).** Bonds are ordered by
//!    their endpoints (a translation-invariant total order). Every
//!    translation class of connected `n`-bond clusters has exactly one
//!    representative whose smallest bond starts in the unit cell; Redelmeier's
//!    algorithm, run on the line graph (bonds adjacent when they share a site)
//!    from each such root bond while forbidding bonds below the root, visits
//!    each of those representatives exactly once. It is a depth-first
//!    backtracking search: each stack frame adds one bond popped from an
//!    "untried" list of candidates; after the subtree *with* that bond is
//!    explored, the bond stays marked for the rest of the frame so no
//!    sibling can add it again. Every cluster is therefore generated once,
//!    with no hash set of seen clusters. (This implementation copies the
//!    untried list per frame, `O(n²)` words; Redelmeier's in-place version is
//!    `O(n)`.)
//! 2. **Canonicalise.** Each visited embedding is converted to an abstract
//!    graph and canonically labelled ([`crate::canon`]); the code is the
//!    cluster's topology.
//! 3. **Lattice constants.** `L(c)` is the number of embeddings that produced
//!    code `c`, divided by the number of sites per unit cell.
//! 4. **Multiplicities.** Redelmeier again, now on each topology's own line
//!    graph, enumerates its connected proper sub-bond-sets once each;
//!    canonicalising them gives `M(s, c)`. The single site is the order-0
//!    cluster with `M = n_sites(c)`.
//!
//! Step 1 is where the work is: every embedding has to be visited to count
//! `L(c)`, so the gain over "grow by one bond and deduplicate" is that no
//! embedding is produced twice and nothing needs to be stored.

use super::{ClusterGenerator, ClusterType};
use crate::canon::{SmallGraph, canonical_form, from_code};
use crate::error::NlceError;
use crate::graph::{Bond, ClusterGraph, ClusterKey, Topology};
use crate::lattice::Lattice;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, VecDeque};

/// Topological bond expansion up to `max_order` bonds.
#[derive(Clone, Debug)]
pub struct BondGenerator<L> {
    /// The infinite lattice. Its `Site` order must be translation invariant
    /// (true for lexicographic coordinates), and `unit_cell_sites` must hold
    /// exactly one site per translation class.
    pub lattice: L,
    /// Largest number of bonds per cluster.
    pub max_order: usize,
}

/// Embedding statistics of every topology up to the maximum order.
#[derive(Clone, Debug)]
pub struct EmbeddingCensus {
    /// `fixed_clusters[n]`: number of connected `n`-bond clusters per unit
    /// cell, counted up to translation (index 0 unused).
    pub fixed_clusters: Vec<u64>,
    /// Embeddings per unit cell for each topology, keyed by canonical code.
    pub embeddings: BTreeMap<Vec<u64>, u64>,
}

impl<L: Lattice> BondGenerator<L> {
    /// Bond expansion with clusters of up to `max_order` bonds.
    pub fn new(lattice: L, max_order: usize) -> Self {
        Self { lattice, max_order }
    }

    /// Enumerate every embedding up to `max_order` bonds (step 1–3 above).
    ///
    /// # Errors
    /// `InvalidInput` if the unit cell is empty or a cluster would exceed the
    /// canonicaliser's 32-vertex limit.
    pub fn census(&self) -> Result<EmbeddingCensus, NlceError> {
        let n = self.max_order;
        if n + 1 > SmallGraph::MAX_VERTICES {
            return Err(NlceError::InvalidInput(format!(
                "max_order {n} exceeds the {}-vertex canonicaliser limit",
                SmallGraph::MAX_VERTICES
            )));
        }
        let patch = Patch::new(&self.lattice, n)?;
        // Split the search tree at a shallow depth for parallelism.
        let split = n.min(4);
        let mut tasks: Vec<Task> = Vec::new();
        let mut shallow = Walker::new(&patch, n);
        for &root in &patch.roots {
            shallow.run_root(root, Some((split, &mut tasks)));
        }
        let partials: Vec<Walker> = tasks
            .into_par_iter()
            .map(|t| {
                let mut w = Walker::new(&patch, n);
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
}

impl<L: Lattice> ClusterGenerator for BondGenerator<L> {
    fn clusters(&self) -> Result<Vec<ClusterType>, NlceError> {
        self.clusters_from_census(&self.census()?)
    }
}

impl<L: Lattice> BondGenerator<L> {
    /// Build the cluster DAG (step 4 above) from a precomputed
    /// [`census`](Self::census), e.g. to reuse one census for statistics and
    /// for the expansion.
    ///
    /// # Errors
    /// `InconsistentClusters` if a sub-cluster is missing from the census
    /// (the census came from a smaller `max_order` or another lattice).
    pub fn clusters_from_census(
        &self,
        census: &EmbeddingCensus,
    ) -> Result<Vec<ClusterType>, NlceError> {
        let per_cell = self.lattice.unit_cell_sites().len() as f64;
        let site_code = canonical_form(&SmallGraph::from_edges(1, &[]), false).code;
        let key = |code: &[u64]| ClusterKey {
            topology: Topology::Canonical(code.to_vec()),
            bond_labels: Vec::new(),
            site_labels: Vec::new(),
        };

        let mut out: Vec<ClusterType> = vec![ClusterType {
            key: key(&site_code),
            order: 0,
            graph: ClusterGraph::new(1, Vec::new())?,
            lattice_constant: 1.0,
            subclusters: Vec::new(),
        }];
        let built: Vec<ClusterType> = census
            .embeddings
            .par_iter()
            .map(|(code, &count)| {
                let g = from_code(code);
                let n_bonds = g.n_edges();
                let autos = canonical_form(&g, true).automorphisms;
                let bonds: Vec<Bond> = g
                    .edges()
                    .into_iter()
                    .map(|(a, b)| Bond::new(a as u32, b as u32))
                    .collect();
                let mut subclusters: Vec<(ClusterKey, u64)> =
                    vec![(key(&site_code), g.n_vertices() as u64)];
                for (sub, m) in sub_bond_sets(&g) {
                    if !census.embeddings.contains_key(&sub) {
                        return Err(NlceError::InconsistentClusters(format!(
                            "sub-cluster of {} does not embed in the lattice",
                            key(code)
                        )));
                    }
                    subclusters.push((key(&sub), m));
                }
                Ok(ClusterType {
                    key: key(code),
                    order: n_bonds,
                    graph: ClusterGraph::new(g.n_vertices(), bonds)?.with_automorphisms(autos)?,
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

/// Canonical codes of all connected proper sub-bond-sets of `g` (at least one
/// bond) with their multiplicities, via Redelmeier on `g`'s line graph.
fn sub_bond_sets(g: &SmallGraph) -> BTreeMap<Vec<u64>, u64> {
    let edges = g.edges();
    let e = edges.len();
    let mut line_adj = vec![Vec::new(); e];
    for i in 0..e {
        for j in 0..e {
            let (a, b) = edges[i];
            let (c, d) = edges[j];
            if i != j && (a == c || a == d || b == c || b == d) {
                line_adj[i].push(j as u32);
            }
        }
    }
    let ends: Vec<(u32, u32)> = edges.iter().map(|&(a, b)| (a as u32, b as u32)).collect();
    let space = BondSpace {
        line_adj: &line_adj,
        ends: &ends,
        n_sites: g.n_vertices(),
    };
    let mut w = Walker::with_space(space, e.saturating_sub(1));
    for root in 0..e as u32 {
        w.run_root(root, None);
    }
    w.embeddings.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Finite patch of the lattice with integer bond ids
// ---------------------------------------------------------------------------

/// Every site within `n` steps of a unit-cell site, and the bonds among them,
/// with bond ids assigned in the translation-invariant order.
struct Patch {
    line_adj: Vec<Vec<u32>>,
    ends: Vec<(u32, u32)>,
    n_sites: usize,
    roots: Vec<u32>,
}

impl Patch {
    fn new<L: Lattice>(lattice: &L, n: usize) -> Result<Self, NlceError> {
        let cell = lattice.unit_cell_sites();
        if cell.is_empty() {
            return Err(NlceError::InvalidInput(
                "lattice has an empty unit cell".into(),
            ));
        }
        let mut dist: HashMap<L::Site, usize> = HashMap::new();
        let mut queue: VecDeque<L::Site> = VecDeque::new();
        for &s in &cell {
            dist.insert(s, 0);
            queue.push_back(s);
        }
        while let Some(s) = queue.pop_front() {
            let d = dist[&s];
            if d == n {
                continue;
            }
            for (t, _) in lattice.neighbors(s) {
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(t) {
                    e.insert(d + 1);
                    queue.push_back(t);
                }
            }
        }
        let mut sites: Vec<L::Site> = dist.keys().copied().collect();
        sites.sort();
        let index: HashMap<L::Site, u32> = sites
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as u32))
            .collect();
        // Sites are indexed in sorted order, so (min, max) index pairs sort
        // exactly like the lattice bonds.
        let mut bonds: Vec<(u32, u32)> = Vec::new();
        for (i, &s) in sites.iter().enumerate() {
            for (t, _) in lattice.neighbors(s) {
                if let Some(&j) = index.get(&t)
                    && (i as u32) < j
                {
                    bonds.push((i as u32, j));
                }
            }
        }
        bonds.sort_unstable();
        bonds.dedup();
        let mut at_site: Vec<Vec<u32>> = vec![Vec::new(); sites.len()];
        for (b, &(i, j)) in bonds.iter().enumerate() {
            at_site[i as usize].push(b as u32);
            at_site[j as usize].push(b as u32);
        }
        let line_adj: Vec<Vec<u32>> = bonds
            .iter()
            .enumerate()
            .map(|(b, &(i, j))| {
                at_site[i as usize]
                    .iter()
                    .chain(&at_site[j as usize])
                    .copied()
                    .filter(|&c| c != b as u32)
                    .collect()
            })
            .collect();
        let cell_idx: Vec<u32> = cell.iter().map(|s| index[s]).collect();
        let roots: Vec<u32> = bonds
            .iter()
            .enumerate()
            .filter(|(_, (i, _))| cell_idx.contains(i))
            .map(|(b, _)| b as u32)
            .collect();
        Ok(Self {
            line_adj,
            ends: bonds,
            n_sites: sites.len(),
            roots,
        })
    }

    fn space(&self) -> BondSpace<'_> {
        BondSpace {
            line_adj: &self.line_adj,
            ends: &self.ends,
            n_sites: self.n_sites,
        }
    }
}

/// A set of bonds with integer ids ordered for Redelmeier.
#[derive(Clone, Copy)]
struct BondSpace<'a> {
    line_adj: &'a [Vec<u32>],
    ends: &'a [(u32, u32)],
    n_sites: usize,
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
    space: BondSpace<'a>,
    max: usize,
    root: u32,
    reached: Vec<bool>,
    cluster: Vec<u32>,
    site_uses: Vec<u8>,
    fixed: Vec<u64>,
    embeddings: HashMap<Vec<u64>, u64>,
}

impl<'a> Walker<'a> {
    fn new(patch: &'a Patch, max: usize) -> Self {
        Self::with_space(patch.space(), max)
    }

    fn with_space(space: BondSpace<'a>, max: usize) -> Self {
        Self {
            space,
            max,
            root: 0,
            reached: vec![false; space.ends.len()],
            cluster: Vec::with_capacity(max),
            site_uses: vec![0; space.n_sites],
            fixed: vec![0; max + 1],
            embeddings: HashMap::new(),
        }
    }

    /// All clusters whose smallest bond is `root`. With `split = Some((d, _))`
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
        for &b in &t.cluster {
            self.push(b);
        }
        let mut untried = t.untried;
        self.extend(&mut untried, &mut None);
    }

    fn push(&mut self, b: u32) {
        self.cluster.push(b);
        let (i, j) = self.space.ends[b as usize];
        self.site_uses[i as usize] += 1;
        self.site_uses[j as usize] += 1;
    }

    fn pop(&mut self) {
        let b = self.cluster.pop().unwrap();
        let (i, j) = self.space.ends[b as usize];
        self.site_uses[i as usize] -= 1;
        self.site_uses[j as usize] -= 1;
    }

    /// Redelmeier's step: pop each untried bond in turn, add it, record the
    /// cluster, then recurse with the untried set enlarged by its new
    /// neighbours (only bonds above the root, each added at most once).
    fn extend(&mut self, untried: &mut Vec<u32>, split: &mut Option<(usize, &mut Vec<Task>)>) {
        while let Some(b) = untried.pop() {
            self.push(b);
            self.record();
            if self.cluster.len() < self.max {
                let mut next = untried.clone();
                let mut added: Vec<u32> = Vec::new();
                for &c in &self.space.line_adj[b as usize] {
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
            self.pop();
        }
    }

    /// Canonicalise the current cluster and count it.
    fn record(&mut self) {
        self.fixed[self.cluster.len()] += 1;
        let mut local: Vec<u32> = Vec::with_capacity(self.cluster.len() + 1);
        for &b in &self.cluster {
            let (i, j) = self.space.ends[b as usize];
            for s in [i, j] {
                if !local.contains(&s) {
                    local.push(s);
                }
            }
        }
        let pos = |s: u32| local.iter().position(|&x| x == s).unwrap();
        let edges: Vec<(usize, usize)> = self
            .cluster
            .iter()
            .map(|&b| {
                let (i, j) = self.space.ends[b as usize];
                (pos(i), pos(j))
            })
            .collect();
        let code = canonical_form(&SmallGraph::from_edges(local.len(), &edges), false).code;
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
