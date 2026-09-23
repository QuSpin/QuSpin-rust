//! Canonical labelling of small graphs (graph isomorphism classes).
//!
//! Topological NLCE merges every lattice embedding of the same abstract
//! graph into one cluster type, so each embedding needs a *canonical form*:
//! a code that is equal for two graphs exactly when they are isomorphic.
//!
//! The algorithm is the individualisation–refinement scheme behind nauty,
//! without its pruning (clusters here have at most a few dozen vertices):
//!
//! 1. **Colour refinement.** Start from one colour class and repeatedly split
//!    classes by the multiset of neighbour colours, ordering the new classes
//!    by a label-independent key, until the partition is stable (equitable).
//!    For most small graphs (paths, most trees) this already separates every
//!    vertex.
//! 2. **Individualisation.** If a class still has several vertices, branch:
//!    give each vertex of the first such class its own colour in turn and
//!    refine again. Every leaf of this search tree is a discrete partition,
//!    i.e. a relabelling of the vertices.
//! 3. **Canonical code.** Each leaf yields the relabelled adjacency matrix;
//!    the lexicographically smallest one is the canonical form. Leaves that
//!    reach the same minimum differ by an automorphism, so the search also
//!    produces the full automorphism group.
//!
//! Because the refinement keys and the branching order depend only on the
//! graph structure, isomorphic inputs explore the same search tree up to
//! relabelling and end at the same minimum.

/// A simple undirected graph on at most 32 vertices, as adjacency bitmasks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SmallGraph {
    /// `adj[v]` has bit `u` set iff `u–v` is an edge. No self-loops.
    pub adj: Vec<u32>,
}

impl SmallGraph {
    /// Largest supported vertex count.
    pub const MAX_VERTICES: usize = 32;

    /// Graph on `n` vertices with the given edges.
    ///
    /// # Panics
    /// If `n > 32`, an endpoint is out of range, or an edge is a self-loop.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        assert!(n <= Self::MAX_VERTICES, "at most 32 vertices, got {n}");
        let mut adj = vec![0u32; n];
        for &(a, b) in edges {
            assert!(a < n && b < n && a != b, "bad edge ({a}, {b})");
            adj[a] |= 1 << b;
            adj[b] |= 1 << a;
        }
        Self { adj }
    }

    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.adj.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.adj
            .iter()
            .map(|a| a.count_ones() as usize)
            .sum::<usize>()
            / 2
    }

    /// Edges `(a, b)` with `a < b`, in lexicographic order.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (a, &row) in self.adj.iter().enumerate() {
            let mut m = row & !((2u32 << a) - 1); // neighbours b > a
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                out.push((a, b));
                m &= m - 1;
            }
        }
        out
    }

    /// The graph with vertex `v` renamed to `perm[v]`.
    pub fn relabel(&self, perm: &[usize]) -> Self {
        let mut adj = vec![0u32; self.adj.len()];
        for (v, &row) in self.adj.iter().enumerate() {
            let mut m = row;
            while m != 0 {
                let u = m.trailing_zeros() as usize;
                adj[perm[v]] |= 1 << perm[u];
                m &= m - 1;
            }
        }
        Self { adj }
    }
}

/// Result of [`canonical_form`].
#[derive(Clone, Debug)]
pub struct Canonical {
    /// The canonical code: `[n, row_0, …, row_{n−1}]`, the adjacency rows of
    /// the canonically relabelled graph. Equal iff the graphs are isomorphic;
    /// [`from_code`] rebuilds the canonical graph from it.
    pub code: Vec<u64>,
    /// Canonical relabelling: vertex `v` of the input becomes `labeling[v]`.
    pub labeling: Vec<usize>,
    /// Automorphisms of the *canonical* graph (non-identity site maps,
    /// forming a group with the identity). Empty unless requested.
    pub automorphisms: Vec<Vec<usize>>,
}

/// Rebuild the canonical graph encoded by a [`Canonical::code`].
pub fn from_code(code: &[u64]) -> SmallGraph {
    let n = code[0] as usize;
    SmallGraph {
        adj: code[1..=n].iter().map(|&r| r as u32).collect(),
    }
}

/// Refinement key of one vertex: its colour, then its neighbour colours
/// sorted ascending and padded with `u8::MAX`. Any label-independent total
/// order works; this one needs no heap allocation.
type RefineKey = (u8, [u8; SmallGraph::MAX_VERTICES]);

/// Refine `colors` (ranks `0..k`) to the coarsest stable partition finer than
/// it. New ranks are assigned in order of [`RefineKey`], which is independent
/// of vertex labels.
fn refine(g: &SmallGraph, colors: &mut [u32]) {
    let n = colors.len();
    let mut n_cells = count_cells(colors);
    let mut keys: [(RefineKey, u8); SmallGraph::MAX_VERTICES] =
        [((0, [0; SmallGraph::MAX_VERTICES]), 0); SmallGraph::MAX_VERTICES];
    loop {
        for (v, slot) in keys.iter_mut().enumerate().take(n) {
            let mut nb = [u8::MAX; SmallGraph::MAX_VERTICES];
            let mut len = 0;
            let mut m = g.adj[v];
            while m != 0 {
                nb[len] = colors[m.trailing_zeros() as usize] as u8;
                len += 1;
                m &= m - 1;
            }
            nb[..len].sort_unstable();
            *slot = ((colors[v] as u8, nb), v as u8);
        }
        let keys = &mut keys[..n];
        keys.sort_unstable();
        let mut rank = 0u32;
        for i in 0..n {
            if i > 0 && keys[i].0 != keys[i - 1].0 {
                rank += 1;
            }
            colors[keys[i].1 as usize] = rank;
        }
        let cells = rank as usize + 1;
        if cells == n_cells {
            return;
        }
        n_cells = cells;
    }
}

fn count_cells(colors: &[u32]) -> usize {
    colors.iter().max().map_or(0, |&m| m as usize + 1)
}

/// Adjacency rows of `g` under `labeling` (old vertex → new index).
fn relabeled_rows(g: &SmallGraph, labeling: &[u32]) -> Vec<u32> {
    let mut rows = vec![0u32; labeling.len()];
    for (v, &row) in g.adj.iter().enumerate() {
        let mut m = row;
        let mut r = 0u32;
        while m != 0 {
            r |= 1 << labeling[m.trailing_zeros() as usize];
            m &= m - 1;
        }
        rows[labeling[v] as usize] = r;
    }
    rows
}

struct Search<'a> {
    g: &'a SmallGraph,
    best_rows: Option<Vec<u32>>,
    best_leaves: Vec<Vec<u32>>,
    want_automorphisms: bool,
}

impl Search<'_> {
    fn descend(&mut self, colors: Vec<u32>) {
        let n = colors.len();
        if count_cells(&colors) == n {
            let rows = relabeled_rows(self.g, &colors);
            match &self.best_rows {
                Some(b) if rows > *b => {}
                Some(b) if rows == *b => {
                    if self.want_automorphisms {
                        self.best_leaves.push(colors);
                    }
                }
                _ => {
                    self.best_rows = Some(rows);
                    self.best_leaves.clear();
                    self.best_leaves.push(colors);
                }
            }
            return;
        }
        // First non-singleton cell (lowest colour with >1 vertex).
        let mut size = vec![0usize; n];
        for &c in &colors {
            size[c as usize] += 1;
        }
        let target = (0..n).find(|&c| size[c] > 1).unwrap() as u32;
        for v in 0..n {
            if colors[v] != target {
                continue;
            }
            // Individualise v: it keeps colour `target`, the rest of its
            // cell moves just above it; later cells shift up by one.
            let mut child: Vec<u32> = colors
                .iter()
                .enumerate()
                .map(|(u, &c)| {
                    if c > target || (c == target && u != v) {
                        c + 1
                    } else {
                        c
                    }
                })
                .collect();
            refine(self.g, &mut child);
            self.descend(child);
        }
    }
}

/// Canonical form of `g` (optionally with its automorphism group).
///
/// # Panics
/// If `g` has more than 32 vertices.
pub fn canonical_form(g: &SmallGraph, automorphisms: bool) -> Canonical {
    let n = g.n_vertices();
    assert!(n <= SmallGraph::MAX_VERTICES);
    if n == 0 {
        return Canonical {
            code: vec![0],
            labeling: Vec::new(),
            automorphisms: Vec::new(),
        };
    }
    let mut colors = vec![0u32; n];
    refine(g, &mut colors);
    let mut s = Search {
        g,
        best_rows: None,
        best_leaves: Vec::new(),
        want_automorphisms: automorphisms,
    };
    s.descend(colors);
    let rows = s.best_rows.expect("search reaches at least one leaf");
    let leaves = s.best_leaves;
    let labeling: Vec<usize> = leaves[0].iter().map(|&c| c as usize).collect();

    let mut autos = Vec::new();
    if automorphisms {
        // Leaf k maps input vertex v to label π_k(v); on the canonical graph
        // (vertex i = π_0⁻¹(i)) the automorphism is i ↦ π_k(π_0⁻¹(i)).
        let mut inv0 = vec![0usize; n];
        for (v, &l) in labeling.iter().enumerate() {
            inv0[l] = v;
        }
        for leaf in &leaves[1..] {
            let perm: Vec<usize> = (0..n).map(|i| leaf[inv0[i]] as usize).collect();
            if !autos.contains(&perm) {
                autos.push(perm);
            }
        }
    }
    let mut code = Vec::with_capacity(n + 1);
    code.push(n as u64);
    code.extend(rows.iter().map(|&r| r as u64));
    Canonical {
        code,
        labeling,
        automorphisms: autos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// All labelled graphs on `n` vertices, by edge bitmask.
    fn all_graphs(n: usize) -> impl Iterator<Item = SmallGraph> {
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
            .collect();
        (0u64..1 << pairs.len()).map(move |mask| {
            let e: Vec<(usize, usize)> = pairs
                .iter()
                .enumerate()
                .filter(|(i, _)| mask >> i & 1 == 1)
                .map(|(_, &p)| p)
                .collect();
            SmallGraph::from_edges(n, &e)
        })
    }

    #[test]
    fn counts_unlabelled_graphs() {
        // OEIS A000088: 1, 2, 4, 11, 34, 156 graphs on 1..6 vertices.
        for (n, want) in [(1, 1), (2, 2), (3, 4), (4, 11), (5, 34), (6, 156)] {
            let classes: HashSet<Vec<u64>> = all_graphs(n)
                .map(|g| canonical_form(&g, false).code)
                .collect();
            assert_eq!(classes.len(), want, "n = {n}");
        }
    }

    #[test]
    fn invariant_under_relabelling_and_code_round_trips() {
        let g =
            SmallGraph::from_edges(7, &[(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (1, 6)]);
        let c = canonical_form(&g, false);
        for perm in [
            [6, 5, 4, 3, 2, 1, 0],
            [2, 0, 1, 4, 3, 6, 5],
            [1, 2, 3, 4, 5, 6, 0],
        ] {
            assert_eq!(canonical_form(&g.relabel(&perm), false).code, c.code);
        }
        assert_eq!(g.relabel(&c.labeling), from_code(&c.code));
    }

    #[test]
    fn automorphism_group_orders() {
        let path = SmallGraph::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let star = SmallGraph::from_edges(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let square = SmallGraph::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        for (g, order) in [(path, 2), (star, 24), (square, 8)] {
            let c = canonical_form(&g, true);
            assert_eq!(c.automorphisms.len() + 1, order);
            let canon = from_code(&c.code);
            for p in &c.automorphisms {
                assert_eq!(canon.relabel(p), canon);
            }
        }
    }
}
