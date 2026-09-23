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

/// Optional vertex and edge labels of a [`SmallGraph`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphLabels {
    /// Label of each vertex.
    pub vertex: Vec<u32>,
    /// `edge[a * n + b]` = label of edge `a–b` (symmetric; ignored for
    /// non-edges).
    pub edge: Vec<u32>,
}

/// A simple undirected graph on at most 32 vertices, as adjacency bitmasks,
/// optionally with vertex and edge labels (e.g. sublattices, bond types).
/// Isomorphisms must preserve labels.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SmallGraph {
    /// `adj[v]` has bit `u` set iff `u–v` is an edge. No self-loops.
    pub adj: Vec<u32>,
    /// Labels; `None` means every label is 0.
    pub labels: Option<Box<GraphLabels>>,
}

impl SmallGraph {
    /// Largest supported vertex count.
    pub const MAX_VERTICES: usize = 32;

    /// Unlabelled graph on `n` vertices with the given edges.
    ///
    /// # Panics
    /// If `n > 32`, an endpoint is out of range, or an edge is a self-loop.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let labelled: Vec<(usize, usize, u32)> = edges.iter().map(|&(a, b)| (a, b, 0)).collect();
        Self::from_labeled_edges(&vec![0; n], &labelled)
    }

    /// Labelled graph: `vertex_labels[v]` for each vertex and `(a, b, label)`
    /// edges. Stored unlabelled if every label is 0.
    ///
    /// # Panics
    /// As [`from_edges`](Self::from_edges).
    pub fn from_labeled_edges(vertex_labels: &[u32], edges: &[(usize, usize, u32)]) -> Self {
        let n = vertex_labels.len();
        assert!(n <= Self::MAX_VERTICES, "at most 32 vertices, got {n}");
        let mut adj = vec![0u32; n];
        for &(a, b, _) in edges {
            assert!(a < n && b < n && a != b, "bad edge ({a}, {b})");
            adj[a] |= 1 << b;
            adj[b] |= 1 << a;
        }
        // The label table is only allocated when some label is non-zero.
        let labelled = vertex_labels.iter().any(|&l| l != 0) || edges.iter().any(|e| e.2 != 0);
        let labels = labelled.then(|| {
            let mut edge = vec![0u32; n * n];
            for &(a, b, l) in edges {
                edge[a * n + b] = l;
                edge[b * n + a] = l;
            }
            Box::new(GraphLabels {
                vertex: vertex_labels.to_vec(),
                edge,
            })
        });
        Self { adj, labels }
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

    /// Label of vertex `v` (0 if unlabelled).
    pub fn vertex_label(&self, v: usize) -> u32 {
        self.labels.as_ref().map_or(0, |l| l.vertex[v])
    }

    /// Label of edge `a–b` (0 if unlabelled; meaningless for non-edges).
    pub fn edge_label(&self, a: usize, b: usize) -> u32 {
        self.labels
            .as_ref()
            .map_or(0, |l| l.edge[a * self.adj.len() + b])
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

    /// Edges `(a, b, label)` with `a < b`, in lexicographic order.
    pub fn labeled_edges(&self) -> Vec<(usize, usize, u32)> {
        self.edges()
            .into_iter()
            .map(|(a, b)| (a, b, self.edge_label(a, b)))
            .collect()
    }

    /// Vertex labels (all 0 if unlabelled).
    pub fn vertex_labels(&self) -> Vec<u32> {
        (0..self.n_vertices())
            .map(|v| self.vertex_label(v))
            .collect()
    }

    /// The graph with vertex `v` renamed to `perm[v]`.
    pub fn relabel(&self, perm: &[usize]) -> Self {
        let n = self.n_vertices();
        let mut vl = vec![0u32; n];
        for (v, &p) in perm.iter().enumerate() {
            vl[p] = self.vertex_label(v);
        }
        let edges: Vec<(usize, usize, u32)> = self
            .labeled_edges()
            .into_iter()
            .map(|(a, b, l)| (perm[a], perm[b], l))
            .collect();
        Self::from_labeled_edges(&vl, &edges)
    }
}

/// Result of [`canonical_form`].
#[derive(Clone, Debug)]
pub struct Canonical {
    /// The canonical code: `[n, row_0, …, row_{n−1}]`, the adjacency rows of
    /// the canonically relabelled graph, followed for labelled graphs by
    /// [`LABEL_MARK`], the vertex labels, and the edge labels in canonical
    /// edge order. Equal iff the graphs are isomorphic (labels included);
    /// [`from_code`] rebuilds the canonical graph from it.
    pub code: Vec<u64>,
    /// Canonical relabelling: vertex `v` of the input becomes `labeling[v]`.
    pub labeling: Vec<usize>,
    /// Automorphisms of the *canonical* graph (non-identity, label
    /// preserving, forming a group with the identity). Empty unless
    /// requested.
    pub automorphisms: Vec<Vec<usize>>,
}

/// Separator between adjacency rows and labels in a canonical code (never a
/// valid row: rows use at most 32 bits).
pub const LABEL_MARK: u64 = u64::MAX;

/// Rebuild the canonical graph encoded by a [`Canonical::code`].
pub fn from_code(code: &[u64]) -> SmallGraph {
    let n = code[0] as usize;
    let adj: Vec<u32> = code[1..=n].iter().map(|&r| r as u32).collect();
    let g = SmallGraph { adj, labels: None };
    if code.len() == n + 1 {
        return g;
    }
    debug_assert_eq!(code[n + 1], LABEL_MARK);
    let vl: Vec<u32> = code[n + 2..2 * n + 2].iter().map(|&l| l as u32).collect();
    let edges: Vec<(usize, usize, u32)> = g
        .edges()
        .into_iter()
        .zip(&code[2 * n + 2..])
        .map(|((a, b), &l)| (a, b, l as u32))
        .collect();
    SmallGraph::from_labeled_edges(&vl, &edges)
}

/// Element of a refinement key: a neighbour's colour, combined with the
/// connecting edge's label rank for labelled graphs. Unlabelled graphs use
/// narrow `u8` keys (sorting them is the canonicaliser's inner loop);
/// whether a graph is labelled does not depend on vertex names, so each
/// graph consistently takes one path.
trait KeyElem: Copy + Ord {
    const PAD: Self;
    fn make(edge_rank: u16, color: u32) -> Self;
}

impl KeyElem for u8 {
    const PAD: Self = u8::MAX;
    fn make(_edge_rank: u16, color: u32) -> Self {
        color as u8
    }
}

impl KeyElem for u16 {
    const PAD: Self = u16::MAX;
    fn make(edge_rank: u16, color: u32) -> Self {
        (edge_rank << 5) | color as u16
    }
}

/// Per-graph data shared by refinement and the search.
struct Ctx<'a> {
    g: &'a SmallGraph,
    /// Edge-label ranks (`n × n`), `None` if unlabelled.
    erank: Option<Vec<u16>>,
}

impl Ctx<'_> {
    /// Refine `colors` (ranks `0..k`) to the coarsest stable partition finer
    /// than it. New ranks are assigned in order of the key (colour, sorted
    /// neighbour entries), which is independent of vertex names.
    fn refine(&self, colors: &mut [u32]) {
        if self.erank.is_some() {
            self.refine_with::<u16>(colors);
        } else {
            self.refine_with::<u8>(colors);
        }
    }

    fn refine_with<T: KeyElem>(&self, colors: &mut [u32]) {
        const MAX: usize = SmallGraph::MAX_VERTICES;
        let n = colors.len();
        let mut n_cells = count_cells(colors);
        let mut keys: [((u8, [T; MAX]), u8); MAX] = [((0, [T::PAD; MAX]), 0); MAX];
        loop {
            for (v, slot) in keys.iter_mut().enumerate().take(n) {
                let mut nb = [T::PAD; MAX];
                let mut len = 0;
                let mut m = self.g.adj[v];
                while m != 0 {
                    let u = m.trailing_zeros() as usize;
                    let lab = self.erank.as_ref().map_or(0, |r| r[v * n + u]);
                    nb[len] = T::make(lab, colors[u]);
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

    /// Code of `g` under `labeling` (old vertex → new index).
    fn code(&self, labeling: &[u32]) -> Vec<u64> {
        let g = self.g;
        let n = labeling.len();
        let mut rows = vec![0u32; n];
        for (v, &row) in g.adj.iter().enumerate() {
            let mut m = row;
            let mut r = 0u32;
            while m != 0 {
                r |= 1 << labeling[m.trailing_zeros() as usize];
                m &= m - 1;
            }
            rows[labeling[v] as usize] = r;
        }
        let mut code = Vec::with_capacity(2 * n + 2);
        code.push(n as u64);
        code.extend(rows.iter().map(|&r| r as u64));
        if g.labels.is_some() {
            let mut inv = vec![0usize; n];
            for (v, &l) in labeling.iter().enumerate() {
                inv[l as usize] = v;
            }
            code.push(LABEL_MARK);
            code.extend((0..n).map(|i| g.vertex_label(inv[i]) as u64));
            for (a, &row) in rows.iter().enumerate() {
                let mut m = row & !((2u32 << a) - 1);
                while m != 0 {
                    let b = m.trailing_zeros() as usize;
                    code.push(g.edge_label(inv[a], inv[b]) as u64);
                    m &= m - 1;
                }
            }
        }
        code
    }
}

fn count_cells(colors: &[u32]) -> usize {
    colors.iter().max().map_or(0, |&m| m as usize + 1)
}

struct Search<'a> {
    ctx: Ctx<'a>,
    best: Option<Vec<u64>>,
    best_leaves: Vec<Vec<u32>>,
    want_automorphisms: bool,
}

impl Search<'_> {
    fn descend(&mut self, colors: Vec<u32>) {
        let n = colors.len();
        if count_cells(&colors) == n {
            let code = self.ctx.code(&colors);
            match &self.best {
                Some(b) if code > *b => {}
                Some(b) if code == *b => {
                    if self.want_automorphisms {
                        self.best_leaves.push(colors);
                    }
                }
                _ => {
                    self.best = Some(code);
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
            self.ctx.refine(&mut child);
            self.descend(child);
        }
    }
}

/// Ranks of the distinct values of `labels`, in sorted order (a
/// label-independent relabelling of the label values).
fn ranks(labels: &[u32]) -> Vec<u32> {
    let mut distinct = labels.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    labels
        .iter()
        .map(|l| distinct.binary_search(l).unwrap() as u32)
        .collect()
}

/// Canonical form of `g` (optionally with its automorphism group).
///
/// # Panics
/// If `g` has more than 32 vertices or more than 2048 distinct edge labels.
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
    // Vertex labels seed the colours; edge labels enter the refinement keys.
    let (mut colors, erank) = match &g.labels {
        None => (vec![0u32; n], None),
        Some(l) => {
            let er = ranks(&l.edge);
            assert!(
                er.iter().all(|&r| r < 2048),
                "too many distinct edge labels"
            );
            (
                ranks(&l.vertex),
                Some(er.iter().map(|&r| r as u16).collect::<Vec<u16>>()),
            )
        }
    };
    let ctx = Ctx { g, erank };
    ctx.refine(&mut colors);
    let mut s = Search {
        ctx,
        best: None,
        best_leaves: Vec::new(),
        want_automorphisms: automorphisms,
    };
    s.descend(colors);
    let code = s.best.expect("search reaches at least one leaf");
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
    fn labels_distinguish_and_count_classes() {
        // Path a–b–c with edge labels in {0, 1}: classes {00, 01, 11} (the
        // path's reflection identifies 01 with 10).
        let mut edge_classes = HashSet::new();
        let mut vertex_classes = HashSet::new();
        for x in 0..2 {
            for y in 0..2 {
                let g = SmallGraph::from_labeled_edges(&[0, 0, 0], &[(0, 1, x), (1, 2, y)]);
                edge_classes.insert(canonical_form(&g, false).code);
            }
        }
        // Vertex labels in {0, 1} on the same path: 2 (centre) × 3 (ends).
        for l in 0..8u32 {
            let vl = [l & 1, (l >> 1) & 1, (l >> 2) & 1];
            let g = SmallGraph::from_labeled_edges(&vl, &[(0, 1, 0), (1, 2, 0)]);
            vertex_classes.insert(canonical_form(&g, false).code);
        }
        assert_eq!(edge_classes.len(), 3);
        assert_eq!(vertex_classes.len(), 6);
        // Unlabelled graphs keep their compact code.
        let plain = SmallGraph::from_edges(3, &[(0, 1), (1, 2)]);
        assert_eq!(canonical_form(&plain, false).code.len(), 4);
    }

    #[test]
    fn labelled_round_trip_and_automorphisms() {
        // Square with one distinct bond: only the reflection fixing it.
        let g = SmallGraph::from_labeled_edges(
            &[0, 0, 0, 0],
            &[(0, 1, 7), (1, 2, 0), (2, 3, 0), (3, 0, 0)],
        );
        let c = canonical_form(&g, true);
        assert_eq!(c.automorphisms.len() + 1, 2);
        assert_eq!(g.relabel(&c.labeling), from_code(&c.code));
        for perm in [[3, 2, 1, 0], [1, 2, 3, 0]] {
            assert_eq!(canonical_form(&g.relabel(&perm), false).code, c.code);
        }
        let canon = from_code(&c.code);
        for p in &c.automorphisms {
            assert_eq!(canon.relabel(p), canon);
        }
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
