//! Rectangle expansion (Gan & Hazzard, arXiv:2005.03177).

use super::{ClusterGenerator, ClusterType};
use crate::error::NlceError;
use crate::graph::ClusterKey;
use crate::lattice::Lattice;

/// How the expansion order of an `m × n` rectangle is measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RectangleOrder {
    /// `order = m + n` (the usual rectangle-expansion order).
    #[default]
    SemiPerimeter,
    /// `order = m · n` (number of sites).
    Sites,
}

/// All `m × n` open rectangles up to a maximum order, on a 2D-coordinate
/// lattice.
///
/// Each shape is solved once and counted in every orientation the lattice's
/// point group produces, so on the square lattice `L(m×n) = 2` for `m ≠ n`
/// and `1` for `m = n`. Sub-cluster multiplicities are analytic: an
/// oriented `w' × h'` rectangle fits `(w − w' + 1)(h − h' + 1)` times into a
/// `w × h` one, summed over the distinct orientations of the sub-rectangle.
/// Rectangles whose induced graph is disconnected on the lattice (e.g.
/// `2 × n` on [`ChainLattice`](crate::lattice::ChainLattice)) are skipped.
#[derive(Clone, Debug)]
pub struct RectangleGenerator<L> {
    /// The infinite lattice.
    pub lattice: L,
    /// Largest order included.
    pub max_order: usize,
    /// Order measure (default `m + n`).
    pub order_by: RectangleOrder,
    /// Optional cap on the number of sites per cluster.
    pub max_sites: Option<usize>,
}

impl<L: Lattice<Site = [i32; 2]>> RectangleGenerator<L> {
    /// Rectangles with `m + n <= max_order` and no site cap.
    pub fn new(lattice: L, max_order: usize) -> Self {
        Self {
            lattice,
            max_order,
            order_by: RectangleOrder::SemiPerimeter,
            max_sites: None,
        }
    }

    /// Use a different order measure.
    pub fn with_order(mut self, order_by: RectangleOrder) -> Self {
        self.order_by = order_by;
        self
    }

    /// Exclude rectangles with more than `max_sites` sites.
    pub fn with_max_sites(mut self, max_sites: usize) -> Self {
        self.max_sites = Some(max_sites);
        self
    }

    fn order_of(&self, m: usize, n: usize) -> usize {
        match self.order_by {
            RectangleOrder::SemiPerimeter => m + n,
            RectangleOrder::Sites => m * n,
        }
    }

    /// Sites of a `width × height` rectangle anchored at the origin.
    fn rect_sites(width: usize, height: usize) -> Vec<[i32; 2]> {
        (0..width as i32)
            .flat_map(|x| (0..height as i32).map(move |y| [x, y]))
            .collect()
    }

    /// Bounding boxes `(w, h)` of the distinct orientations of an `m × n`
    /// rectangle (realised as `n` wide, `m` high).
    fn oriented_extents(&self, m: usize, n: usize) -> Vec<(usize, usize)> {
        self.lattice
            .distinct_orientations(&Self::rect_sites(n, m))
            .iter()
            .map(|sites| {
                let w = sites.iter().map(|s| s[0]).max().unwrap_or(0) + 1;
                let h = sites.iter().map(|s| s[1]).max().unwrap_or(0) + 1;
                (w as usize, h as usize)
            })
            .collect()
    }
}

impl<L: Lattice<Site = [i32; 2]>> ClusterGenerator for RectangleGenerator<L> {
    fn clusters(&self) -> Result<Vec<ClusterType>, NlceError> {
        let per_cell = self.lattice.unit_cell_sites().len();
        if per_cell == 0 {
            return Err(NlceError::InvalidInput(
                "lattice has an empty unit cell".into(),
            ));
        }
        // Candidate shapes (m <= n), realised as n wide and m high.
        let mut shapes: Vec<(usize, usize)> = Vec::new();
        for m in 1..=self.max_order {
            for n in m..=self.max_order {
                if self.order_of(m, n) > self.max_order {
                    continue;
                }
                if self.max_sites.is_some_and(|cap| m * n > cap) {
                    continue;
                }
                shapes.push((m, n));
            }
        }

        let mut out: Vec<ClusterType> = Vec::new();
        let mut included: Vec<(usize, usize)> = Vec::new();
        for &(m, n) in &shapes {
            let graph = self.lattice.cluster_graph(&Self::rect_sites(n, m))?;
            if !graph.is_connected() {
                continue;
            }
            let orientations = self.oriented_extents(m, n).len();
            included.push((m, n));
            out.push(ClusterType {
                key: ClusterKey::rectangle(m as u32, n as u32),
                order: self.order_of(m, n),
                graph,
                lattice_constant: orientations as f64 / per_cell as f64,
                subclusters: Vec::new(),
            });
        }

        // Multiplicities: embeddings of each included sub-shape in the fixed
        // (n wide, m high) realisation of each cluster.
        for (c, &(m, n)) in out.iter_mut().zip(&included) {
            for &(ms, ns) in &included {
                if (ms, ns) == (m, n) || ms * ns > m * n {
                    continue;
                }
                let count: u64 = self
                    .oriented_extents(ms, ns)
                    .into_iter()
                    .filter(|&(w, h)| w <= n && h <= m)
                    .map(|(w, h)| ((n - w + 1) * (m - h + 1)) as u64)
                    .sum();
                if count > 0 {
                    c.subclusters
                        .push((ClusterKey::rectangle(ms as u32, ns as u32), count));
                }
            }
        }

        out.sort_by(|a, b| {
            (a.order, a.graph.n_sites, &a.key).cmp(&(b.order, b.graph.n_sites, &b.key))
        });
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{ChainLattice, SquareLattice};

    fn find(cs: &[ClusterType], m: u32, n: u32) -> &ClusterType {
        cs.iter()
            .find(|c| c.key == ClusterKey::rectangle(m, n))
            .unwrap()
    }

    fn mult(c: &ClusterType, m: u32, n: u32) -> u64 {
        c.subclusters
            .iter()
            .find(|(k, _)| *k == ClusterKey::rectangle(m, n))
            .map_or(0, |(_, v)| *v)
    }

    #[test]
    fn square_rectangles_to_order_4() {
        let cs = RectangleGenerator::new(SquareLattice, 4)
            .clusters()
            .unwrap();
        let keys: Vec<String> = cs.iter().map(|c| c.key.to_string()).collect();
        assert_eq!(keys, ["1x1", "1x2", "1x3", "2x2"]);
        assert_eq!(find(&cs, 1, 1).lattice_constant, 1.0);
        assert_eq!(find(&cs, 1, 2).lattice_constant, 2.0);
        assert_eq!(find(&cs, 2, 2).lattice_constant, 1.0);
        let c22 = find(&cs, 2, 2);
        assert_eq!(mult(c22, 1, 1), 4);
        assert_eq!(mult(c22, 1, 2), 4); // two horizontal + two vertical bonds
        assert_eq!(mult(c22, 1, 3), 0);
    }

    #[test]
    fn square_multiplicities_in_2x3() {
        let cs = RectangleGenerator::new(SquareLattice, 5)
            .clusters()
            .unwrap();
        let c = find(&cs, 2, 3);
        assert_eq!(mult(c, 1, 1), 6);
        assert_eq!(mult(c, 1, 2), 7); // (3-2+1)*2 horizontal + 3*(2-2+1) vertical
        assert_eq!(mult(c, 1, 3), 2); // only along the long side
        assert_eq!(mult(c, 2, 2), 2);
    }

    #[test]
    fn chain_only_has_1xn() {
        let cs = RectangleGenerator::new(ChainLattice, 6).clusters().unwrap();
        let keys: Vec<String> = cs.iter().map(|c| c.key.to_string()).collect();
        assert_eq!(keys, ["1x1", "1x2", "1x3", "1x4", "1x5"]);
        assert!(cs.iter().all(|c| c.lattice_constant == 1.0));
        assert_eq!(mult(find(&cs, 1, 5), 1, 2), 4);
    }

    #[test]
    fn site_order_and_cap() {
        let cs = RectangleGenerator::new(SquareLattice, 6)
            .with_order(RectangleOrder::Sites)
            .clusters()
            .unwrap();
        assert!(cs.iter().all(|c| c.graph.n_sites <= 6));
        assert!(cs.iter().any(|c| c.key == ClusterKey::rectangle(2, 3)));
        let cs = RectangleGenerator::new(SquareLattice, 8)
            .with_max_sites(9)
            .clusters()
            .unwrap();
        assert!(cs.iter().all(|c| c.graph.n_sites <= 9));
        assert!(cs.iter().any(|c| c.key == ClusterKey::rectangle(3, 3)));
    }
}
