//! Infinite lattices: sites, bonds, and point-group symmetry.
//!
//! A [`Lattice`] is the thermodynamic-limit system the expansion targets.
//! Generators use it to realise finite clusters as [`ClusterGraph`]s, to count
//! distinct orientations (the lattice constant `L(c)`), and to find cluster
//! automorphisms. The bond generator grows clusters with
//! [`Lattice::neighbors`] from [`Lattice::unit_cell_sites`].

use crate::error::NlceError;
use crate::graph::{Bond, ClusterGraph};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

/// An infinite lattice with a finite point group.
pub trait Lattice: Sync {
    /// Lattice-site coordinate.
    type Site: Copy + Eq + Ord + Hash + Debug + Send + Sync;

    /// The sites of one unit cell. Lattice constants are reported per site,
    /// i.e. normalised by `unit_cell_sites().len()`.
    fn unit_cell_sites(&self) -> Vec<Self::Site>;

    /// Neighbours of `site`, each with a bond label (label 0 unless the
    /// lattice has several bond types, e.g. J1 = 0 and J2 = 1).
    fn neighbors(&self, site: Self::Site) -> Vec<(Self::Site, u32)>;

    /// Label of `site` (e.g. its sublattice when sublattices are physically
    /// distinct). Default 0.
    fn site_label(&self, _site: Self::Site) -> u32 {
        0
    }

    /// Point-group operations as site maps (identity included). Each must be
    /// a lattice automorphism that preserves bond labels.
    fn point_group(&self) -> Vec<fn(Self::Site) -> Self::Site>;

    /// Translate a finite site set by a lattice vector into a canonical
    /// position, preserving element order (output `k` is the translate of
    /// input `k`). Translated copies of a cluster must map to the same set.
    fn translate(&self, sites: &[Self::Site]) -> Vec<Self::Site>;

    /// Canonical form of a site set: [`translate`](Self::translate)d and
    /// sorted, so translated copies of a cluster compare equal.
    fn canonical_form(&self, sites: &[Self::Site]) -> Vec<Self::Site> {
        let mut v = self.translate(sites);
        v.sort();
        v
    }

    /// Distinct images of the site set under the point group, each in
    /// canonical position. `len()` of the result is the number of
    /// orientations in which the cluster embeds with a fixed anchor.
    fn distinct_orientations(&self, sites: &[Self::Site]) -> Vec<Vec<Self::Site>> {
        let images: BTreeSet<Vec<Self::Site>> = self
            .point_group()
            .into_iter()
            .map(|g| {
                let img: Vec<Self::Site> = sites.iter().map(|&s| g(s)).collect();
                self.canonical_form(&img)
            })
            .collect();
        images.into_iter().collect()
    }

    /// Induced open-boundary cluster on `sites`. Site indices follow the
    /// [`canonical_form`](Self::canonical_form) order of `sites`.
    /// Automorphisms are the site maps induced by the point-group operations
    /// that fix the site set up to translation (the stabiliser, which is a
    /// group, so no closure step is needed).
    ///
    /// # Errors
    /// `InvalidInput` if `sites` contains duplicates.
    fn cluster_graph(&self, sites: &[Self::Site]) -> Result<ClusterGraph, NlceError> {
        let canon = self.canonical_form(sites);
        if canon.windows(2).any(|w| w[0] == w[1]) {
            return Err(NlceError::InvalidInput("duplicate cluster sites".into()));
        }
        let index: HashMap<Self::Site, usize> =
            canon.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let mut bonds = Vec::new();
        for (i, &s) in canon.iter().enumerate() {
            for (t, label) in self.neighbors(s) {
                if let Some(&j) = index.get(&t)
                    && i < j
                {
                    bonds.push(Bond {
                        i: i as u32,
                        j: j as u32,
                        label,
                    });
                }
            }
        }
        let identity: Vec<usize> = (0..canon.len()).collect();
        let mut autos: Vec<Vec<usize>> = Vec::new();
        for g in self.point_group() {
            let img: Vec<Self::Site> = canon.iter().map(|&s| g(s)).collect();
            let img_canon = self.canonical_form(&img);
            if img_canon != canon {
                continue;
            }
            let shifted = self.translate(&img);
            let perm: Vec<usize> = shifted.iter().map(|s| index[s]).collect();
            if perm != identity && !autos.contains(&perm) {
                autos.push(perm);
            }
        }
        let mut graph = ClusterGraph::new(canon.len(), bonds)?;
        graph.site_labels = canon.iter().map(|&s| self.site_label(s)).collect();
        graph.with_automorphisms(autos)
    }
}

/// Translate 2D coordinates so the minimum x and y are zero.
fn shift_2d(sites: &[[i32; 2]]) -> Vec<[i32; 2]> {
    let mx = sites.iter().map(|s| s[0]).min().unwrap_or(0);
    let my = sites.iter().map(|s| s[1]).min().unwrap_or(0);
    sites.iter().map(|s| [s[0] - mx, s[1] - my]).collect()
}

/// The square lattice with nearest-neighbour bonds (label 0) and point group
/// D4 (8 elements). Sites are `[x, y]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SquareLattice;

impl Lattice for SquareLattice {
    type Site = [i32; 2];

    fn unit_cell_sites(&self) -> Vec<[i32; 2]> {
        vec![[0, 0]]
    }

    fn neighbors(&self, s: [i32; 2]) -> Vec<([i32; 2], u32)> {
        let [x, y] = s;
        vec![
            ([x + 1, y], 0),
            ([x - 1, y], 0),
            ([x, y + 1], 0),
            ([x, y - 1], 0),
        ]
    }

    fn point_group(&self) -> Vec<fn([i32; 2]) -> [i32; 2]> {
        vec![
            |[x, y]| [x, y],
            |[x, y]| [-y, x],
            |[x, y]| [-x, -y],
            |[x, y]| [y, -x],
            |[x, y]| [-x, y],
            |[x, y]| [x, -y],
            |[x, y]| [y, x],
            |[x, y]| [-y, -x],
        ]
    }

    fn translate(&self, sites: &[[i32; 2]]) -> Vec<[i32; 2]> {
        shift_2d(sites)
    }
}

/// The infinite 1D chain with nearest-neighbour bonds (label 0) and point
/// group `{1, x → −x}`. Sites are `[x, 0]` so rectangle generators can treat
/// it as a 2D lattice with no bonds along y.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChainLattice;

impl Lattice for ChainLattice {
    type Site = [i32; 2];

    fn unit_cell_sites(&self) -> Vec<[i32; 2]> {
        vec![[0, 0]]
    }

    fn neighbors(&self, s: [i32; 2]) -> Vec<([i32; 2], u32)> {
        let [x, y] = s;
        vec![([x + 1, y], 0), ([x - 1, y], 0)]
    }

    fn point_group(&self) -> Vec<fn([i32; 2]) -> [i32; 2]> {
        vec![|[x, y]| [x, y], |[x, y]| [-x, y]]
    }

    fn translate(&self, sites: &[[i32; 2]]) -> Vec<[i32; 2]> {
        shift_2d(sites)
    }
}

/// Square lattice with nearest-neighbour bonds (label 0) and
/// next-nearest-neighbour diagonal bonds (label 1), for J1–J2 models.
/// Point group D4, sites `[x, y]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SquareJ1J2Lattice;

impl Lattice for SquareJ1J2Lattice {
    type Site = [i32; 2];

    fn unit_cell_sites(&self) -> Vec<[i32; 2]> {
        vec![[0, 0]]
    }

    fn neighbors(&self, s: [i32; 2]) -> Vec<([i32; 2], u32)> {
        let mut out = SquareLattice.neighbors(s);
        let [x, y] = s;
        for (dx, dy) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
            out.push(([x + dx, y + dy], 1));
        }
        out
    }

    fn point_group(&self) -> Vec<fn([i32; 2]) -> [i32; 2]> {
        SquareLattice.point_group()
    }

    fn translate(&self, sites: &[[i32; 2]]) -> Vec<[i32; 2]> {
        shift_2d(sites)
    }
}

/// Triangular lattice in lattice coordinates `[x, y]` (primitive vectors at
/// 60°), six nearest neighbours (label 0), point group C6v.
#[derive(Clone, Copy, Debug, Default)]
pub struct TriangularLattice;

/// 60° rotation of triangular-lattice coordinates.
fn tri_rot([x, y]: [i32; 2]) -> [i32; 2] {
    [-y, x + y]
}

/// Mirror exchanging the two primitive vectors.
fn tri_mirror([x, y]: [i32; 2]) -> [i32; 2] {
    [y, x]
}

impl Lattice for TriangularLattice {
    type Site = [i32; 2];

    fn unit_cell_sites(&self) -> Vec<[i32; 2]> {
        vec![[0, 0]]
    }

    fn neighbors(&self, [x, y]: [i32; 2]) -> Vec<([i32; 2], u32)> {
        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]]
            .iter()
            .map(|d| ([x + d[0], y + d[1]], 0))
            .collect()
    }

    fn point_group(&self) -> Vec<fn([i32; 2]) -> [i32; 2]> {
        vec![
            |s| s,
            tri_rot,
            |s| tri_rot(tri_rot(s)),
            |s| tri_rot(tri_rot(tri_rot(s))),
            |s| tri_rot(tri_rot(tri_rot(tri_rot(s)))),
            |s| tri_rot(tri_rot(tri_rot(tri_rot(tri_rot(s))))),
            tri_mirror,
            |s| tri_mirror(tri_rot(s)),
            |s| tri_mirror(tri_rot(tri_rot(s))),
            |s| tri_mirror(tri_rot(tri_rot(tri_rot(s)))),
            |s| tri_mirror(tri_rot(tri_rot(tri_rot(tri_rot(s))))),
            |s| tri_mirror(tri_rot(tri_rot(tri_rot(tri_rot(tri_rot(s)))))),
        ]
    }

    fn translate(&self, sites: &[[i32; 2]]) -> Vec<[i32; 2]> {
        shift_2d(sites)
    }
}

/// Honeycomb lattice: sites `[x, y, s]` with cell `(x, y)` of the underlying
/// triangular Bravais lattice and sublattice `s ∈ {0, 1}`; `A(x, y)` bonds to
/// `B(x, y)`, `B(x − 1, y)`, `B(x, y − 1)` (label 0). Point group C6v (up to
/// translation). Site labels are 0 on both sublattices.
#[derive(Clone, Copy, Debug, Default)]
pub struct HoneycombLattice;

/// 120° rotation about an A site.
fn honey_rot([x, y, s]: [i32; 3]) -> [i32; 3] {
    [-x - y - s, x, s]
}

/// Mirror exchanging the primitive vectors (fixes the A–B(x, y) bond).
fn honey_mirror([x, y, s]: [i32; 3]) -> [i32; 3] {
    [y, x, s]
}

/// Inversion through the centre of the A(0,0)–B(0,0) bond.
fn honey_inv([x, y, s]: [i32; 3]) -> [i32; 3] {
    [-x, -y, 1 - s]
}

impl Lattice for HoneycombLattice {
    type Site = [i32; 3];

    fn unit_cell_sites(&self) -> Vec<[i32; 3]> {
        vec![[0, 0, 0], [0, 0, 1]]
    }

    fn neighbors(&self, [x, y, s]: [i32; 3]) -> Vec<([i32; 3], u32)> {
        if s == 0 {
            vec![([x, y, 1], 0), ([x - 1, y, 1], 0), ([x, y - 1, 1], 0)]
        } else {
            vec![([x, y, 0], 0), ([x + 1, y, 0], 0), ([x, y + 1, 0], 0)]
        }
    }

    fn point_group(&self) -> Vec<fn([i32; 3]) -> [i32; 3]> {
        vec![
            |s| s,
            honey_rot,
            |s| honey_rot(honey_rot(s)),
            honey_mirror,
            |s| honey_mirror(honey_rot(s)),
            |s| honey_mirror(honey_rot(honey_rot(s))),
            honey_inv,
            |s| honey_inv(honey_rot(s)),
            |s| honey_inv(honey_rot(honey_rot(s))),
            |s| honey_inv(honey_mirror(s)),
            |s| honey_inv(honey_mirror(honey_rot(s))),
            |s| honey_inv(honey_mirror(honey_rot(honey_rot(s)))),
        ]
    }

    fn translate(&self, sites: &[[i32; 3]]) -> Vec<[i32; 3]> {
        let mx = sites.iter().map(|s| s[0]).min().unwrap_or(0);
        let my = sites.iter().map(|s| s[1]).min().unwrap_or(0);
        sites.iter().map(|s| [s[0] - mx, s[1] - my, s[2]]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: i32, h: i32) -> Vec<[i32; 2]> {
        (0..w).flat_map(|x| (0..h).map(move |y| [x, y])).collect()
    }

    #[test]
    fn square_orientations() {
        assert_eq!(SquareLattice.distinct_orientations(&rect(2, 3)).len(), 2);
        assert_eq!(SquareLattice.distinct_orientations(&rect(3, 3)).len(), 1);
        assert_eq!(ChainLattice.distinct_orientations(&rect(4, 1)).len(), 1);
    }

    #[test]
    fn square_cluster_graph_has_d4_or_c2v() {
        let g = SquareLattice.cluster_graph(&rect(3, 3)).unwrap();
        assert_eq!(g.bonds.len(), 12);
        assert_eq!(g.automorphisms.len(), 7);
        let g = SquareLattice.cluster_graph(&rect(2, 3)).unwrap();
        assert_eq!(g.automorphisms.len(), 3);
        let g = ChainLattice.cluster_graph(&rect(5, 1)).unwrap();
        assert_eq!(g.bonds.len(), 4);
        assert_eq!(g.automorphisms.len(), 1);
    }

    /// Every point-group operation maps bonds to bonds with the same label,
    /// and the operations are distinct up to translation.
    fn check_point_group<L: Lattice>(lattice: &L, sites: &[L::Site], order: usize) {
        let ops = lattice.point_group();
        assert_eq!(ops.len(), order);
        for g in &ops {
            for &s in sites {
                for (t, l) in lattice.neighbors(s) {
                    let (gs, gt) = (g(s), g(t));
                    assert!(
                        lattice.neighbors(gs).contains(&(gt, l)),
                        "{s:?}–{t:?} maps to a non-bond"
                    );
                }
            }
        }
        // An asymmetric site set has `order` distinct images.
        assert_eq!(lattice.distinct_orientations(sites).len(), order);
    }

    #[test]
    fn point_groups_are_lattice_automorphisms() {
        check_point_group(&SquareLattice, &[[0, 0], [1, 0], [2, 0], [0, 1]], 8);
        check_point_group(&SquareJ1J2Lattice, &[[0, 0], [1, 0], [2, 0], [0, 1]], 8);
        check_point_group(&TriangularLattice, &[[0, 0], [1, 0], [2, 0], [0, 1]], 12);
        check_point_group(
            &HoneycombLattice,
            &[[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0]],
            12,
        );
    }

    #[test]
    fn chain_ignores_vertical_neighbours() {
        let g = ChainLattice.cluster_graph(&rect(2, 2)).unwrap();
        assert_eq!(g.bonds.len(), 2);
    }
}
