//! Concrete finite clusters ([`ClusterGraph`]) and their cache keys
//! ([`ClusterKey`]).
//!
//! A [`ClusterGraph`] is what a solver sees: a finite set of sites
//! `0..n_sites` with labelled bonds and open boundaries. It carries no
//! information about how it was generated, so the rectangle and bond
//! generators (and any future one) produce the same type.

use crate::error::NlceError;
use std::collections::HashSet;
use std::fmt;

/// An undirected bond between sites `i` and `j` of a cluster.
///
/// `label` distinguishes bond classes (e.g. `J1` vs `J2`, or disorder
/// realisations). Phase 1 models ignore it, but it is part of the graph
/// identity checked by automorphisms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bond {
    /// First site.
    pub i: u32,
    /// Second site.
    pub j: u32,
    /// Bond class label.
    pub label: u32,
}

impl Bond {
    /// Bond with `label = 0`.
    pub fn new(i: u32, j: u32) -> Self {
        Self { i, j, label: 0 }
    }

    /// Orientation-independent form (`i <= j`), used for set comparisons.
    fn normalized(self) -> Self {
        if self.i <= self.j {
            self
        } else {
            Self {
                i: self.j,
                j: self.i,
                label: self.label,
            }
        }
    }
}

/// A finite, open-boundary cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterGraph {
    /// Number of sites; sites are `0..n_sites`.
    pub n_sites: usize,
    /// Bonds (each unordered pair at most once).
    pub bonds: Vec<Bond>,
    /// Per-site labels (all `0` in Phase 1).
    pub site_labels: Vec<u32>,
    /// Non-identity graph automorphisms as site permutations
    /// (`perm[i]` = image of site `i`). Together with the implicit identity
    /// they form a group. May be empty: solvers then simply do not use
    /// lattice symmetries, which is correct but slower.
    pub automorphisms: Vec<Vec<usize>>,
}

impl ClusterGraph {
    /// Build a cluster without automorphisms.
    ///
    /// # Errors
    /// `InvalidInput` if a bond references a site `>= n_sites`, is a
    /// self-loop, or appears twice.
    pub fn new(n_sites: usize, bonds: Vec<Bond>) -> Result<Self, NlceError> {
        let mut seen = HashSet::new();
        for b in &bonds {
            if b.i as usize >= n_sites || b.j as usize >= n_sites {
                return Err(NlceError::InvalidInput(format!(
                    "bond ({}, {}) out of range for {n_sites} sites",
                    b.i, b.j
                )));
            }
            if b.i == b.j {
                return Err(NlceError::InvalidInput(format!(
                    "self-loop on site {}",
                    b.i
                )));
            }
            let n = b.normalized();
            if !seen.insert((n.i, n.j)) {
                return Err(NlceError::InvalidInput(format!(
                    "duplicate bond ({}, {})",
                    n.i, n.j
                )));
            }
        }
        Ok(Self {
            n_sites,
            bonds,
            site_labels: vec![0; n_sites],
            automorphisms: Vec::new(),
        })
    }

    /// Attach automorphisms, validating that each is a bond- and
    /// label-preserving permutation, that none is the identity or repeated,
    /// and that together with the identity they are closed under composition.
    ///
    /// # Errors
    /// `InvalidInput` describing the first violated condition.
    pub fn with_automorphisms(mut self, automorphisms: Vec<Vec<usize>>) -> Result<Self, NlceError> {
        let n = self.n_sites;
        let bond_set: HashSet<Bond> = self.bonds.iter().map(|b| b.normalized()).collect();
        let identity: Vec<usize> = (0..n).collect();
        let mut elements: HashSet<Vec<usize>> = HashSet::new();
        for p in &automorphisms {
            if p.len() != n {
                return Err(NlceError::InvalidInput(format!(
                    "automorphism has length {} but cluster has {n} sites",
                    p.len()
                )));
            }
            let mut hit = vec![false; n];
            for &v in p {
                if v >= n || std::mem::replace(&mut hit[v], true) {
                    return Err(NlceError::InvalidInput(format!(
                        "automorphism {p:?} is not a permutation"
                    )));
                }
            }
            if *p == identity {
                return Err(NlceError::InvalidInput(
                    "the identity automorphism is implicit and must not be listed".into(),
                ));
            }
            for b in &self.bonds {
                let image = Bond {
                    i: p[b.i as usize] as u32,
                    j: p[b.j as usize] as u32,
                    label: b.label,
                };
                if !bond_set.contains(&image.normalized()) {
                    return Err(NlceError::InvalidInput(format!(
                        "automorphism {p:?} does not map bond ({}, {}) to a bond",
                        b.i, b.j
                    )));
                }
            }
            if (0..n).any(|i| self.site_labels[p[i]] != self.site_labels[i]) {
                return Err(NlceError::InvalidInput(format!(
                    "automorphism {p:?} does not preserve site labels"
                )));
            }
            if !elements.insert(p.clone()) {
                return Err(NlceError::InvalidInput(format!(
                    "automorphism {p:?} listed twice"
                )));
            }
        }
        for a in &automorphisms {
            for b in &automorphisms {
                let ab = compose(a, b);
                if ab != identity && !elements.contains(&ab) {
                    return Err(NlceError::InvalidInput(format!(
                        "automorphisms are not closed: {a:?} ∘ {b:?} = {ab:?} is missing"
                    )));
                }
            }
        }
        self.automorphisms = automorphisms;
        Ok(self)
    }

    /// Open-boundary `width × height` rectangle of the square lattice.
    ///
    /// Site `(x, y)` has index `x + width * y`. Automorphisms are the
    /// non-identity elements among the two axis reflections and their
    /// product (the rectangle's C2v subgroup), which suffices for
    /// block-diagonalisation with one-dimensional characters. For
    /// `width == height` the diagonal reflections are deliberately omitted.
    ///
    /// # Errors
    /// `InvalidInput` if either extent is zero.
    pub fn rectangle(width: usize, height: usize) -> Result<Self, NlceError> {
        if width == 0 || height == 0 {
            return Err(NlceError::InvalidInput(format!(
                "rectangle extents must be positive, got {width} × {height}"
            )));
        }
        let idx = |x: usize, y: usize| (x + width * y) as u32;
        let mut bonds = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if x + 1 < width {
                    bonds.push(Bond::new(idx(x, y), idx(x + 1, y)));
                }
                if y + 1 < height {
                    bonds.push(Bond::new(idx(x, y), idx(x, y + 1)));
                }
            }
        }
        let n = width * height;
        let identity: Vec<usize> = (0..n).collect();
        let reflect = |fx: bool, fy: bool| -> Vec<usize> {
            let mut p = vec![0; n];
            for y in 0..height {
                for x in 0..width {
                    let xi = if fx { width - 1 - x } else { x };
                    let yi = if fy { height - 1 - y } else { y };
                    p[idx(x, y) as usize] = idx(xi, yi) as usize;
                }
            }
            p
        };
        let mut autos: Vec<Vec<usize>> = Vec::new();
        for p in [
            reflect(true, false),
            reflect(false, true),
            reflect(true, true),
        ] {
            if p != identity && !autos.contains(&p) {
                autos.push(p);
            }
        }
        Self::new(n, bonds)?.with_automorphisms(autos)
    }
}

impl ClusterGraph {
    /// `true` if every site is reachable from site 0 through bonds (an
    /// empty cluster counts as connected).
    pub fn is_connected(&self) -> bool {
        if self.n_sites == 0 {
            return true;
        }
        let mut adj = vec![Vec::new(); self.n_sites];
        for b in &self.bonds {
            adj[b.i as usize].push(b.j as usize);
            adj[b.j as usize].push(b.i as usize);
        }
        let mut seen = vec![false; self.n_sites];
        let mut stack = vec![0];
        seen[0] = true;
        while let Some(v) = stack.pop() {
            for &w in &adj[v] {
                if !std::mem::replace(&mut seen[w], true) {
                    stack.push(w);
                }
            }
        }
        seen.into_iter().all(|x| x)
    }
}

/// `(a ∘ b)[i] = a[b[i]]`.
pub(crate) fn compose(a: &[usize], b: &[usize]) -> Vec<usize> {
    b.iter().map(|&i| a[i]).collect()
}

/// Shape identity of a cluster, independent of labels.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Topology {
    /// `m × n` rectangle with `m <= n` (orientation is quotiented out).
    Rectangle {
        /// Shorter extent.
        m: u32,
        /// Longer extent.
        n: u32,
    },
    /// Canonical code of an arbitrary graph ([`crate::canon::Canonical::code`]),
    /// used by the topological bond expansion.
    Canonical(Vec<u64>),
}

/// Cache key for a cluster type: its topology plus optional bond / site
/// labels in canonical site order.
///
/// Phase 1 only uses the topology (labels empty). Disorder or inhomogeneous
/// states fill in the labels, so that clusters with equal shape but
/// different couplings are cached separately.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClusterKey {
    /// Shape of the cluster.
    pub topology: Topology,
    /// Bond labels in canonical bond order (empty if unlabelled).
    pub bond_labels: Vec<u32>,
    /// Site labels in canonical site order (empty if unlabelled).
    pub site_labels: Vec<u32>,
}

impl ClusterKey {
    /// Unlabelled rectangle key; the extents are sorted so that `m × n` and
    /// `n × m` give the same key.
    pub fn rectangle(a: u32, b: u32) -> Self {
        Self {
            topology: Topology::Rectangle {
                m: a.min(b),
                n: a.max(b),
            },
            bond_labels: Vec::new(),
            site_labels: Vec::new(),
        }
    }
}

impl fmt::Display for ClusterKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.topology {
            Topology::Rectangle { m, n } => write!(f, "{m}x{n}")?,
            Topology::Canonical(code) => {
                // `[n, row_0, …]`: vertex and bond counts, then the rows.
                let rows = &code[1..];
                let bonds: u32 = rows.iter().map(|r| r.count_ones()).sum::<u32>() / 2;
                let hex: Vec<String> = rows.iter().map(|r| format!("{r:x}")).collect();
                write!(f, "{}v{bonds}b[{}]", code[0], hex.join("."))?
            }
        }
        if !self.bond_labels.is_empty() || !self.site_labels.is_empty() {
            write!(f, "[b{:?} s{:?}]", self.bond_labels, self.site_labels)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_counts() {
        let g = ClusterGraph::rectangle(3, 4).unwrap();
        assert_eq!(g.n_sites, 12);
        assert_eq!(g.bonds.len(), 2 * 4 + 3 * 3);
        assert_eq!(g.automorphisms.len(), 3);
        let chain = ClusterGraph::rectangle(5, 1).unwrap();
        assert_eq!(chain.bonds.len(), 4);
        assert_eq!(chain.automorphisms.len(), 1);
        assert!(
            ClusterGraph::rectangle(1, 1)
                .unwrap()
                .automorphisms
                .is_empty()
        );
    }

    #[test]
    fn rejects_bad_automorphisms() {
        let g = ClusterGraph::new(3, vec![Bond::new(0, 1), Bond::new(1, 2)]).unwrap();
        // Swapping 0 and 1 does not preserve bond (1, 2).
        assert!(g.clone().with_automorphisms(vec![vec![1, 0, 2]]).is_err());
        assert!(g.clone().with_automorphisms(vec![vec![0, 1, 2]]).is_err());
        assert!(g.with_automorphisms(vec![vec![2, 1, 0]]).is_ok());
        // Not closed: 4-cycle rotation without its square.
        let ring = ClusterGraph::new(
            4,
            vec![
                Bond::new(0, 1),
                Bond::new(1, 2),
                Bond::new(2, 3),
                Bond::new(3, 0),
            ],
        )
        .unwrap();
        assert!(ring.with_automorphisms(vec![vec![1, 2, 3, 0]]).is_err());
    }

    #[test]
    fn keys_quotient_orientation() {
        assert_eq!(ClusterKey::rectangle(2, 3), ClusterKey::rectangle(3, 2));
        assert_eq!(ClusterKey::rectangle(3, 2).to_string(), "2x3");
    }
}
