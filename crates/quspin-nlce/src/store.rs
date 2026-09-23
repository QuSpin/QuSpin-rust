//! Saving and loading cluster DAGs.
//!
//! The cluster DAG (topologies, lattice constants, sub-cluster
//! multiplicities, automorphisms) depends only on the lattice and the
//! generator, not on the model, and can be expensive to build (≈17 s for the
//! 12-bond square-lattice expansion). A [`ClusterSet`] stores it in a small
//! line-based text format and is itself a [`ClusterGenerator`], so a loaded
//! file plugs straight into [`run_nlce`](crate::combiner::run_nlce).
//!
//! # Format (`quspin-nlce clusters v1`)
//!
//! ```text
//! quspin-nlce clusters v1
//! description <free text, one line>
//! clusters <count>
//! <one line per cluster, in DAG order>
//! ```
//!
//! Each cluster line is a `;`-separated list of `name=value` fields:
//!
//! | field | value |
//! |---|---|
//! | `order` | expansion order |
//! | `sites` | number of sites |
//! | `L` | lattice constant (shortest round-trip `f64`) |
//! | `key` | `rect:m:n` or `canon:` + comma-separated hex code words |
//! | `key_bond_labels`, `key_site_labels` | optional, comma-separated |
//! | `bonds` | comma-separated `i-j` or `i-j:label` |
//! | `site_labels` | optional, comma-separated (omitted if all zero) |
//! | `autos` | automorphisms, `|`-separated, each a comma-separated permutation |
//! | `subs` | comma-separated `index*multiplicity`, `index` a 0-based line index of an earlier cluster |
//!
//! Loading rebuilds every [`ClusterGraph`] through its validating
//! constructors and checks that sub-clusters point backwards and orders are
//! non-decreasing.

use crate::error::NlceError;
use crate::generator::{ClusterGenerator, ClusterType};
use crate::graph::{Bond, ClusterGraph, ClusterKey, Topology};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

const MAGIC: &str = "quspin-nlce clusters v1";

/// A cluster DAG with a free-text description of where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterSet {
    /// Provenance, e.g. `"bond expansion, square lattice, 12 bonds"`.
    pub description: String,
    /// Clusters in DAG order (see [`ClusterGenerator::clusters`]).
    pub clusters: Vec<ClusterType>,
}

impl ClusterGenerator for ClusterSet {
    fn clusters(&self) -> Result<Vec<ClusterType>, NlceError> {
        Ok(self.clusters.clone())
    }
}

impl ClusterSet {
    /// Run `generator` once and keep its clusters.
    ///
    /// # Errors
    /// Whatever the generator returns.
    pub fn from_generator<G: ClusterGenerator>(
        generator: &G,
        description: impl Into<String>,
    ) -> Result<Self, NlceError> {
        Ok(Self {
            description: description.into(),
            clusters: generator.clusters()?,
        })
    }

    /// The clusters of order `<= max_order`. Valid because sub-clusters
    /// never have a higher order than their parent.
    pub fn truncated(&self, max_order: usize) -> Self {
        Self {
            description: format!("{} (truncated to order {max_order})", self.description),
            clusters: self
                .clusters
                .iter()
                .filter(|c| c.order <= max_order)
                .cloned()
                .collect(),
        }
    }

    /// Write to `path` (created or overwritten).
    ///
    /// # Errors
    /// `Io` on file errors; `InvalidInput` if the description spans lines or
    /// a sub-cluster is missing.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), NlceError> {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        self.write(&mut w)?;
        w.flush()?;
        Ok(())
    }

    /// Read from `path`.
    ///
    /// # Errors
    /// `Io` on file errors; `InvalidInput` (with the line number) on a
    /// malformed file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, NlceError> {
        Self::read(BufReader::new(std::fs::File::open(path)?))
    }

    /// Serialise to any writer.
    ///
    /// # Errors
    /// See [`save`](Self::save).
    pub fn write<W: Write>(&self, mut w: W) -> Result<(), NlceError> {
        if self.description.contains('\n') {
            return Err(NlceError::InvalidInput(
                "cluster-set description must be a single line".into(),
            ));
        }
        let index: HashMap<&ClusterKey, usize> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (&c.key, i))
            .collect();
        writeln!(w, "{MAGIC}")?;
        writeln!(w, "description {}", self.description)?;
        writeln!(w, "clusters {}", self.clusters.len())?;
        for c in &self.clusters {
            let mut fields = vec![
                format!("order={}", c.order),
                format!("sites={}", c.graph.n_sites),
                format!("L={:?}", c.lattice_constant),
                format!("key={}", encode_topology(&c.key.topology)),
            ];
            if !c.key.bond_labels.is_empty() {
                fields.push(format!("key_bond_labels={}", join(&c.key.bond_labels)));
            }
            if !c.key.site_labels.is_empty() {
                fields.push(format!("key_site_labels={}", join(&c.key.site_labels)));
            }
            let bonds: Vec<String> = c
                .graph
                .bonds
                .iter()
                .map(|b| {
                    if b.label == 0 {
                        format!("{}-{}", b.i, b.j)
                    } else {
                        format!("{}-{}:{}", b.i, b.j, b.label)
                    }
                })
                .collect();
            fields.push(format!("bonds={}", bonds.join(",")));
            if c.graph.site_labels.iter().any(|&l| l != 0) {
                fields.push(format!("site_labels={}", join(&c.graph.site_labels)));
            }
            let autos: Vec<String> = c.graph.automorphisms.iter().map(|p| join(p)).collect();
            fields.push(format!("autos={}", autos.join("|")));
            let mut subs = Vec::with_capacity(c.subclusters.len());
            for (k, m) in &c.subclusters {
                let i = index.get(k).ok_or_else(|| {
                    NlceError::InvalidInput(format!(
                        "sub-cluster {k} of {} is not in the set",
                        c.key
                    ))
                })?;
                subs.push(format!("{i}*{m}"));
            }
            fields.push(format!("subs={}", subs.join(",")));
            writeln!(w, "{}", fields.join(";"))?;
        }
        Ok(())
    }

    /// Parse from any buffered reader.
    ///
    /// # Errors
    /// See [`load`](Self::load).
    pub fn read<R: BufRead>(r: R) -> Result<Self, NlceError> {
        let mut lines = r.lines().enumerate();
        let mut next = |what: &str| -> Result<(usize, String), NlceError> {
            match lines.next() {
                Some((i, l)) => Ok((i + 1, l?)),
                None => Err(NlceError::InvalidInput(format!(
                    "unexpected end of file, expected {what}"
                ))),
            }
        };
        let (_, magic) = next("header")?;
        if magic.trim_end() != MAGIC {
            return Err(NlceError::InvalidInput(format!(
                "line 1: expected `{MAGIC}`, got `{magic}`"
            )));
        }
        let (ln, d) = next("description")?;
        let description = d
            .strip_prefix("description ")
            .or_else(|| d.strip_prefix("description"))
            .ok_or_else(|| bad(ln, "expected `description …`"))?
            .to_string();
        let (ln, c) = next("cluster count")?;
        let count: usize = c
            .strip_prefix("clusters ")
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| bad(ln, "expected `clusters <count>`"))?;

        let mut clusters: Vec<ClusterType> = Vec::with_capacity(count);
        for _ in 0..count {
            let (ln, line) = next("a cluster line")?;
            let c = parse_cluster(&line, &clusters).map_err(|e| bad(ln, &e))?;
            if clusters.last().is_some_and(|p| p.order > c.order) {
                return Err(bad(ln, "orders must be non-decreasing"));
            }
            clusters.push(c);
        }
        Ok(Self {
            description,
            clusters,
        })
    }
}

fn bad(line: usize, msg: &str) -> NlceError {
    NlceError::InvalidInput(format!("line {line}: {msg}"))
}

fn join<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn encode_topology(t: &Topology) -> String {
    match t {
        Topology::Rectangle { m, n } => format!("rect:{m}:{n}"),
        Topology::Canonical(code) => {
            let words: Vec<String> = code.iter().map(|w| format!("{w:x}")).collect();
            format!("canon:{}", words.join(","))
        }
    }
}

fn decode_topology(s: &str) -> Result<Topology, String> {
    if let Some(rest) = s.strip_prefix("rect:") {
        let (m, n) = rest.split_once(':').ok_or("rect key needs `rect:m:n`")?;
        Ok(Topology::Rectangle {
            m: m.parse()
                .map_err(|_| format!("bad rectangle extent `{m}`"))?,
            n: n.parse()
                .map_err(|_| format!("bad rectangle extent `{n}`"))?,
        })
    } else if let Some(rest) = s.strip_prefix("canon:") {
        let code = rest
            .split(',')
            .map(|w| u64::from_str_radix(w, 16).map_err(|_| format!("bad code word `{w}`")))
            .collect::<Result<_, _>>()?;
        Ok(Topology::Canonical(code))
    } else {
        Err(format!("unknown key `{s}`"))
    }
}

fn parse_list<T: std::str::FromStr>(s: &str, what: &str) -> Result<Vec<T>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|x| x.parse().map_err(|_| format!("bad {what} `{x}`")))
        .collect()
}

fn parse_cluster(line: &str, earlier: &[ClusterType]) -> Result<ClusterType, String> {
    let mut f: HashMap<&str, &str> = HashMap::new();
    for field in line.split(';') {
        let (k, v) = field
            .split_once('=')
            .ok_or_else(|| format!("field `{field}` has no `=`"))?;
        if f.insert(k, v).is_some() {
            return Err(format!("duplicate field `{k}`"));
        }
    }
    let get = |k: &str| {
        f.get(k)
            .copied()
            .ok_or_else(|| format!("missing field `{k}`"))
    };
    let order: usize = get("order")?.parse().map_err(|_| "bad `order`")?;
    let sites: usize = get("sites")?.parse().map_err(|_| "bad `sites`")?;
    let lattice_constant: f64 = get("L")?.parse().map_err(|_| "bad `L`")?;
    let key = ClusterKey {
        topology: decode_topology(get("key")?)?,
        bond_labels: parse_list(
            f.get("key_bond_labels").copied().unwrap_or(""),
            "key bond label",
        )?,
        site_labels: parse_list(
            f.get("key_site_labels").copied().unwrap_or(""),
            "key site label",
        )?,
    };
    let mut bonds = Vec::new();
    let bond_field = get("bonds")?;
    if !bond_field.is_empty() {
        for b in bond_field.split(',') {
            let (ij, label) = match b.split_once(':') {
                Some((ij, l)) => (
                    ij,
                    l.parse().map_err(|_| format!("bad bond label in `{b}`"))?,
                ),
                None => (b, 0),
            };
            let (i, j) = ij
                .split_once('-')
                .ok_or_else(|| format!("bad bond `{b}`"))?;
            bonds.push(Bond {
                i: i.parse().map_err(|_| format!("bad bond `{b}`"))?,
                j: j.parse().map_err(|_| format!("bad bond `{b}`"))?,
                label,
            });
        }
    }
    let mut graph = ClusterGraph::new(sites, bonds).map_err(|e| e.to_string())?;
    if let Some(sl) = f.get("site_labels") {
        let labels: Vec<u32> = parse_list(sl, "site label")?;
        if labels.len() != sites {
            return Err(format!("{} site labels for {sites} sites", labels.len()));
        }
        graph.site_labels = labels;
    }
    let autos_field = get("autos")?;
    let autos: Vec<Vec<usize>> = if autos_field.is_empty() {
        Vec::new()
    } else {
        autos_field
            .split('|')
            .map(|p| parse_list(p, "automorphism entry"))
            .collect::<Result<_, _>>()?
    };
    let graph = graph.with_automorphisms(autos).map_err(|e| e.to_string())?;
    let mut subclusters = Vec::new();
    let subs_field = get("subs")?;
    if !subs_field.is_empty() {
        for s in subs_field.split(',') {
            let (i, m) = s
                .split_once('*')
                .ok_or_else(|| format!("bad sub-cluster `{s}`"))?;
            let i: usize = i.parse().map_err(|_| format!("bad sub-cluster `{s}`"))?;
            let m: u64 = m.parse().map_err(|_| format!("bad sub-cluster `{s}`"))?;
            let sub = earlier.get(i).ok_or_else(|| {
                format!("sub-cluster index {i} does not refer to an earlier cluster")
            })?;
            subclusters.push((sub.key.clone(), m));
        }
    }
    Ok(ClusterType {
        key,
        order,
        graph,
        lattice_constant,
        subclusters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{BondGenerator, RectangleGenerator};
    use crate::lattice::SquareLattice;

    fn round_trip(set: &ClusterSet) -> ClusterSet {
        let mut buf = Vec::new();
        set.write(&mut buf).unwrap();
        ClusterSet::read(buf.as_slice()).unwrap()
    }

    #[test]
    fn round_trips_both_generators() {
        let bond =
            ClusterSet::from_generator(&BondGenerator::new(SquareLattice, 7), "bond 7").unwrap();
        assert_eq!(round_trip(&bond), bond);
        let rect = ClusterSet::from_generator(&RectangleGenerator::new(SquareLattice, 7), "rect 7")
            .unwrap();
        assert_eq!(round_trip(&rect), rect);
    }

    #[test]
    fn truncation_matches_lower_order_generation() {
        let big = ClusterSet::from_generator(&BondGenerator::new(SquareLattice, 7), "").unwrap();
        let small = BondGenerator::new(SquareLattice, 5).clusters().unwrap();
        assert_eq!(big.truncated(5).clusters, small);
    }

    #[test]
    fn rejects_malformed_files() {
        let good = {
            let set =
                ClusterSet::from_generator(&BondGenerator::new(SquareLattice, 2), "x").unwrap();
            let mut buf = Vec::new();
            set.write(&mut buf).unwrap();
            String::from_utf8(buf).unwrap()
        };
        assert!(ClusterSet::read(good.as_bytes()).is_ok());
        for broken in [
            good.replacen("v1", "v2", 1),
            good.replacen("clusters 3", "clusters 4", 1),
            good.replacen("subs=0*2", "subs=2*2", 1),
            good.replacen("bonds=0-1", "bonds=0-5", 1),
            good.replacen("L=2.0", "L=two", 1),
        ] {
            assert_ne!(broken, good, "test edit did not apply");
            assert!(ClusterSet::read(broken.as_bytes()).is_err(), "{broken}");
        }
    }
}
