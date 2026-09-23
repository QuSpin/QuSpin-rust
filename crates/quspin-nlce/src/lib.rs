//! Numerical linked-cluster expansion (NLCE) for finite-temperature
//! thermodynamics of spin-1/2 lattice models, on top of QuSpin-rust.
//!
//! An extensive property per site in the thermodynamic limit is expanded as
//!
//! ```text
//! P / N = Σ_c L(c) W(c),     W(c) = P(c) − Σ_{s ⊂ c} M(s, c) W(s),
//! ```
//!
//! where `L(c)` is the number of embeddings of cluster `c` per lattice site
//! and `M(s, c)` the number of embeddings of sub-cluster `s` in `c`
//! (Tang, Khatami & Rigol, arXiv:1207.3366). Implemented: the rectangle
//! expansion (Gan & Hazzard, arXiv:2005.03177) and topological expansions
//! over a pluggable node kind (bonds, sites, …) on any lattice, with
//! clusters merged by labelled graph isomorphism.
//!
//! The pipeline is split along traits so each stage can be swapped:
//!
//! | Stage | Trait | Phase 1 impl |
//! |---|---|---|
//! | Infinite lattice | [`Lattice`] | [`ChainLattice`], [`SquareLattice`], [`SquareJ1J2Lattice`], [`TriangularLattice`], [`HoneycombLattice`] |
//! | Cluster DAG | [`ClusterGenerator`] | [`RectangleGenerator`], [`TopologicalGenerator`] over a [`NodeKind`] ([`Bonds`], [`Sites`]) |
//! | Hamiltonian | [`Model`] | [`Xxz`], [`LabeledXxz`] |
//! | Per-cluster property | [`ClusterSolver`] | [`ExactDiagSolver`] → [`Thermo`] |
//! | Inclusion–exclusion | [`combine`] / [`run_nlce`] over any [`Property`] | — |
//! | Resummation | [`Resummation`] | [`Bare`], [`Wynn`], [`Euler`] |
//! | Saved DAGs | [`ClusterSet`] (a [`ClusterGenerator`]) | text file, [`store`] |
//!
//! ```no_run
//! use quspin_nlce::*;
//!
//! let temps: Vec<f64> = (1..=20).map(|i| 0.25 * i as f64).collect();
//! let solver = ExactDiagSolver::new(temps).unwrap();
//! let generator = RectangleGenerator::new(SquareLattice, 6);
//! let result = run_nlce(&generator, &solver, &Xxz::heisenberg(1.0)).unwrap();
//! for (order, sum) in &result.partial_sums {
//!     println!("order {order}: E/N(T=5) = {}", sum.energy[19]);
//! }
//! ```

#![warn(missing_docs)]

pub mod canon;
pub mod combiner;
pub mod error;
pub mod generator;
pub mod graph;
pub mod lattice;
pub mod model;
pub mod property;
pub mod resum;
pub mod solver;
pub mod store;

pub use combiner::{ClusterCache, NlceResult, combine, run_nlce, run_nlce_cached};
pub use error::NlceError;
pub use generator::{
    Adjacency, BondGenerator, Bonds, ClusterGenerator, ClusterType, EmbeddingCensus, Node,
    NodeKind, RectangleGenerator, RectangleOrder, SiteGenerator, Sites, TopologicalGenerator,
};
pub use graph::{Bond, ClusterGraph, ClusterKey, Topology};
pub use lattice::{
    ChainLattice, HoneycombLattice, Lattice, SquareJ1J2Lattice, SquareLattice, TriangularLattice,
};
pub use model::{LabeledXxz, Model, Xxz};
pub use property::{Componentwise, Property, Thermo};
pub use resum::{Bare, Euler, Resummation, Wynn};
pub use solver::{ClusterSolver, ExactDiagSolver, MAX_ED_SITES, Spectrum, SpectrumBlock};
pub use store::ClusterSet;
