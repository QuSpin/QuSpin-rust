//! The symmetry-blocked ED spectrum must equal brute-force full-space ED.

use num_complex::Complex;
use quspin_core::basis::SpaceKind;
use quspin_core::basis::dispatch::GenericBasis;
use quspin_core::dtype::ValueDType;
use quspin_core::{QMatrixInner, eigvalsh, with_qmatrix};
use quspin_nlce::*;
use std::sync::Arc;

type C64 = Complex<f64>;

/// Full Hilbert space, no symmetries: dense `eigvalsh` of the whole matrix.
fn brute_force<M: Model>(g: &ClusterGraph, model: &M) -> Vec<f64> {
    let op = model.operator(g).unwrap();
    let basis = GenericBasis::new(g.n_sites, 2, SpaceKind::Full, false).unwrap();
    let qm = QMatrixInner::build_spin(&op, &basis, ValueDType::Complex128);
    let dense: Vec<C64> = with_qmatrix!(&qm, _M, _C, mat, {
        mat.to_dense::<C64>(&vec![C64::new(1.0, 0.0); mat.num_coeff()])
            .unwrap()
    });
    eigvalsh(&dense, basis.size()).unwrap()
}

fn rect(w: i32, h: i32) -> Vec<[i32; 2]> {
    (0..w).flat_map(|x| (0..h).map(move |y| [x, y])).collect()
}

fn check<M: Model>(g: &ClusterGraph, model: &M) {
    let solver = ExactDiagSolver::new(vec![1.0]).unwrap();
    let got = solver.spectrum(g, model).unwrap().all_eigenvalues();
    let want = brute_force(g, model);
    assert_eq!(got.len(), want.len());
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-11, "max eigenvalue error {err:e}");
}

#[test]
fn blocked_spectrum_matches_full_space() {
    let models = [
        Xxz::heisenberg(1.0),
        Xxz::xx(1.0),
        Xxz::ising(-1.0),
        Xxz {
            jxy: 0.7,
            jz: 1.3,
            hz: 0.4,
        },
    ];
    let graphs = [
        ClusterGraph::rectangle(1, 1).unwrap(),
        ClusterGraph::rectangle(2, 1).unwrap(),
        ClusterGraph::rectangle(2, 2).unwrap(),
        ClusterGraph::rectangle(3, 2).unwrap(),
        ClusterGraph::rectangle(7, 1).unwrap(),
        // D4 automorphisms from the lattice: reduced to a Z2 × Z2 subgroup.
        SquareLattice.cluster_graph(&rect(3, 3)).unwrap(),
        SquareLattice.cluster_graph(&rect(2, 4)).unwrap(),
        // No automorphisms at all (plain S^z sectors).
        ClusterGraph::new(4, vec![Bond::new(0, 1), Bond::new(1, 2), Bond::new(1, 3)]).unwrap(),
    ];
    for g in &graphs {
        for m in &models {
            check(g, m);
        }
    }
}

#[test]
fn heisenberg_dimer_thermo_is_analytic() {
    // S0·S1: singlet −3/4, triplet +1/4 (×3).
    let temps: Arc<[f64]> = vec![0.3, 1.0, 4.0].into();
    let solver = ExactDiagSolver::new(temps.clone())
        .unwrap()
        .with_magnetization();
    let th = solver
        .solve(
            &ClusterGraph::rectangle(2, 1).unwrap(),
            &Xxz::heisenberg(1.0),
        )
        .unwrap();
    for (i, &t) in temps.iter().enumerate() {
        let b = 1.0 / t;
        let z: f64 = (0.75 * b).exp() + 3.0 * (-0.25 * b).exp();
        let e = (-0.75 * (0.75 * b).exp() + 0.75 * (-0.25 * b).exp()) / z;
        let e2 = (0.5625 * (0.75 * b).exp() + 0.1875 * (-0.25 * b).exp()) / z;
        let chi = b * 2.0 * (-0.25 * b).exp() / z; // ⟨M²⟩: m = ±1 triplets
        assert!((th.ln_z[i] - z.ln()).abs() < 1e-13);
        assert!((th.energy[i] - e).abs() < 1e-13);
        assert!((th.entropy[i] - (z.ln() + b * e)).abs() < 1e-13);
        assert!((th.specific_heat[i] - b * b * (e2 - e * e)).abs() < 1e-13);
        assert!(th.magnetization.as_ref().unwrap()[i].abs() < 1e-14);
        assert!((th.susceptibility.as_ref().unwrap()[i] - chi).abs() < 1e-13);
    }
}

#[test]
fn field_gives_nonzero_magnetization() {
    let temps: Arc<[f64]> = vec![0.5].into();
    let solver = ExactDiagSolver::new(temps).unwrap().with_magnetization();
    let model = Xxz {
        jxy: 0.0,
        jz: 0.0,
        hz: 1.0,
    };
    let th = solver
        .solve(&ClusterGraph::rectangle(1, 1).unwrap(), &model)
        .unwrap();
    // Free spin: ⟨S^z⟩ = tanh(h / 2T) / 2.
    let want = 0.5 * (1.0f64).tanh();
    assert!((th.magnetization.unwrap()[0] - want).abs() < 1e-14);
}

#[test]
fn rejects_bad_temperatures() {
    assert!(ExactDiagSolver::new(Vec::<f64>::new()).is_err());
    assert!(ExactDiagSolver::new(vec![1.0, 0.0]).is_err());
    assert!(ExactDiagSolver::new(vec![f64::NAN]).is_err());
}
