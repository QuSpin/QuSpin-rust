//! Write NLCE results (per site) for the Heisenberg, Ising and XX models to
//! CSV files in the given directory (default `.`), for plotting:
//!
//! - bare partial sums: rectangle expansion for all three models, plus the
//!   12-bond topological bond expansion for Heisenberg and Ising (one cluster
//!   DAG, reused for both models);
//! - `*_resummed.csv`: Wynn / Euler resummations of those sums.
//!
//! Takes ≈7 min in release.
//!
//! ```sh
//! cargo run --release -p quspin-nlce --example export_csv -- crates/quspin-nlce/plots
//! python crates/quspin-nlce/plots/make_plots.py crates/quspin-nlce/plots   # needs matplotlib
//! ```

use quspin_nlce::*;
use std::io::Write;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// `n` log-spaced temperatures in `[lo, hi]`.
fn log_grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64))
        .collect()
}

/// Run the expansion and write its bare partial sums.
fn export<G: ClusterGenerator>(
    path: &Path,
    generator: &G,
    model: &Xxz,
    temps: &[f64],
) -> Result<NlceResult<Thermo>> {
    let solver = ExactDiagSolver::new(temps.to_vec())?.with_magnetization();
    let result = run_nlce(generator, &solver, model)?;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "order,T,energy,entropy,specific_heat,susceptibility")?;
    for (order, p) in &result.partial_sums {
        let chi = p.susceptibility.as_ref().expect("requested");
        for (i, t) in temps.iter().enumerate() {
            writeln!(
                f,
                "{order},{t},{},{},{},{}",
                p.energy[i], p.entropy[i], p.specific_heat[i], chi[i]
            )?;
        }
    }
    println!("wrote {}", path.display());
    Ok(result)
}

/// Write `(expansion, method, estimate)` rows.
fn export_resummed(path: &Path, rows: &[(&str, String, Thermo)]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "expansion,method,T,energy,entropy,specific_heat")?;
    for (expansion, method, p) in rows {
        for (i, t) in p.temps.iter().enumerate() {
            writeln!(
                f,
                "{expansion},{method},{t},{},{},{}",
                p.energy[i], p.entropy[i], p.specific_heat[i]
            )?;
        }
    }
    println!("wrote {}", path.display());
    Ok(())
}

/// Bare, Wynn and Euler estimates of one partial-sum sequence.
fn resummations(
    expansion: &'static str,
    sums: &[Thermo],
    wynn: &[usize],
    euler: &[usize],
) -> Result<Vec<(&'static str, String, Thermo)>> {
    let mut rows = vec![(expansion, "bare".to_string(), Bare.resum(sums)?)];
    for &cycles in wynn {
        rows.push((
            expansion,
            format!("wynn{cycles}"),
            Wynn { cycles }.resum(sums)?,
        ));
    }
    for &start in euler {
        rows.push((
            expansion,
            format!("euler{start}"),
            Euler { start }.resum(sums)?,
        ));
    }
    Ok(rows)
}

fn main() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&dir)?;
    let grid_ising = log_grid(0.2, 10.0, 121);
    let grid_heis = log_grid(0.2, 20.0, 121);
    export(
        &dir.join("xx_chain.csv"),
        &RectangleGenerator::new(ChainLattice, 15),
        &Xxz::xx(1.0),
        &log_grid(0.1, 10.0, 121),
    )?;
    let ising_rect = export(
        &dir.join("ising.csv"),
        &RectangleGenerator::new(SquareLattice, 8),
        &Xxz::ising(-1.0),
        &grid_ising,
    )?;
    let heis_rect = export(
        &dir.join("heisenberg.csv"),
        &RectangleGenerator::new(SquareLattice, 8),
        &Xxz::heisenberg(1.0),
        &grid_heis,
    )?;
    let bonds = ClusterSet::from_generator(
        &BondGenerator::new(SquareLattice, 12),
        "topological bond expansion, square lattice, 12 bonds",
    )?;
    let heis_bond = export(
        &dir.join("heisenberg_bond.csv"),
        &bonds,
        &Xxz::heisenberg(1.0),
        &grid_heis,
    )?;
    let ising_bond = export(
        &dir.join("ising_bond.csv"),
        &bonds,
        &Xxz::ising(-1.0),
        &grid_ising,
    )?;

    // Rectangle orders 2..=8 (7 sums); bond orders 0..=12 (13 sums, Euler
    // start = order). Odd bond orders vanish for Ising: resum even ones.
    let mut rows = resummations("rect", &heis_rect.sums(), &[1, 2, 3], &[])?;
    rows.extend(resummations(
        "bond",
        &heis_bond.sums(),
        &[2, 3, 4, 5],
        &[3, 5, 7],
    )?);
    export_resummed(&dir.join("heisenberg_resummed.csv"), &rows)?;
    let even: Vec<Thermo> = ising_bond
        .partial_sums
        .iter()
        .filter(|(o, _)| o % 2 == 0)
        .map(|(_, p)| p.clone())
        .collect();
    let mut rows = resummations("rect", &ising_rect.sums(), &[1, 2, 3], &[])?;
    rows.extend(resummations("bond", &even, &[1, 2, 3], &[])?);
    export_resummed(&dir.join("ising_resummed.csv"), &rows)?;
    Ok(())
}
