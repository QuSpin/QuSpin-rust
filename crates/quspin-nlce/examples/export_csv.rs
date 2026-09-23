//! Write bare NLCE partial sums (per site) for the Heisenberg, Ising and XX
//! models to CSV files in the given directory (default `.`), for plotting.
//!
//! ```sh
//! cargo run --release -p quspin-nlce --example export_csv -- crates/quspin-nlce/plots
//! python crates/quspin-nlce/plots/make_plots.py crates/quspin-nlce/plots   # needs matplotlib
//! ```

use quspin_nlce::*;
use std::io::Write;

/// `n` log-spaced temperatures in `[lo, hi]`.
fn log_grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64))
        .collect()
}

fn export<G: ClusterGenerator>(
    path: &std::path::Path,
    generator: &G,
    model: &Xxz,
    temps: Vec<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let solver = ExactDiagSolver::new(temps.clone())?.with_magnetization();
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
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&dir)?;
    export(
        &dir.join("xx_chain.csv"),
        &RectangleGenerator::new(ChainLattice, 15),
        &Xxz::xx(1.0),
        log_grid(0.1, 10.0, 121),
    )?;
    export(
        &dir.join("ising.csv"),
        &RectangleGenerator::new(SquareLattice, 8),
        &Xxz::ising(-1.0),
        log_grid(0.2, 10.0, 121),
    )?;
    export(
        &dir.join("heisenberg.csv"),
        &RectangleGenerator::new(SquareLattice, 8),
        &Xxz::heisenberg(1.0),
        log_grid(0.2, 20.0, 121),
    )?;
    Ok(())
}
