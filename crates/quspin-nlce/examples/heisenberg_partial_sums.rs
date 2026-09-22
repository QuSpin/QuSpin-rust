//! Rectangle NLCE for the square-lattice spin-1/2 Heisenberg antiferromagnet
//! `H = Σ S_i · S_j`: prints the bare partial sums of E/N, S/N and C/N order
//! by order.
//!
//! ```sh
//! cargo run --release -p quspin-nlce --example heisenberg_partial_sums [max_order]
//! ```
//!
//! `max_order` (= m + n of the largest rectangle) defaults to 8, which
//! includes the 16-site 4 × 4 cluster (a few tens of seconds in release).

use quspin_nlce::*;

/// Selects one quantity of a [`Thermo`].
type Getter = fn(&Thermo) -> &Vec<f64>;

fn main() -> Result<(), NlceError> {
    let max_order: usize = match std::env::args().nth(1) {
        Some(s) => s.parse().map_err(|_| {
            NlceError::InvalidInput(format!("max_order must be an integer, got {s:?}"))
        })?,
        None => 8,
    };
    let temps = vec![0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
    let solver = ExactDiagSolver::new(temps.clone())?;
    let generator = RectangleGenerator::new(SquareLattice, max_order);

    let clusters = generator.clusters()?;
    println!(
        "Heisenberg AFM, square lattice: {} rectangle types up to order m + n = {max_order}",
        clusters.len()
    );
    let result = run_nlce(&generator, &solver, &Xxz::heisenberg(1.0))?;

    let quantities: [(&str, Getter); 3] = [
        ("E/N", |p| &p.energy),
        ("S/N", |p| &p.entropy),
        ("C/N", |p| &p.specific_heat),
    ];
    for (name, get) in quantities {
        println!("\n{name}: bare partial sums");
        print!("{:>6}", "order");
        for t in &temps {
            print!("{:>11}", format!("T={t}"));
        }
        println!();
        for (order, sum) in &result.partial_sums {
            print!("{order:>6}");
            for v in get(sum) {
                print!("{v:>11.6}");
            }
            println!();
        }
    }

    let sums: Vec<Thermo> = result.partial_sums.iter().map(|(_, p)| p.clone()).collect();
    let best = Bare.resum(&sums)?;
    println!(
        "\nBare resummation (highest order), E/N at T=1: {:.6}",
        best.energy[2]
    );
    Ok(())
}
