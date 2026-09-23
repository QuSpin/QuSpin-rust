//! Rectangle NLCE for the square-lattice spin-1/2 Heisenberg antiferromagnet
//! `H = Σ S_i · S_j`: prints the bare partial sums of E/N, S/N and C/N order
//! by order.
//!
//! ```sh
//! cargo run --release -p quspin-nlce --example heisenberg_partial_sums [max_order] [rect|bond]
//! ```
//!
//! - `rect` (default): rectangle expansion, `max_order = m + n` of the largest
//!   rectangle, default 8 (includes the 16-site 4 × 4 cluster; a few tens of
//!   seconds in release).
//! - `bond`: topological bond expansion, `max_order` = number of bonds,
//!   default 12 (4424 topologies, up to 13 sites).
//! - `load <file>`: a cluster DAG saved by `bond_census` (or
//!   [`ClusterSet::save`]), truncated to `max_order`.

use quspin_nlce::*;

/// Selects one quantity of a [`Thermo`].
type Getter = fn(&Thermo) -> &Vec<f64>;

fn main() -> Result<(), NlceError> {
    let kind = std::env::args().nth(2).unwrap_or_else(|| "rect".into());
    let max_order: usize = match std::env::args().nth(1) {
        Some(s) => s.parse().map_err(|_| {
            NlceError::InvalidInput(format!("max_order must be an integer, got {s:?}"))
        })?,
        None if kind == "bond" || kind == "load" => 12,
        None => 8,
    };
    let temps = vec![0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
    let solver = ExactDiagSolver::new(temps.clone())?;
    let t0 = std::time::Instant::now();
    let result = match kind.as_str() {
        "rect" => {
            println!("Heisenberg AFM, square lattice: rectangles up to order m + n = {max_order}");
            run_nlce(
                &RectangleGenerator::new(SquareLattice, max_order),
                &solver,
                &Xxz::heisenberg(1.0),
            )?
        }
        "bond" => {
            println!(
                "Heisenberg AFM, square lattice: topological bond clusters up to {max_order} bonds"
            );
            run_nlce(
                &BondGenerator::new(SquareLattice, max_order),
                &solver,
                &Xxz::heisenberg(1.0),
            )?
        }
        "load" => {
            let path = std::env::args()
                .nth(3)
                .ok_or_else(|| NlceError::InvalidInput("`load` needs a file path".into()))?;
            let set = ClusterSet::load(&path)?;
            println!("Heisenberg AFM: {} (order <= {max_order})", set.description);
            run_nlce(&set.truncated(max_order), &solver, &Xxz::heisenberg(1.0))?
        }
        other => {
            return Err(NlceError::InvalidInput(format!(
                "expansion must be `rect`, `bond` or `load`, got {other:?}"
            )));
        }
    };
    println!(
        "{} cluster types generated, solved and combined in {:.1?}",
        result.cache.properties.len(),
        t0.elapsed()
    );

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
