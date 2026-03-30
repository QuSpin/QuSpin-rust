/// Integration tests: build small Hamiltonians, print basis + dense matrix.
///
/// Run with:  cargo test -p quspin-core --test matrix_display -- --nocapture
use num_complex::Complex;
use quspin_core::basis::BasisSpace;
use quspin_core::basis::seed::state_to_str;
use quspin_core::basis::space::FullSpace;
use quspin_core::hamiltonian::hardcore::hamiltonian::HardcoreOperator;
use quspin_core::hamiltonian::hardcore::op::{HardcoreOp, OpEntry};
use quspin_core::qmatrix::build::build_from_basis;
use quspin_core::qmatrix::matrix::QMatrix;
use smallvec::smallvec;

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

fn op_char(op: HardcoreOp) -> char {
    match op {
        HardcoreOp::X => 'X',
        HardcoreOp::Y => 'Y',
        HardcoreOp::Z => 'Z',
        HardcoreOp::P => '+',
        HardcoreOp::M => '-',
        HardcoreOp::N => 'N',
    }
}

fn fmt_complex(c: Complex<f64>) -> String {
    let re = c.re;
    let im = c.im;
    if im.abs() < 1e-12 {
        format!("{re:+.3}")
    } else if re.abs() < 1e-12 {
        format!("{im:+.3}i")
    } else {
        format!("{re:+.3}{im:+.3}i")
    }
}

fn print_hamiltonian(ham: &HardcoreOperator<u8>) {
    println!("Hamiltonian (max_site={}):", ham.max_site());
    for term in ham.terms() {
        let ops: String = term
            .ops
            .iter()
            .map(|&(op, site)| format!("{}@{}", op_char(op), site))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "  cindex={} coeff={}  {}",
            term.cindex,
            fmt_complex(term.coeff),
            ops
        );
    }
}

fn print_basis(basis: &FullSpace<u32>) {
    println!(
        "Basis (full, n_sites={}, size={}):",
        basis.n_sites(),
        basis.size()
    );
    for i in 0..basis.size() {
        let s = state_to_str(basis.state_at(i), basis.n_sites());
        println!("  {i:>3}. |{s}>");
    }
}

fn print_dense(mat: &QMatrix<f64, i64, u8>, coeff: &[f64]) {
    let dense = mat.to_dense(coeff).unwrap();
    let dim = mat.dim();
    println!("Dense matrix (coeff={coeff:?}, shape={dim}x{dim}):");
    for r in 0..dim {
        let row: Vec<String> = (0..dim)
            .map(|c| format!("{:>8.4}", dense[r * dim + c]))
            .collect();
        println!("  [{}]", row.join("  "));
    }
}

fn show(label: &str, ham: &HardcoreOperator<u8>, basis: &FullSpace<u32>, coeff: &[f64]) {
    let sep = "=".repeat(60);
    println!("\n{sep}");
    println!("{label}");
    println!("{sep}");
    print_hamiltonian(ham);
    println!();
    print_basis(basis);
    println!();
    let mat: QMatrix<f64, i64, u8> = build_from_basis(ham, basis);
    print_dense(&mat, coeff);
}

// ---------------------------------------------------------------------------
// Hamiltonians
// ---------------------------------------------------------------------------

fn single_z(site: u32, _n_sites: usize) -> HardcoreOperator<u8> {
    HardcoreOperator::new(vec![OpEntry::new(
        0u8,
        Complex::new(1.0, 0.0),
        smallvec![(HardcoreOp::Z, site)],
    )])
}

fn single_x(site: u32, _n_sites: usize) -> HardcoreOperator<u8> {
    HardcoreOperator::new(vec![OpEntry::new(
        0u8,
        Complex::new(1.0, 0.0),
        smallvec![(HardcoreOp::X, site)],
    )])
}

fn two_body(op0: HardcoreOp, op1: HardcoreOp, _n_sites: usize) -> HardcoreOperator<u8> {
    HardcoreOperator::new(vec![OpEntry::new(
        0u8,
        Complex::new(1.0, 0.0),
        smallvec![(op0, 0), (op1, 1)],
    )])
}

fn heisenberg_chain(n_sites: usize) -> HardcoreOperator<u8> {
    // H = Σ_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    // All bonds share cindex=0.
    let mut terms = Vec::new();
    for i in 0..(n_sites - 1) as u32 {
        for (oa, ob) in [
            (HardcoreOp::X, HardcoreOp::X),
            (HardcoreOp::Y, HardcoreOp::Y),
            (HardcoreOp::Z, HardcoreOp::Z),
        ] {
            terms.push(OpEntry::new(
                0u8,
                Complex::new(1.0, 0.0),
                smallvec![(oa, i), (ob, i + 1)],
            ));
        }
    }
    HardcoreOperator::new(terms)
}

/// H = J Σ_i ZZ_{i,i+1} + h Σ_i X_i
/// cindex=0 → J (ZZ bonds), cindex=1 → h (X field)
fn tfi_chain(n_sites: usize) -> HardcoreOperator<u8> {
    let mut terms = Vec::new();
    for i in 0..(n_sites - 1) as u32 {
        terms.push(OpEntry::new(
            0u8,
            Complex::new(1.0, 0.0),
            smallvec![(HardcoreOp::Z, i), (HardcoreOp::Z, i + 1)],
        ));
    }
    for i in 0..n_sites as u32 {
        terms.push(OpEntry::new(
            1u8,
            Complex::new(1.0, 0.0),
            smallvec![(HardcoreOp::X, i)],
        ));
    }
    HardcoreOperator::new(terms)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Assertion helper
// ---------------------------------------------------------------------------

const TOL: f64 = 1e-12;

fn assert_dense_eq(mat: &QMatrix<f64, i64, u8>, coeff: &[f64], expected: &[f64]) {
    let dense = mat.to_dense(coeff).unwrap();
    assert_eq!(
        dense.len(),
        expected.len(),
        "dense length mismatch: got {}, expected {}",
        dense.len(),
        expected.len()
    );
    for (i, (&got, &exp)) in dense.iter().zip(expected.iter()).enumerate() {
        let dim = mat.dim();
        assert!(
            (got - exp).abs() < TOL,
            "dense[{},{}] = {got:.6}, expected {exp:.6}",
            i / dim,
            i % dim,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn display_z_1site() {
    // Basis (descending): 0→|1>, 1→|0>
    // QuSpin convention: Z|1> = +|1>, Z|0> = -|0>  →  diagonal [+1, -1]
    let ham = single_z(0, 1);
    let basis = FullSpace::<u32>::new(2, 1, false);
    show("Z on 1 site", &ham, &basis, &[1.0]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0], &[
         1.0,  0.0,
         0.0, -1.0,
    ]);
}

#[test]
fn display_x_1site() {
    // X|1> = |0>, X|0> = |1>  →  off-diagonal [[0,1],[1,0]]
    let ham = single_x(0, 1);
    let basis = FullSpace::<u32>::new(2, 1, false);
    show("X on 1 site", &ham, &basis, &[1.0]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0], &[
        0.0, 1.0,
        1.0, 0.0,
    ]);
}

#[test]
fn display_xx_2site() {
    // XX flips both bits: |11>↔|00>, |01>↔|10>  →  anti-diagonal
    let ham = two_body(HardcoreOp::X, HardcoreOp::X, 2);
    let basis = FullSpace::<u32>::new(2, 2, false);
    show("XX on 2 sites", &ham, &basis, &[1.0]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0], &[
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
    ]);
}

#[test]
fn display_yy_2site() {
    // YY: amplitude i*s0 * i*s1; for |11>→|00>: i*(-1) * i*(-1) = -1
    //                              for |00>→|11>: i*(+1) * i*(+1) = -1
    //                              for |01>↔|10>: +1 each
    let ham = two_body(HardcoreOp::Y, HardcoreOp::Y, 2);
    let basis = FullSpace::<u32>::new(2, 2, false);
    show("YY on 2 sites", &ham, &basis, &[1.0]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0], &[
         0.0,  0.0,  0.0, -1.0,
         0.0,  0.0,  1.0,  0.0,
         0.0,  1.0,  0.0,  0.0,
        -1.0,  0.0,  0.0,  0.0,
    ]);
}

#[test]
fn display_zz_2site() {
    // ZZ diagonal: (+1)(+1)=+1 for |11>, (-1)(+1)=-1 for |01>,
    //              (+1)(-1)=-1 for |10>, (+1)(+1)=+1 for |00>
    //   (Z conv: s = 1-2n, so s=−1 for occupied, s=+1 for empty)
    let ham = two_body(HardcoreOp::Z, HardcoreOp::Z, 2);
    let basis = FullSpace::<u32>::new(2, 2, false);
    show("ZZ on 2 sites", &ham, &basis, &[1.0]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0], &[
         1.0,  0.0,  0.0,  0.0,
         0.0, -1.0,  0.0,  0.0,
         0.0,  0.0, -1.0,  0.0,
         0.0,  0.0,  0.0,  1.0,
    ]);
}

#[test]
fn display_heisenberg_2site() {
    // H = XX + YY + ZZ
    // |11>: ZZ=+1, XX and YY off-diag to |00>  →  row [+1, 0, 0, 0]
    // |01>: ZZ=-1, XX+YY connect to |10> with amp 1+1=2  →  row [0,-1,+2,0]
    // |10>: ZZ=-1, XX+YY connect to |01> with amp 1+1=2  →  row [0,+2,-1,0]
    // |00>: ZZ=+1, XX and YY off-diag to |11>  →  row [0, 0, 0, +1]
    let ham = heisenberg_chain(2);
    let basis = FullSpace::<u32>::new(2, 2, false);
    show("Heisenberg XXX chain, 2 sites", &ham, &basis, &[1.0]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0], &[
        1.0,  0.0,  0.0,  0.0,
        0.0, -1.0,  2.0,  0.0,
        0.0,  2.0, -1.0,  0.0,
        0.0,  0.0,  0.0,  1.0,
    ]);
}

#[test]
fn display_heisenberg_3site() {
    // Heisenberg XXX chain, L=3, 8x8.
    // Reference generated with QuSpin spin_basis_1d(L=3, pauli=1),
    // static=[["xx",bonds],["yy",bonds],["zz",bonds]].
    // QuSpin and our FullSpace both enumerate states in descending order [7..0].
    let ham = heisenberg_chain(3);
    let basis = FullSpace::<u32>::new(2, 3, false);
    show("Heisenberg XXX chain, 3 sites", &ham, &basis, &[1.0]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0], &[
         2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  2.0, -2.0,  0.0,  2.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,
         0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  2.0,  0.0, -2.0,  2.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,
    ]);
}

#[test]
fn display_heisenberg_4site() {
    // Heisenberg XXX chain, L=4, 16x16.
    // Reference generated with QuSpin spin_basis_1d(L=4, pauli=1).
    let ham = heisenberg_chain(4);
    let basis = FullSpace::<u32>::new(2, 4, false);
    show("Heisenberg XXX chain, 4 sites", &ham, &basis, &[1.0]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0], &[
         3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  1.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  2.0, -1.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  1.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  2.0,  0.0, -1.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  2.0,  0.0, -3.0,  2.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  2.0, -1.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0, -1.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  2.0, -3.0,  0.0,  2.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0, -1.0,  0.0,  2.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  1.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0, -1.0,  2.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  1.0,  0.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  3.0,
    ]);
}

#[test]
fn display_tfi_3site() {
    // Transverse-field Ising chain, L=3: H = J Σ ZZ_{i,i+1} + h Σ X_i
    // J=1 (cindex=0), h=0.5 (cindex=1).
    // Reference generated with QuSpin spin_basis_1d(L=3, pauli=1),
    // static=[["zz",zz_bonds],["x",x_field]].
    let ham = tfi_chain(3);
    let basis = FullSpace::<u32>::new(2, 3, false);
    show("TFI chain J=1 h=0.5, 3 sites", &ham, &basis, &[1.0, 0.5]);
    let mat: QMatrix<f64, i64, u8> = build_from_basis(&ham, &basis);
    #[rustfmt::skip]
    assert_dense_eq(&mat, &[1.0, 0.5], &[
         2.0,  0.5,  0.5,  0.0,  0.5,  0.0,  0.0,  0.0,
         0.5,  0.0,  0.0,  0.5,  0.0,  0.5,  0.0,  0.0,
         0.5,  0.0, -2.0,  0.5,  0.0,  0.0,  0.5,  0.0,
         0.0,  0.5,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5,
         0.5,  0.0,  0.0,  0.0,  0.0,  0.5,  0.5,  0.0,
         0.0,  0.5,  0.0,  0.0,  0.5, -2.0,  0.0,  0.5,
         0.0,  0.0,  0.5,  0.0,  0.5,  0.0,  0.0,  0.5,
         0.0,  0.0,  0.0,  0.5,  0.0,  0.5,  0.5,  2.0,
    ]);
}
