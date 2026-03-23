"""Generate reference dense matrices using QuSpin for integration test validation."""

import numpy as np

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def fmt_matrix(mat, label):
    arr = np.array(mat)
    n = arr.shape[0]
    print(f"\n// {label}")
    print(f"// dim={n}x{n}")
    rows = []
    for r in range(n):
        entries = ", ".join(f"{arr[r,c]:+.1f}" for c in range(n))
        rows.append(f"        {entries},")
    print("    #[rustfmt::skip]")
    print("    assert_dense_eq(&mat, &[1.0], &[")
    for row in rows:
        print(row)
    print("    ]);")


def show_basis(basis, label):
    print(f"\n// {label} basis.states (QuSpin order, index 0 = first row/col):")
    print(f"// {basis.states}")


def make_ham(L, static):
    basis = spin_basis_1d(L=L, pauli=1)
    H = hamiltonian(
        static, [], basis=basis, check_pcon=False, check_herm=False, check_symm=False
    )
    return H, basis


# ── 1-site operators ──────────────────────────────────────────────────────────
print("=" * 60)
print("1-SITE OPERATORS (L=1)")
print("=" * 60)

for op in ["z", "x", "y"]:
    H, basis = make_ham(1, [[op, [[1.0, 0]]]])
    show_basis(basis, f"{op.upper()} L=1")
    fmt_matrix(H.todense(), f"single {op.upper()} site=0, L=1")

# ── 2-site two-body operators ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2-SITE TWO-BODY OPERATORS (L=2)")
print("=" * 60)

for op in ["xx", "yy", "zz", "+-", "-+"]:
    H, basis = make_ham(2, [[op, [[1.0, 0, 1]]]])
    show_basis(basis, f"{op.upper()} L=2")
    fmt_matrix(H.todense(), f"two-body {op.upper()} sites=(0,1), L=2")

# ── Heisenberg chain ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HEISENBERG CHAIN")
print("=" * 60)

for L in [2, 3, 4]:
    bonds = [[1.0, i, i + 1] for i in range(L - 1)]
    static = [["xx", bonds], ["yy", bonds], ["zz", bonds]]
    H, basis = make_ham(L, static)
    show_basis(basis, f"Heisenberg L={L}")
    fmt_matrix(H.todense(), f"Heisenberg XXX chain L={L}")

# ── Transverse-field Ising ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRANSVERSE-FIELD ISING (L=3)")
print("=" * 60)
# H = J Σ ZZ + h Σ X,  J=1, h=0.5
L = 3
zz_bonds = [[1.0, i, i + 1] for i in range(L - 1)]
x_field = [[0.5, i] for i in range(L)]
static = [["zz", zz_bonds], ["x", x_field]]
H, basis = make_ham(L, static)
show_basis(basis, "TFI L=3")
fmt_matrix(H.todense(), "TFI J=1 h=0.5 L=3")

# ── Mixed +- / -+ / zz (XXZ) ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("XXZ CHAIN (L=3, Jxy=1, Jz=0.5)")
print("=" * 60)
L = 3
bonds = [[1.0, i, i + 1] for i in range(L - 1)]
zz_bonds = [[0.5, i, i + 1] for i in range(L - 1)]
static = [["+-", bonds], ["-+", bonds], ["zz", zz_bonds]]
H, basis = make_ham(L, static)
show_basis(basis, "XXZ L=3")
fmt_matrix(H.todense(), "XXZ Jxy=1 Jz=0.5 L=3")
