# QuSpin-rs

Rust rewrite of the [QuSpin](https://github.com/QuSpin/QuSpin) quantum many-body
library, with Python bindings via PyO3 / maturin.

This site documents the Python package `quspin_rs` exported from the `quspin-rs`
project.

## Install

```sh
just sync       # uv sync --dev --all-extras
just develop    # uv run maturin develop  (off-by-default: large-int)
```

For wide-integer (≥512-bit) bases, build with the feature on:

```sh
just develop --features large-int
```

## Quick tour

```python
import numpy as np
from quspin_rs import (
    Lattice, Local, SpinBasis, SymmetryGroup, PauliOperator, QMatrix,
)
from quspin_rs._rs import ExpmOp, Hamiltonian, Static

n_sites = 4
H = PauliOperator([("XX", [[1.0, i, (i + 1) % n_sites] for i in range(n_sites)])])

# Translation × spin-flip symmetric basis (k=0 momentum, eta=-1 parity sector).
T = SymmetryGroup(n_sites=n_sites, lhss=2)
T.add_cyclic(Lattice([(i + 1) % n_sites for i in range(n_sites)]), k=0)
Z = SymmetryGroup(n_sites=n_sites, lhss=2)
Z.add_cyclic(Local([1, 0]), eta=-1)
group = T.product(Z)

basis = SpinBasis.symmetric(group, H, seeds=["0011"])
print(basis.size)

mat = QMatrix.build_pauli(H, basis, np.dtype("float64"))

# Time-evolve a state under H: psi ← exp(-i·dt·H) · psi
ham = Hamiltonian(mat, [Static()])
expm_op = ExpmOp(ham.as_linearoperator(0.0), a=-1j * 0.05)
worker = expm_op.worker()              # 1-D worker (n_vec=0); 2*dim scratch reused
psi = np.zeros(basis.size, dtype=np.complex128)
psi[0] = 1.0
worker.apply(psi)                      # one timestep, no allocations
```

## Where to start

- **[Symmetry-group guide](guide/symmetry.md)** — how to build groups for
  `*Basis.symmetric(...)`.
- **[Basis types](api/basis.md)** — `SpinBasis`, `FermionBasis`,
  `BosonBasis`, `GenericBasis`.
- **[Symmetry API](api/symmetry.md)** — `SymmetryGroup` plus the
  `Lattice` / `Local` / `Composite` element constructors.
- **[Operators](api/operators.md)** — `PauliOperator`, `BondOperator`,
  `BosonOperator`, `FermionOperator`, `MonomialOperator`.
- **[QMatrix & Hamiltonian](api/qmatrix.md)** — sparse matrix construction
  and time-evolution.

## A working demo

`scripts/symmetry_group_demo.py` walks through five canonical patterns
(translation only, abelian product, PZ composite, dihedral closure, and a
compatibility-error case). Run it with:

```sh
uv run python scripts/symmetry_group_demo.py
```
