# QMatrix & Hamiltonian

Sparse matrix construction (`QMatrix`), time-dependent Hamiltonian wrapper
(`Hamiltonian` with `Static` and Python callable coefficients),
matrix-exponential action (`ExpmOp` + `ExpmWorker` / `ExpmWorker2`), and the
Schrödinger-equation integrator (`SchrodingerEq`).

## QMatrix

::: quspin_rs._rs.QMatrix

## Hamiltonian

::: quspin_rs._rs.Hamiltonian

## Static

::: quspin_rs._rs.Static

## Matrix exponential

The matrix-exponential action `f ← exp(a·A) · f` is exposed through three
cooperating types:

1. **`QMatrixLinearOperator`** — a snapshot of a `QMatrix` (or a
   `Hamiltonian` evaluated at a specific `time`) paired with a fully
   evaluated coefficient vector.  Construct with
   `QMatrix.as_linearoperator(coeffs)` or
   `Hamiltonian.as_linearoperator(time)`.
2. **`ExpmOp`** — caches the partitioned-Taylor parameters
   `(μ, s, m_star, tol)` for a given `(qop, a)` pair.  Construction runs
   the parameter selection exactly once.
3. **`ExpmWorker`** (1-D) / **`ExpmWorker2`** (2-D batch) — packages
   reusable scratch memory bound to an `ExpmOp`.  Built via
   `expm_op.worker(n_vec, work)`, which dispatches at construction time:
   `n_vec == 0` (the default) returns `ExpmWorker` for single-vector
   application; `n_vec > 0` returns `ExpmWorker2` whose `apply` accepts
   a `(dim, k)` array with `k <= n_vec`.

```python
import numpy as np
from quspin_rs._rs import (
    ExpmOp, Hamiltonian, PauliOperator, QMatrix, SpinBasis, Static,
)

H = PauliOperator([("XX", [[1.0, 0, 1]])])
basis = SpinBasis.full(2)
mat = QMatrix.build_pauli(H, basis, np.dtype("float64"))
ham = Hamiltonian(mat, [Static()])

# Snapshot the Hamiltonian at t=0 and cache the Taylor parameters once.
qop = ham.as_linearoperator(0.0)
expm_op = ExpmOp(qop, a=-1j * 0.05)            # `a` = -i·dt for time evolution

# Single-vector worker (n_vec=0, the default).
worker = expm_op.worker()
psi = np.array([1, 0, 0, 0], dtype=np.complex128)
worker.apply(psi)         # psi ← exp(-i·dt·H) · psi
worker.apply(psi)         # second step, no allocations

# Batch worker — accepts a (dim, k) array with k <= n_vec.
batch_worker = expm_op.worker(n_vec=4)
Psi = np.eye(4, dtype=np.complex128)
batch_worker.apply(Psi)   # Psi ← exp(-i·dt·H) · Psi  (one column per state)
```

The `apply` methods borrow the input numpy array directly — no allocations
on the hot path.  The optional `work=` argument to `worker(...)` lets the
caller pre-allocate the worker's scratch buffer.

### QMatrixLinearOperator

::: quspin_rs._rs.QMatrixLinearOperator

### ExpmOp

::: quspin_rs._rs.ExpmOp

### ExpmWorker

::: quspin_rs._rs.ExpmWorker

### ExpmWorker2

::: quspin_rs._rs.ExpmWorker2

## SchrodingerEq

::: quspin_rs._rs.SchrodingerEq
