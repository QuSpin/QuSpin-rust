import numpy as np

from quspin_rs import Lattice, SymmetryGroup
from quspin_rs._rs import (
    ExpmOp,
    GenericBasis,
    Hamiltonian,
    MonomialOperator,
    QMatrix,
    Static,
)


def dimer_operator(L: int, pbc: bool = True):
    # dimer chain space PBC:
    # space: 3*L bits, one bit per bond.
    # * - * - * -
    # |   |   |
    # * - * - * -
    # flat index: i
    # mid ==> i % 3 == 0
    # top ==> i % 3 == 1
    # bottom ==> i % 3 == 2
    #
    # seed: 100100...
    # Kinetic + Potential term acts on [3*i...(3*i+4) % 3 * L for i in range(L)]
    # Kinetic 1001 <-> 0110
    # potential: 0110 or 1001 gets V

    if pbc:
        max_size = 3 * L
        bonds = [
            (3 * i, 3 * i + 1, 3 * i + 2, (3 * i + 3) % max_size) for i in range(L)
        ]
        seed = "".join("100" for i in range(L))
        translation = [(i + 3) % max_size for i in range(len(seed))]

        group = SymmetryGroup(n_sites=len(seed), lhss=2)
        group.add_cyclic(Lattice(translation), k=0)
    else:
        max_size = 3 * L + 1  # the the last rung is new an extra site.
        bonds = [(3 * i, 3 * i + 1, 3 * i + 2, (3 * i + 3)) for i in range(L - 1)]
        seed = ("".join("100" for i in range(L))) + "1"

        group = None

    perm = np.arange(2**4).astype(np.uint64)
    has_v = np.logical_or(perm == 6, perm == 9)

    v_term = (perm.astype(np.intp), has_v.astype(np.complex128), bonds)

    perm = perm ^ perm[-1]
    amp = np.zeros_like(perm, dtype=np.complex128)
    amp[has_v] = 1.0

    t_term = (perm.astype(np.intp), amp, bonds)
    operator = MonomialOperator(v_term, t_term, lhss=2)

    return (seed, operator, group)


seed, op, group = dimer_operator(20, pbc=False)
if group is not None:
    basis = GenericBasis.symmetric(group, op, seeds=[seed])
else:
    basis = GenericBasis.subspace(len(seed), 2, op, seeds=[seed])

mat = QMatrix.build_monomial(op, basis, np.dtype("float64"))

ham = Hamiltonian(mat, [Static(), Static()])
h_mat = ham.to_dense(0.0)
expm_op = ExpmOp(ham.as_linearoperator(0.0), a=-1j * 0.05)
worker = expm_op.worker()  # 1-D worker (n_vec=0); 2*dim scratch reused
psi = np.zeros(basis.size, dtype=np.complex128)

psi[0] = 1.0

for i in range(10000000):
    worker.apply(psi)

    print(np.std(np.abs(psi)))
