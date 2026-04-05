import math

import numpy as np

from quspin_rs._rs import PauliOperator, QMatrix, SpinBasis


def _make_operator(L: int) -> PauliOperator:
    """Single-particle XX+YY hopping on L sites with periodic boundary conditions."""
    zz_bonds = [[1.0, i, (i + 1) % L] for i in range(L)]
    x_bonds = [[1.0, i] for i in range(L)]
    return PauliOperator([("zz", zz_bonds)], [("x", x_bonds)])


def _translation_group(
    L: int, k: int = 0
) -> list[tuple[list[int], tuple[float, float]]]:
    """All L elements of the cyclic translation group on L sites.

    Each element is (perm, (re, im)) where perm = T^n and the character is
    exp(2*pi*i*k*n/L) for momentum sector k.
    """
    elements = []
    for power in range(L):
        perm = [(i + power) % L for i in range(L)]
        angle = 2 * math.pi * k * power / L
        elements.append((perm, (math.cos(angle), math.sin(angle))))
    return elements


L = 30
op = _make_operator(L)
seed = "0" * L
symmetries = _translation_group(L)
basis = SpinBasis.symmetric(L, op, [seed], symmetries)
print(basis.size)

Q = QMatrix.build_pauli(op, basis, np.dtype("float32"))

print(Q.dim, Q.num_coeff)
