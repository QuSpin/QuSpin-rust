import math

import numpy as np

from quspin_rs._rs import (
    Hamiltonian,
    PauliOperator,
    QMatrix,
    SchrodingerEq,
    SpinBasis,
    Static,
)


def _make_operator(L: int) -> PauliOperator:
    """Nearest-neighbor XX and YY interactions on L sites (PBC)."""
    bonds = [[1.0, i, (i + 1) % L] for i in range(L)]
    return PauliOperator([("xx", bonds)], [("yy", bonds)])


def _translation_group(
    L: int, k: int = 0
) -> list[tuple[list[int], tuple[float, float]]]:
    """Non-identity elements of the cyclic translation group on L sites.

    Each element is (perm, (re, im)) where perm = T^n and the character is
    exp(2*pi*i*k*n/L) for momentum sector k. The identity (T^0) is implicit
    in `SymBasis` and is not included.
    """
    elements = []
    for power in range(1, L):
        perm = [(i + power) % L for i in range(L)]
        angle = 2 * math.pi * k * power / L
        # TODO: change input to complex number instead of tuple
        elements.append((perm, (math.cos(angle), math.sin(angle))))
    return elements


# TODO build example onf PZ symmetry group, particle-hole symmetry, etc. need to extend symmetry groups

L = 28
M = L // 2
op = _make_operator(L)
seeds = [M * "1" + (L - M) * "0"]
# seeds.append("1" + "0" * (L - 1))

# TODO: where is spin-inversion???
symmetries = _translation_group(L)
basis = SpinBasis.symmetric(L, op, seeds, symmetries)
print(basis)

# TODO:
# * create and expose nbytes calculation to python
# * create some way of visualizing the matrix in, something like Julia's sparse
# * why Fermions?
# * panic if matrix casting loses precision, e.g. casting complex to float with non-zeron imaginary value
Q = QMatrix.build_pauli(op, basis, np.dtype("float64"))


print(Q.dim, Q.num_coeff)


def Hx(t):
    return t


ham = Hamiltonian(Q, [Static(), Hx])
psi0 = np.zeros(basis.size, dtype=np.complex128)
print(basis.index("0" * L))
psi0[basis.index("0" * L)] = 1.0
output = np.zeros_like(psi0)

# TODO:
# * see if we can merge `dot` and `dot_many` into a single python biniding.
# * expose dot_transpose in the same way.
# * Try to make output and overwrite optional
#
ham.dot_many(0.5, psi0.reshape((-1, 1)), output.reshape((-1, 1)), True)


# TODO: Fix this this interface, should have input and output arrays
# This method is going to be removed later on. after issue https://github.com/QuSpin/QuSpin-rust/issues/33
a = 0.01j
print(psi0)
ham.expm_dot(0.0, a, input1 := psi0.copy())
print(input1)
ham.expm_dot(0.0, a, input2 := input1.copy())
print(input2)
ham.expm_dot(0.0, a, input3 := input2.copy())
print(input3)

print(ham.dtype)
indptr, rowptr, data = ham.to_csr(0.2343, True)
print(data)
result = ham.to_dense(1.0)
# print(result)


# TODO make this Stateful object that:
# se.set_initial_value(...)
# psi_1 = se.solve_till(t1)
# psi_2 = se.solve_till(t2)
# psi_1/psi_2 read-only reference to internal state
se = SchrodingerEq(ham)
se.integrate_dense()

# lindblad evolution
