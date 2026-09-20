"""Example 00 — basis objects: construction, interpretation, and use.

Port of QuSpin's ``examples/scripts/example00.py`` to QuSpin-rust.

Demonstrates how to build spin bases with and without symmetries, how to move
between the three representations of a basis state (array index, integer, Fock
string), and how to project state vectors between a symmetry-reduced basis and
the full Hilbert space.

NOTE ON SITE ORDERING: QuSpin-rust currently writes site 0 **leftmost** in a
Fock string, so ``"01"`` means site 0 empty, site 1 occupied, and encodes the
integer 2. Old QuSpin uses the opposite (site 0 rightmost) convention; the
convention is still due to be flipped to match. See
https://github.com/QuSpin/QuSpin-rust/issues/74.
"""

import numpy as np

from quspin_rs import Lattice, PauliOperator, SpinBasis, SymmetryGroup

L = 2  # system size

# ---------------------------------------------------------------------------
# No symmetries
# ---------------------------------------------------------------------------
print("\n----------------------------")
print("---  NO  SYMMETRIES  -------")
print("----------------------------\n")

basis = SpinBasis.full(L)
print(basis)

# `states` is the third column when printing the basis (not consecutive once
# symmetries are present -- see below); `array_inds` is the first column and is
# always consecutive.
states = basis.states
array_inds = np.arange(basis.Ns)

print("\n'array index' and 'states' columns when printing the basis:")
print("array indices:   ", array_inds)
print("states in int rep:", states)

# Find the array index of a state from its integer representation. The array
# index is what you need to read off matrix elements.
s = basis.states[2]
array_ind_s = basis.index(s)
print("\narray index of s, and s (in int rep):")
print(array_ind_s, s)

# ---------------------------------------------------------------------------
# States: ket and integer representations
# ---------------------------------------------------------------------------

# Integer representation from a Fock-state string. The ket-forming '|' and '>'
# are optional, and sites may also be given as separate tokens ("0 1", "0,1").
fock_state_str_s = "|01>"
int_rep_s = basis.state_to_int(fock_state_str_s)
print("\nFock state string of s, and s (in int rep):")
print(fock_state_str_s, int_rep_s)

# ... and back again.
fock_s = basis.int_to_state(int_rep_s, bracket_notation=True)
print("\nFock state string of s, and s (in int rep):")
print(fock_s, int_rep_s)

fock_s = basis.int_to_state(int_rep_s, bracket_notation=False)
print("Fock state string (without | and >) of s, and s (in int rep):")
print(fock_s, int_rep_s)

# Fock state from an array index.
array_ind_s = 2
int_rep_s = basis.states[array_ind_s]
fock_s = basis.int_to_state(int_rep_s, bracket_notation=True)
print("\narray index, int rep, and Fock state rep of s:")
print(array_ind_s, int_rep_s, fock_s)  # compare with print(basis) above

# ---------------------------------------------------------------------------
# States: array/vector representation
# ---------------------------------------------------------------------------

psi_s = np.zeros(basis.Ns)
array_ind_s = basis.index(basis.state_to_int("01"))
psi_s[array_ind_s] = 1.0  # the pure state |01>
print("\nstate psi_s in the basis:")
print(psi_s)

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
print("\n----------------------------")
print("-------  OPERATORS  -------")
print("----------------------------\n")

# PauliOperator objects are required to generate basis objects with symmetries.
#
# A two-site Pauli operator which breaks all symmetries except reflection.
twosite_op = PauliOperator(
    [
        # XX interaction between sites 0 and 1, strength 0.73
        ("XX", [[0.73, 0, 1]]),
        # transverse field along X on both sites, strength 0.31
        ("X", [[0.31, 0], [0.31, 1]]),
        # longitudinal field along Z on both sites, strength 0.17
        ("Z", [[0.17, 0], [0.17, 1]]),
        # XZ interaction both ways between sites 0 and 1, strength 0.23
        ("XZ", [[0.23, 0, 1], [0.23, 1, 0]]),
    ]
)

print("\nA PauliOperator is not yet a Hamiltonian matrix; printing it returns:")
print(twosite_op)

# An XX chain with periodic boundary conditions: has more symmetries than just
# reflection/parity.
bonds = [[1.0, i, (i + 1) % L] for i in range(L)]
XX_chain_op = PauliOperator([("XX", bonds)])

# ---------------------------------------------------------------------------
# Symmetries
# ---------------------------------------------------------------------------
print("\n\n\n----------------------------")
print("-------  SYMMETRIES  -------")
print("----------------------------\n")

# A basis with symmetries is generated from a symmetry group plus a
# PauliOperator, which generates the states in a given symmetry sector by
# repeated application to a set of seed states.

sites = np.arange(L)  # lattice sites
P = sites[::-1]  # action of parity/reflection on the lattice sites

# Triplet group: parity sector +1. `lhss` is the local Hilbert space size.
group_triplet = SymmetryGroup(n_sites=L, lhss=2)
group_triplet.add_cyclic(Lattice(P), eta=+1)

# Singlet group: parity sector -1.
group_singlet = SymmetryGroup(n_sites=L, lhss=2)
group_singlet.add_cyclic(Lattice(P), eta=-1)
group_singlet.validate()  # check that the group is closed

# ---------------------------------------------------------------------------
# Minimal demo: why the operator and seed choice matter
# ---------------------------------------------------------------------------
print("\nOperator/seed demo (triplet sector, eta=+1):\n")

# XX_chain_op has extra symmetries, so one seed may not reach all triplet
# states.
triplet_xx_one_seed = SpinBasis.symmetric(group_triplet, XX_chain_op, seeds=["01"])
print("1) XX_chain_op + seeds=['01'] -> size:", triplet_xx_one_seed.size)
print("   states:", triplet_xx_one_seed.states)
print("   one seed does not reach the whole triplet block.\n")

# A second seed reaches the disconnected component and recovers the full block.
triplet_xx_two_seeds = SpinBasis.symmetric(
    group_triplet, XX_chain_op, seeds=["01", "00"]
)
print("2) XX_chain_op + seeds=['01','00'] -> size:", triplet_xx_two_seeds.size)
print("   states:", triplet_xx_two_seeds.states)
print("   the second seed recovers the full triplet block.\n")

# twosite_op breaks the extra symmetries, so one seed suffices.
triplet_twosite_one_seed = SpinBasis.symmetric(group_triplet, twosite_op, seeds=["01"])
print("3) twosite_op + seeds=['01'] -> size:", triplet_twosite_one_seed.size)
print("   states:", triplet_twosite_one_seed.states)
print("   one seed generates the full triplet block.\n\n")

# Build the singlet and triplet bases.
#
# twosite_op reaches all states in the triplet sector from a single seed;
# acting on |00> with it reaches the 01/10 orbit, which forms the singlet.
basis_triplet = SpinBasis.symmetric(group_triplet, twosite_op, seeds=["01"])
basis_singlet = SpinBasis.symmetric(group_singlet, twosite_op, seeds=["00"])

print("full basis:")
print(basis)

print("\n\npblock=+1 basis:\n")
print(basis_triplet)
print("  * the integer rep column is no longer consecutive: |10> (int 1) falls")
print("    outside this symmetry sector.")
print("  * the array index column is still consecutive, but the indices differ")
print("    from the full basis -- e.g. for |00>.")
print("  * |00> and |11> are invariant under parity, so they correspond to the")
print("    physical states |00> and |11>.")
print("  * |01> is NOT invariant under parity: it represents the physical")
print("    symmetric superposition (|01> + |10>)/sqrt(2). QuSpin keeps track of")
print("    the 1/sqrt(2) under the hood.")

print("\n\npblock=-1 basis:\n")
print(basis_singlet)
print("  * |01> here represents the physical ANTI-symmetric superposition")
print("    (|01> - |10>)/sqrt(2).")
print("  * NOTE: the same reference state |01> labels both the symmetric and")
print("    the antisymmetric superposition, because QuSpin labels a")
print("    superposition by one fixed representative of the orbit.\n")

# ---------------------------------------------------------------------------
# Transform states from one basis to the other
# ---------------------------------------------------------------------------

array_ind_s = basis_triplet.index(basis.state_to_int("01"))
psi_symm_s = np.zeros(basis_triplet.Ns)
psi_symm_s[array_ind_s] = 1.0  # the state (|01> + |10>)/sqrt(2)
print("state psi_symm_s in the symmetry-reduced basis_triplet:")
print(psi_symm_s)

# The corresponding state in the full basis. Note the 1/sqrt(2).
psi_s = basis_triplet.project_from(psi_symm_s, sparse=False)
print("\nstate psi_s in the full basis (note the 1/sqrt(2)):")
print(psi_s)

# A full-basis state can also be projected into a symmetry-reduced basis.
psi_s = np.zeros(basis.Ns)
array_ind_s = basis.index(basis.state_to_int("01"))
psi_s[array_ind_s] = 1.0  # the state |01> in the full basis

# Projects |01> onto (|01> - |10>)/sqrt(2) in basis_singlet.
psi_symm_s = basis_singlet.project_to(psi_s, sparse=False)
print("\nstate psi_symm_s in basis_singlet.")
print("NOTE: the projection is not normalised!")
print(psi_symm_s)

psi_symm_s = psi_symm_s / np.linalg.norm(psi_symm_s)

# Lift the projected state back to the full basis.
psi_lifted_s = basis_singlet.project_from(psi_symm_s, sparse=False)
print("\nstate psi_lifted_s = (|01> - |10>)/sqrt(2) in the full basis.")
print("NOTE: information was lost by the first projection!")
print(psi_lifted_s)
