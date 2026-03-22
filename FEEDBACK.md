# Symmetry/Transformations (quspin-core)

I think the symmetry transformation should have the system size `n_sites` as an attribute of the object.
After that we can update the behavior of bitflip to make the `locs` field optional, and if
it is None the mask simply spans all `n_sites`. Validation about the full synnetry group is
discussed below

So you would have something like `BitFlip(n_sites: u32, locs: Optional<Vec<u32>>)`


# Validation (quspin-core)

* When constructing the QMatrix in the build_* methods we should make sure that the `n_sites` of the basis matches the n_sites of the hamiltonian
* When constructing a symmetric space the `n_sites` should be inffered from the symmetry group.
* All group elements should have the same `n_sites` so this needs to be added as validation


# Restructure dispatched code (quspin-core, quspin-py)

I see that in the python bindings we split up the code into a type erased dispatched objects that are wrapped in PyO3 objects. I think we should actually move the type erased version into quspin core as this type erasure will be useful for building the eventual c++ FFI code.

That means we also need to refactor the file structure of quspin-core to match the file structure of quspin-py
