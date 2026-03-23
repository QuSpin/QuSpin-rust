"""QuSpin Rust core.

``quspin_rs`` is a standalone package that the original ``quspin`` package
depends on.  It exposes the compiled Rust extension (``quspin_rs._rs``) as a
clean public API.

Integration contract (Option A)
---------------------------------
The original ``quspin`` package adds ``quspin-rs`` as an install-time
dependency and imports from ``quspin_rs`` wherever it dispatches to the Rust
backend::

    from quspin_rs import PyHardcoreHamiltonian, PyHardcoreBasis, PyQMatrix
"""

from quspin_rs._rs import (
    PyGrpElement,
    PyHardcoreBasis,
    PyHardcoreHamiltonian,
    PyLatticeElement,
    PyQMatrix,
    PySymmetryGrp,
)

__all__ = [
    "PyGrpElement",
    "PyHardcoreBasis",
    "PyHardcoreHamiltonian",
    "PyLatticeElement",
    "PyQMatrix",
    "PySymmetryGrp",
]
