"""QuSpin Rust core — standalone development shim.

This file is **excluded from the distributed wheel** (see ``[tool.maturin]``
``exclude`` in ``pyproject.toml``).  It exists only so that ``import quspin``
works when developing ``quspin-rs`` in isolation without the original QuSpin
package installed.

Integration contract (Option A)
--------------------------------
When the original ``quspin`` package adds ``quspin-rs`` as a dependency it
should import from ``._rs`` wherever it dispatches to the Rust backend, e.g.::

    from quspin._rs import PyHardcoreHamiltonian, PyHardcoreBasis, PyQMatrix

The original ``quspin/__init__.py`` then owns the namespace; this file is
never shipped in the wheel.
"""

from quspin._rs import (
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
