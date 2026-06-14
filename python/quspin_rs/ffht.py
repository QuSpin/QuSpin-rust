"""High-level Fast Hadamard Transform (FHT) interface.

Wraps the low-level ``_rs.fht_{f32,f64}[_oop]`` bindings behind a single
``fht`` function that dispatches on ``arr.dtype`` and chooses the
in-place or out-of-place variant based on the ``inplace`` flag.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import _rs

__all__ = ["ffht"]

_INPLACE_DISPATCH = {
    np.dtype(np.float32): _rs.fht_f32,
    np.dtype(np.float64): _rs.fht_f64,
}

_OOP_DISPATCH = {
    np.dtype(np.float32): _rs.fht_f32_oop,
    np.dtype(np.float64): _rs.fht_f64_oop,
}


def ffht(
    arr: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    inplace: bool = False,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64] | None:
    """Fast Hadamard Transform of a 1-D ``float32`` or ``float64`` array.

    Dispatches to the appropriate Rust implementation based on
    ``arr.dtype``, and to the in-place or out-of-place variant based on
    ``inplace``.

    Args:
        arr: 1-D ``float32`` or ``float64`` array. ``arr.shape[0]`` must
            be a power of two and ``arr`` must be C-contiguous.
        inplace: If ``True``, transform ``arr`` in place and return
            ``None``. If ``False`` (default), leave ``arr`` unchanged
            and return a new array containing the transform.

    Returns:
        A new array containing the transform if ``inplace`` is
        ``False``; otherwise ``None``.

    Raises:
        TypeError: If ``arr.dtype`` is not ``float32`` or ``float64``.
        ValueError: If ``arr`` is not C-contiguous or its length is not
            a power of two.
    """
    dispatch = _INPLACE_DISPATCH if inplace else _OOP_DISPATCH

    try:
        fn = dispatch[arr.dtype]
    except KeyError:
        raise TypeError(
            f"fht: unsupported dtype {arr.dtype!r}; expected float32 or float64"
        ) from None

    return fn(arr)