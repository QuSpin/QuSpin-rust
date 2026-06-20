"""Tests for the high-level ``quspin_rs.ffht.ffht`` dispatcher.

Run with: pytest test_ffht.py
"""

import numpy as np
import pytest

from quspin_rs.ffht import ffht

EPS_F32 = 1e-5
EPS_F64 = 1e-12


# ---------------------------------------------------------------------
# Out-of-place (default): returns a new array, input unchanged.
# ---------------------------------------------------------------------


def test_fht_oop_f32_returns_new_array():
    x = np.arange(1, 9, dtype=np.float32)
    x_copy = x.copy()

    y = ffht(x)

    assert y is not None
    assert y.dtype == np.float32
    np.testing.assert_array_equal(x, x_copy)  # input unchanged

    expected = np.array([36.0, -4.0, -8.0, 0.0, -16.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(y, expected, atol=EPS_F32)


def test_fht_oop_f64_returns_new_array():
    x = np.arange(1, 5, dtype=np.float64)
    x_copy = x.copy()

    y = ffht(x)

    assert y is not None
    assert y.dtype == np.float64
    np.testing.assert_array_equal(x, x_copy)  # input unchanged

    expected = np.array([10.0, -2.0, -4.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(y, expected, atol=EPS_F64)


# ---------------------------------------------------------------------
# In-place: mutates input, returns None.
# ---------------------------------------------------------------------


def test_fht_inplace_f32_mutates_and_returns_none():
    x = np.arange(1, 9, dtype=np.float32)

    result = ffht(x, inplace=True)

    assert result is None
    expected = np.array([36.0, -4.0, -8.0, 0.0, -16.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(x, expected, atol=EPS_F32)


def test_fht_inplace_f64_mutates_and_returns_none():
    x = np.arange(1, 5, dtype=np.float64)

    result = ffht(x, inplace=True)

    assert result is None
    expected = np.array([10.0, -2.0, -4.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(x, expected, atol=EPS_F64)


# ---------------------------------------------------------------------
# Inplace and out-of-place agree.
# ---------------------------------------------------------------------


def test_fht_inplace_matches_oop_f32():
    x = np.arange(1, 9, dtype=np.float32)

    oop_result = ffht(x)
    assert oop_result is not None

    x_for_inplace = np.arange(1, 9, dtype=np.float32)
    ffht(x_for_inplace, inplace=True)

    np.testing.assert_allclose(oop_result, x_for_inplace, atol=EPS_F32)


# ---------------------------------------------------------------------
# Dtype dispatch: unsupported dtype raises TypeError.
# ---------------------------------------------------------------------


def test_fht_unsupported_dtype_raises_type_error():
    x = np.arange(8, dtype=np.int32)
    with pytest.raises(TypeError, match="unsupported dtype"):
        ffht(x)  # pyright: ignore[reportArgumentType]


def test_fht_unsupported_dtype_raises_type_error_inplace():
    x = np.arange(8, dtype=np.complex128)
    with pytest.raises(TypeError, match="unsupported dtype"):
        ffht(x, inplace=True)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------
# Validation errors (contiguity / power-of-two) pass through from _rs.
# ---------------------------------------------------------------------


def test_fht_non_power_of_two_raises_value_error():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="power of two"):
        ffht(x)


def test_fht_non_contiguous_raises_value_error():
    base = np.arange(16, dtype=np.float32)
    view = base[::2]
    assert not view.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="C-contiguous"):
        ffht(view)


# ---------------------------------------------------------------------
# Larger size sanity check, both dtypes, both modes.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype,eps", [(np.float32, 1e-2), (np.float64, 1e-9)])
@pytest.mark.parametrize("inplace", [False, True])
def test_fht_large_n_involution(dtype, eps, inplace):
    n = 1024
    original = ((np.arange(n) % 7) - 3).astype(dtype)

    if inplace:
        buf = original.copy()
        ffht(buf, inplace=True)
        ffht(buf, inplace=True)
        result = buf
    else:
        once = ffht(original)
        assert once is not None
        result = ffht(once)
        assert result is not None

    expected = original * n
    np.testing.assert_allclose(result, expected, rtol=eps, atol=eps)
