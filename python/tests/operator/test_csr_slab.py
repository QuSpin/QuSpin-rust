"""Tests for PauliOperator.csr_slab (petsc4py-compatible row-range CSR)."""

import numpy as np
import pytest
import scipy.sparse

from quspin_rs._rs import (
    BosonBasis,
    FermionBasis,
    PauliOperator,
    QMatrix,
    SpinBasis,
)


N = 4


def make_xx_zz() -> PauliOperator:
    """4-site XX + ZZ chain — 2 cindices."""
    bonds = [[1.0, i, i + 1] for i in range(N - 1)]
    return PauliOperator([("XX", bonds)], [("ZZ", bonds)])


def make_full_spin_basis() -> SpinBasis:
    return SpinBasis.full(N)


def make_full_fermion_basis() -> FermionBasis:
    return FermionBasis.full(N)


def reference_csr(op, basis, coeffs):
    """Build the full QMatrix and materialise its CSR for comparison."""
    mat = QMatrix.build_pauli(op, basis, np.dtype("complex128"))
    indptr, indices, data = mat.to_csr(coeffs)
    return scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(mat.dim, mat.dim)
    )


class TestCsrSlabFullRange:
    def test_full_range_matches_to_csr_spin(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)

        ref = reference_csr(op, basis, coeffs)
        indptr, indices, data = op.csr_slab(
            basis, coeffs, 0, basis.size, dtype=np.dtype("complex128")
        )
        slab = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(basis.size, basis.size)
        )
        np.testing.assert_array_equal((ref - slab).nnz, 0)
        np.testing.assert_allclose(ref.toarray(), slab.toarray(), atol=1e-12)

    def test_full_range_matches_to_csr_fermion(self):
        op = make_xx_zz()
        basis = make_full_fermion_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)

        ref = reference_csr(op, basis, coeffs)
        indptr, indices, data = op.csr_slab(
            basis, coeffs, 0, basis.size, dtype=np.dtype("complex128")
        )
        slab = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(basis.size, basis.size)
        )
        np.testing.assert_allclose(ref.toarray(), slab.toarray(), atol=1e-12)


class TestCsrSlabPartition:
    def test_partition_round_trip(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        dim = basis.size
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        ref = reference_csr(op, basis, coeffs)

        for k in (1, 2, 3, dim):
            bounds = [i * dim // k for i in range(k + 1)]
            indptr_concat = np.array([0], dtype=np.int64)
            indices_concat = np.zeros(0, dtype=np.int64)
            data_concat = np.zeros(0, dtype=np.complex128)
            for rs, re in zip(bounds[:-1], bounds[1:]):
                ip, ii, dd = op.csr_slab(
                    basis, coeffs, rs, re, dtype=np.dtype("complex128")
                )
                cum_nnz = int(indptr_concat[-1])
                indptr_concat = np.concatenate([indptr_concat, ip[1:] + cum_nnz])
                indices_concat = np.concatenate([indices_concat, ii])
                data_concat = np.concatenate([data_concat, dd])
            slab = scipy.sparse.csr_matrix(
                (data_concat, indices_concat, indptr_concat), shape=(dim, dim)
            )
            np.testing.assert_allclose(
                ref.toarray(), slab.toarray(), atol=1e-12, err_msg=f"k={k}"
            )


class TestCsrSlabEmpty:
    def test_empty_slab_returns_empty_arrays(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)

        for r in (0, 5, basis.size):
            ip, ii, dd = op.csr_slab(
                basis, coeffs, r, r, dtype=np.dtype("complex128")
            )
            assert ip.dtype == np.int64
            assert ii.dtype == np.int64
            assert dd.dtype == np.complex128
            np.testing.assert_array_equal(ip, np.array([0], dtype=np.int64))
            assert ii.size == 0
            assert dd.size == 0


class TestCsrSlabDtypes:
    def test_indptr_indices_int64(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        ip, ii, dd = op.csr_slab(
            basis, coeffs, 0, basis.size, dtype=np.dtype("complex128")
        )
        assert ip.dtype == np.int64
        assert ii.dtype == np.int64
        assert dd.dtype == np.complex128


class TestCsrSlabValidation:
    def test_basis_type_mismatch_raises(self):
        op = make_xx_zz()
        basis = BosonBasis.full(2, lhss=2)  # wrong type for PauliOperator
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        with pytest.raises((TypeError, ValueError)):
            op.csr_slab(basis, coeffs, 0, basis.size, dtype=np.dtype("complex128"))

    def test_wrong_coeffs_size_raises(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j, 0.0 + 0j], dtype=np.complex128)
        with pytest.raises(ValueError):
            op.csr_slab(basis, coeffs, 0, basis.size, dtype=np.dtype("complex128"))

    def test_invalid_row_range_raises(self):
        op = make_xx_zz()
        basis = make_full_spin_basis()
        coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
        # row_start > row_end
        with pytest.raises(ValueError):
            op.csr_slab(basis, coeffs, 10, 5, dtype=np.dtype("complex128"))
        # row_end > basis.size
        with pytest.raises(ValueError):
            op.csr_slab(
                basis, coeffs, 0, basis.size + 1, dtype=np.dtype("complex128")
            )
