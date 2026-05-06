"""Tests for QMatrix/Hamiltonian.as_linearoperator + ExpmOp / ExpmWorker."""

import cmath
import math

import numpy as np
import pytest

from quspin_rs._rs import (
    ExpmOp,
    ExpmWorker,
    ExpmWorker2,
    Hamiltonian,
    PauliOperator,
    QMatrix,
    QMatrixLinearOperator,
    SpinBasis,
    Static,
)


class TestExpmOpAndWorker:
    """Tests for the new linear-operator + cached expm path."""

    def _diagonal_ham(self) -> Hamiltonian:
        """H = diag(1, -1) via Z on a 1-site spin-1/2 basis."""
        op = PauliOperator([("Z", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        return Hamiltonian(mat, [Static()])

    def _xx_ham(self) -> Hamiltonian:
        """H = X_0 X_1 on 2 spin-1/2 sites."""
        op = PauliOperator([("XX", [[1.0, 0, 1]])])
        basis = SpinBasis.full(2)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        return Hamiltonian(mat, [Static()])

    @staticmethod
    def _ref_expm_z(a: complex, f: np.ndarray) -> np.ndarray:
        scale = np.array([cmath.exp(a), cmath.exp(-a)], dtype=np.complex128)
        if f.ndim == 1:
            return scale * f
        return scale[:, None] * f

    # -- as_linearoperator ----------------------------------------------------

    def test_qmatrix_as_linearoperator_returns_correct_type(self):
        op = PauliOperator([("Z", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        coeffs = np.array([1.0 + 0j], dtype=np.complex128)
        qop = mat.as_linearoperator(coeffs)
        assert isinstance(qop, QMatrixLinearOperator)
        assert qop.dim == 2
        assert qop.num_coeff == 1

    def test_qmatrix_as_linearoperator_wrong_coeffs_size(self):
        op = PauliOperator([("Z", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        bad = np.zeros(5, dtype=np.complex128)
        with pytest.raises(Exception):
            mat.as_linearoperator(bad)

    def test_hamiltonian_as_linearoperator_evaluates_at_time(self):
        """Time-dependent coefficient: H(t) = sin(t) * Z."""

        def f(t: float) -> complex:
            return complex(np.sin(t))

        op = PauliOperator([("Z", [[1.0, 0]])])
        basis = SpinBasis.full(1)
        mat = QMatrix.build_pauli(op, basis, np.dtype("float64"))
        ham = Hamiltonian(mat, [f])
        qop_t0 = ham.as_linearoperator(0.0)
        qop_t1 = ham.as_linearoperator(np.pi / 2)
        assert qop_t0.dim == 2 == qop_t1.dim
        # The coefficient is internal — verify indirectly through ExpmOp:
        expm_op_t1 = ExpmOp(qop_t1, complex(-0.5, 0.0))
        worker = expm_op_t1.worker()
        psi = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        worker.apply(psi)
        # H(π/2) = Z, exp(-0.5·Z) on |0⟩ = exp(-0.5)·|0⟩
        np.testing.assert_allclose(psi[0], cmath.exp(-0.5), atol=1e-10)
        np.testing.assert_allclose(psi[1], 0.0, atol=1e-10)

    # -- ExpmOp construction --------------------------------------------------

    def test_expm_op_dim_and_a(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -1.0))
        assert eo.dim == 2
        assert eo.a == complex(0.0, -1.0)

    # -- worker(n_vec) dispatches to the right type ---------------------------

    def test_worker_n_vec_0_returns_1d_worker(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -math.pi / 4))
        w = eo.worker(0)
        assert isinstance(w, ExpmWorker)

    def test_worker_default_n_vec_returns_1d_worker(self):
        """Default n_vec is 0 → 1-D ExpmWorker."""
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -math.pi / 4))
        assert isinstance(eo.worker(), ExpmWorker)

    def test_worker_n_vec_1_returns_2d_worker_capacity_1(self):
        """n_vec=1 unambiguously means 2-D batch of capacity 1."""
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -math.pi / 4))
        w = eo.worker(1)
        assert isinstance(w, ExpmWorker2)
        assert w.n_vec == 1

    def test_worker_n_vec_gt1_returns_2d_worker(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        eo = ExpmOp(qop, complex(0.0, -math.pi / 4))
        w = eo.worker(3)
        assert isinstance(w, ExpmWorker2)
        assert w.n_vec == 3

    # -- 1-D apply correctness ------------------------------------------------

    def test_worker_apply_diagonal_matches_analytic(self):
        ham = self._diagonal_ham()
        qop = ham.as_linearoperator(0.0)
        a = -1j * math.pi / 4
        eo = ExpmOp(qop, a)
        f = np.array([1.0 + 0j, 1.0 + 0j], dtype=np.complex128) / math.sqrt(2)
        expected = self._ref_expm_z(a, f.copy())
        eo.worker().apply(f)
        np.testing.assert_allclose(f, expected, atol=1e-10)

    def test_worker_apply_xx_matches_eigendecomposition(self):
        """Cross-check: ExpmOp.worker().apply == exp(a·H) via eigendecomposition."""
        ham = self._xx_ham()
        a = -1j * math.pi / 3
        H = ham.to_dense(0.0)
        # H is Hermitian; reconstruct exp(a·H) from its eigendecomposition.
        vals, vecs = np.linalg.eigh(H)
        ref_expm = (vecs * np.exp(a * vals)) @ vecs.conj().T

        rng = np.random.default_rng(7)
        f = (rng.standard_normal(4) + 1j * rng.standard_normal(4)).astype(np.complex128)
        expected = ref_expm @ f

        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        eo.worker().apply(f)

        np.testing.assert_allclose(f, expected, atol=1e-10)

    def test_worker_reuse_same_result(self):
        """Calling apply twice with a reset input should give the same answer."""
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        worker = eo.worker()
        f = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        worker.apply(f)
        first = f.copy()
        f[:] = [1.0 + 0j, 0.0 + 0j]
        worker.apply(f)
        np.testing.assert_allclose(f, first, atol=1e-12)

    def test_worker_apply_wrong_shape_raises(self):
        ham = self._diagonal_ham()
        eo = ExpmOp(ham.as_linearoperator(0.0), -1j)
        worker = eo.worker()
        f = np.zeros(5, dtype=np.complex128)
        with pytest.raises(Exception):
            worker.apply(f)

    # -- 2-D apply correctness ------------------------------------------------

    def test_worker2_apply_diagonal_matches_analytic(self):
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        rng = np.random.default_rng(42)
        F = (rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))).astype(
            np.complex128
        )
        expected = self._ref_expm_z(a, F.copy())
        eo.worker(3).apply(F)
        np.testing.assert_allclose(F, expected, atol=1e-10)

    def test_worker2_apply_with_fewer_columns_than_capacity(self):
        """A worker with n_vec=4 should accept input with k=2 columns."""
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        rng = np.random.default_rng(0)
        F = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))).astype(
            np.complex128
        )
        expected = self._ref_expm_z(a, F.copy())
        eo.worker(4).apply(F)
        np.testing.assert_allclose(F, expected, atol=1e-10)

    def test_worker2_apply_wrong_dim_raises(self):
        ham = self._diagonal_ham()
        eo = ExpmOp(ham.as_linearoperator(0.0), -1j)
        worker = eo.worker(2)
        F = np.zeros((5, 2), dtype=np.complex128)
        with pytest.raises(Exception):
            worker.apply(F)

    def test_worker2_too_many_columns_raises(self):
        ham = self._diagonal_ham()
        eo = ExpmOp(ham.as_linearoperator(0.0), -1j)
        worker = eo.worker(2)  # capacity = 2
        F = np.zeros((2, 3), dtype=np.complex128)
        with pytest.raises(Exception):
            worker.apply(F)

    # -- user-supplied work buffer --------------------------------------------

    def test_worker_with_user_work_1d(self):
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        work = np.zeros(2 * eo.dim, dtype=np.complex128)  # length 4
        worker = eo.worker(0, work=work)
        f = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        worker.apply(f)
        # Ground truth via plain worker
        f_ref = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        eo.worker().apply(f_ref)
        np.testing.assert_allclose(f, f_ref, atol=1e-12)

    def test_worker_with_user_work_2d(self):
        ham = self._diagonal_ham()
        a = -1j * math.pi / 4
        eo = ExpmOp(ham.as_linearoperator(0.0), a)
        work = np.zeros((2 * eo.dim, 3), dtype=np.complex128)
        worker = eo.worker(3, work=work)
        F = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.complex128)
        F_ref = F.copy()
        worker.apply(F)
        eo.worker(3).apply(F_ref)
        np.testing.assert_allclose(F, F_ref, atol=1e-12)

    def test_worker_user_work_too_small_raises(self):
        ham = self._diagonal_ham()
        eo = ExpmOp(ham.as_linearoperator(0.0), -1j)
        work = np.zeros(1, dtype=np.complex128)  # too small (need 4)
        with pytest.raises(Exception):
            eo.worker(1, work=work)
