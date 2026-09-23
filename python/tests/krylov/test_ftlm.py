"""Tests for FTLM / LTLM / FTLMDynamic, focused on the Boltzmann energy shift.

``e^{-beta E_n}`` on raw Ritz values overflows to ``inf`` once
``beta * |E_min|`` exceeds ~709, which for any Hamiltonian with a negative
ground state is an ordinary low-temperature regime.  ``e_shift`` subtracts a
reference energy before exponentiating; it cancels in ``sum(oz)/sum(z)``
provided every sample uses the same value, which is why it lives on the
constructor rather than on ``sample()``.
"""

import numpy as np
import pytest

from quspin_rs._rs import (
    FTLM,
    LTLM,
    FTLMDynamic,
    Hamiltonian,
    PauliOperator,
    QMatrix,
    SpinBasis,
    Static,
)

N = 3
DIM = 2**N


def _ham_and_obs(scale: float = 1.0, heisenberg: bool = False):
    """Heisenberg-ish chain H and an observable O, plus their dense forms.

    ``scale`` multiplies H so the spectrum can be pushed far enough from zero
    to overflow the unshifted Boltzmann weight.  ``heisenberg`` adds the YY
    term, making H conserve Sz so the fully polarized states are eigenstates.
    """
    bonds = [[scale, i, i + 1] for i in range(N - 1)]
    terms = ["XX", "YY", "ZZ"] if heisenberg else ["XX", "ZZ"]
    h_op = PauliOperator(*[[(t, bonds)] for t in terms])
    # ZZ, not Z: XX+ZZ is invariant under the global spin flip X^(x)N, under
    # which a single Z is odd, so <Z_0> vanishes identically at every beta and
    # any multiplicative error in oz_r would be invisible.  ZZ is even, and
    # <ZZ_01> = -0.687... here.
    o_op = PauliOperator([("ZZ", [[1.0, 0, 1]])])
    basis = SpinBasis.full(N)

    h_mat = QMatrix.build_pauli(h_op, basis, np.dtype("complex128"))
    o_mat = QMatrix.build_pauli(o_op, basis, np.dtype("complex128"))
    ham = Hamiltonian(h_mat, [Static() for _ in terms])
    obs = Hamiltonian(o_mat, [Static()])

    # `QMatrix` stores the transpose (#121), but XX+ZZ and Z are both
    # real-symmetric, so `to_dense` returns the same matrix either way.
    h_dense = ham.to_dense(0.0)
    o_dense = obs.to_dense(0.0)
    assert np.max(np.abs(h_dense - h_dense.T)) < 1e-12
    return ham, obs, h_dense, o_dense


def _exact_thermal_average(h_dense, o_dense, beta: float) -> complex:
    """Tr(O e^{-beta H}) / Tr(e^{-beta H}) by dense diagonalization."""
    evals, evecs = np.linalg.eigh(h_dense)
    weights = np.exp(-beta * (evals - evals.min()))
    rho = evecs @ np.diag(weights) @ evecs.conj().T
    return np.trace(o_dense @ rho) / np.trace(rho)


def _ftlm_average(estimator, obs, beta: float, k: int) -> complex:
    """Full-trace FTLM: sum over a complete basis of starting vectors.

    Summing over all `dim` unit vectors makes FTLM exact — it is just the
    trace written out — so this compares against dense diagonalization with no
    stochastic tolerance.

    Note the reason it is exact is *not* that ``k = dim`` spans the whole
    space: Lanczos from a computational-basis vector on a symmetric chain
    terminates early (``LanczosBasis::build`` breaks when beta_j underflows).
    It is exact because the space it terminates on is H-invariant and contains
    |r>, so ``e^{-beta H}|r>`` never leaves it and the Ritz pairs are exact
    eigenpairs there.  That is what licenses the tight ``abs=1e-8`` below.
    """
    z_total = 0.0
    oz_total = 0.0 + 0.0j
    for i in range(DIM):
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[i] = 1.0
        z_r, oz_r = estimator.sample(v0, k, obs, beta)
        z_total += z_r
        oz_total += oz_r
    return oz_total / z_total


class TestEShiftSurface:
    def test_default_is_zero(self):
        ham, _, _, _ = _ham_and_obs()
        assert FTLM(ham).e_shift == 0.0
        assert LTLM(ham).e_shift == 0.0
        assert FTLMDynamic(ham).e_shift == 0.0

    def test_roundtrips(self):
        ham, _, _, _ = _ham_and_obs()
        assert FTLM(ham, e_shift=-3.5).e_shift == -3.5
        assert LTLM(ham, -3.5).e_shift == -3.5

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("cls", [FTLM, LTLM, FTLMDynamic])
    def test_rejects_non_finite(self, cls, bad: float):
        ham, _, _, _ = _ham_and_obs()
        with pytest.raises(ValueError, match="finite"):
            cls(ham, e_shift=bad)


class TestEShiftCorrectness:
    @pytest.mark.parametrize("beta", [0.0, 0.5, 2.0])
    def test_full_trace_matches_exact_diagonalization(self, beta: float):
        """With a complete set of start vectors, FTLM is exact."""
        ham, obs, h_dense, o_dense = _ham_and_obs()
        got = _ftlm_average(
            FTLM(ham, e_shift=h_dense.diagonal().real.min()), obs, beta, DIM
        )
        want = _exact_thermal_average(h_dense, o_dense, beta)
        assert got == pytest.approx(want, abs=1e-8)

    @pytest.mark.parametrize("shift", [-4.0, -1.0, 0.0, 2.5])
    def test_ratio_is_invariant_under_the_shift(self, shift: float):
        """Different shifts must give the same thermal average."""
        ham, obs, h_dense, o_dense = _ham_and_obs()
        beta = 1.5
        got = _ftlm_average(FTLM(ham, e_shift=shift), obs, beta, DIM)
        want = _exact_thermal_average(h_dense, o_dense, beta)
        assert got == pytest.approx(want, abs=1e-8)

    @pytest.mark.parametrize("shift", [-4.0, 0.0, 2.5])
    def test_ltlm_full_trace_matches_exact_diagonalization(self, shift: float):
        """Same identity for LTLM, and it pins the half-exponent bookkeeping.

        LTLM's oz_r = <phi|O|phi> with phi = e^{-beta(H-s)/2}|r>, so it picks up
        exp(beta*s) from *two* half-exponent factors while z_r picks it up from
        one full one.  Those powers have to match for the ratio to be
        shift-invariant, and a test that only checks finiteness cannot see it.
        Summing over a complete basis gives
        Tr(e^{-bH/2} O e^{-bH/2}) / Tr(e^{-bH}) = <O> exactly.
        """
        ham, obs, h_dense, o_dense = _ham_and_obs()
        beta = 1.5
        got = _ftlm_average(LTLM(ham, e_shift=shift), obs, beta, DIM)
        want = _exact_thermal_average(h_dense, o_dense, beta)
        assert got == pytest.approx(want, abs=1e-8)


class TestOverflow:
    """The regression: low temperature used to return inf/NaN silently."""

    # scale=30 puts the ground state near -85; beta=25 makes beta*|E_0| >> 709.
    SCALE = 30.0
    BETA = 25.0

    def test_unshifted_raises_instead_of_returning_nan(self):
        ham, obs, _, _ = _ham_and_obs(self.SCALE)
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[0] = 1.0

        with pytest.raises(ValueError, match="e_shift"):
            FTLM(ham).sample(v0, DIM, obs, self.BETA)

    def test_shifted_is_finite_and_correct(self):
        ham, obs, h_dense, o_dense = _ham_and_obs(self.SCALE)
        e0 = np.linalg.eigvalsh(h_dense).min()

        got = _ftlm_average(FTLM(ham, e_shift=e0), obs, self.BETA, DIM)
        want = _exact_thermal_average(h_dense, o_dense, self.BETA)

        assert np.isfinite(got)
        assert got == pytest.approx(want, abs=1e-8)

    def test_ltlm_unshifted_raises(self):
        ham, obs, _, _ = _ham_and_obs(self.SCALE)
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[0] = 1.0

        with pytest.raises(ValueError, match="e_shift"):
            LTLM(ham).sample(v0, DIM, obs, self.BETA)

    def test_ltlm_shifted_is_finite(self):
        ham, obs, h_dense, _ = _ham_and_obs(self.SCALE)
        e0 = np.linalg.eigvalsh(h_dense).min()
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[0] = 1.0

        z_r, oz_r = LTLM(ham, e_shift=e0).sample(v0, DIM, obs, self.BETA)
        assert np.isfinite(z_r)
        assert np.isfinite(oz_r)

    def test_ftlm_dynamic_unshifted_raises(self):
        """The dynamic estimator overflows the same way the other two do."""
        ham, obs, _, _ = _ham_and_obs(self.SCALE)
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[0] = 1.0

        with pytest.raises(ValueError, match="e_shift"):
            FTLMDynamic(ham).sample(
                v0, DIM, obs, self.BETA, np.linspace(-2.0, 2.0, 5), 0.1
            )

    def test_ftlm_dynamic_shifted_is_finite(self):
        ham, obs, h_dense, _ = _ham_and_obs(self.SCALE)
        e0 = np.linalg.eigvalsh(h_dense).min()
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[0] = 1.0

        s = FTLMDynamic(ham, e_shift=e0).sample(
            v0, DIM, obs, self.BETA, np.linspace(-2.0, 2.0, 5), 0.1
        )
        assert np.all(np.isfinite(s))

    def test_overflow_message_says_lower_not_raise(self):
        """The recovery direction must be right; it is the actionable half.

        Overflow means e_shift sits too far ABOVE the spectrum. With the
        default e_shift=0 and E_0 < 0 the fix is to move e_shift DOWN to E_0.
        """
        ham, obs, _, _ = _ham_and_obs(self.SCALE)
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[0] = 1.0

        with pytest.raises(ValueError, match="lower it towards") as exc:
            FTLM(ham).sample(v0, DIM, obs, self.BETA)
        assert "raise" not in str(exc.value)

    @staticmethod
    def _polarized_index(h_dense) -> int:
        """Index of a fully polarized basis state, the top of the spectrum."""
        i = int(np.argmax(np.real(np.diag(h_dense))))
        off_diag = np.delete(h_dense[:, i], i)
        assert np.max(np.abs(off_diag)) < 1e-12, "must be an eigenstate"
        return i

    @pytest.mark.parametrize("cls", [FTLM, LTLM])
    def test_negligible_sample_is_zero_not_an_error(self, cls):
        """A single z_r underflowing to 0 is the right answer, not a failure.

        With e_shift = E_0 exactly, the polarized start vector is an eigenstate
        ~180 above it, so its weight exp(-25*180) is 0.0 in f64.  It must add
        zero to the sum rather than abort the average.
        """
        ham, obs, h_dense, _ = _ham_and_obs(self.SCALE, heisenberg=True)
        e0 = np.linalg.eigvalsh(h_dense).min()
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[self._polarized_index(h_dense)] = 1.0

        z_r, oz_r = cls(ham, e_shift=e0).sample(v0, DIM, obs, self.BETA)
        assert z_r == 0.0
        assert oz_r == 0.0

    @pytest.mark.parametrize("cls", [FTLM, LTLM])
    def test_full_trace_with_negligible_samples_matches_exact(self, cls):
        """The full-trace average survives samples whose weights underflow."""
        ham, obs, h_dense, o_dense = _ham_and_obs(self.SCALE, heisenberg=True)
        e0 = np.linalg.eigvalsh(h_dense).min()

        got = _ftlm_average(cls(ham, e_shift=e0), obs, self.BETA, DIM)
        want = _exact_thermal_average(h_dense, o_dense, self.BETA)

        assert np.isfinite(got)
        assert got == pytest.approx(want, abs=1e-8)

    def test_ftlm_dynamic_negligible_sample_is_zero_not_an_error(self):
        ham, obs, h_dense, _ = _ham_and_obs(self.SCALE, heisenberg=True)
        e0 = np.linalg.eigvalsh(h_dense).min()
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[self._polarized_index(h_dense)] = 1.0

        s = FTLMDynamic(ham, e_shift=e0).sample(
            v0, DIM, obs, self.BETA, np.linspace(-2.0, 2.0, 5), 0.1
        )
        assert np.all(s == 0.0)

    def test_non_finite_observable_contribution_raises(self):
        """`oz_r` can blow up while `z_r` stays finite — the guard is two-sided.

        E_0 = -2.83 here, so beta=8 leaves z_r = 1.4e8: comfortably finite, and
        independent of the observable.  A norm-1e305 observable then pushes
        oz_r past f64 on its own.
        """
        ham, obs, _, _ = _ham_and_obs()
        huge = Hamiltonian(
            QMatrix.build_pauli(
                PauliOperator([("ZZ", [[1e305, 0, 1]])]),
                SpinBasis.full(N),
                np.dtype("complex128"),
            ),
            [Static()],
        )
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[0] = 1.0

        z_r, _ = FTLM(ham).sample(v0, DIM, obs, 8.0)
        assert np.isfinite(z_r), "z_r must stay finite for this test to bite"

        with pytest.raises(ValueError, match="not finite"):
            FTLM(ham).sample(v0, DIM, huge, 8.0)


class TestFTLMDynamic:
    def test_shift_leaves_the_frequency_axis_alone(self):
        """e_shift rescales weights; it must not move the spectral peaks."""
        ham, obs, h_dense, _ = _ham_and_obs()
        omegas = np.linspace(-4.0, 4.0, 65)
        v0 = np.zeros(DIM, dtype=np.complex128)
        v0[0] = 1.0
        beta, eta = 1.0, 0.1

        plain = FTLMDynamic(ham).sample(v0, DIM, obs, beta, omegas, eta)
        shift = -2.0
        shifted = FTLMDynamic(ham, e_shift=shift).sample(
            v0, DIM, obs, beta, omegas, eta
        )

        # Uniform rescaling by e^{beta*shift}, same shape.
        assert np.max(np.abs(shifted - plain * np.exp(beta * shift))) < 1e-10
