use crate::qmatrix::CIndex;
use num_complex::Complex;
use quspin_basis::expand::ExpandRefState;
use quspin_basis::space::{FullSpace, Subspace};
use quspin_basis::sym::{NormInt, SymBasis};
use quspin_basis::traits::BasisSpace;
use quspin_bitbasis::{BitInt, FermionicBitStateOp};
use quspin_operator::Operator;
use quspin_types::QuSpinError;
use rayon::prelude::*;

type C64 = Complex<f64>;

/// Minimum input-space size before switching to parallel application.
const PARALLEL_APPLY_THRESHOLD: usize = 256;

// ---------------------------------------------------------------------------
// ProjectState — zero-cost abstraction over output projection
// ---------------------------------------------------------------------------

/// Project a full-space state into a basis, returning the index and a
/// complex scaling factor.
///
/// For non-symmetric bases the scale is always `1.0`.  For `SymBasis` the
/// state is mapped to its orbit representative and scaled by
/// `conj(χ_g) · √(norm_j / |G|)`.  Fully monomorphized — zero runtime
/// dispatch.
pub trait ProjectState<B: BitInt>: BasisSpace<B> {
    fn project(&self, state: B) -> Option<(usize, C64)>;
}

impl<B: BitInt> ProjectState<B> for FullSpace<B> {
    fn project(&self, state: B) -> Option<(usize, C64)> {
        self.index(state).map(|j| (j, C64::new(1.0, 0.0)))
    }
}

impl<B: BitInt> ProjectState<B> for Subspace<B> {
    fn project(&self, state: B) -> Option<(usize, C64)> {
        self.index(state).map(|j| (j, C64::new(1.0, 0.0)))
    }
}

impl<B: BitInt, L: FermionicBitStateOp<B>, N: NormInt> ProjectState<B> for SymBasis<B, L, N> {
    fn project(&self, state: B) -> Option<(usize, C64)> {
        let (rep, grp_char) = self.get_refstate(state);
        self.index(rep).map(|j| {
            let (_, norm) = self.entry(j);
            let group_order = self.group_order() as f64;
            // Projection scale: `conj(χ) · √(norm_j / |G|)`. Paired with
            // the `1/√(|G|·norm_i)` factor in expansion, a round-trip
            // through expand→apply→project reproduces the matrix element
            // computed by `build_from_symmetric` (`χ · √(norm_j/norm_i)`).
            (
                j,
                grp_char.conj() * C64::new((norm / group_order).sqrt(), 0.0),
            )
        })
    }
}

// ---------------------------------------------------------------------------
// apply_and_project_to
// ---------------------------------------------------------------------------

/// Apply an operator to a vector in `input_space` and project the result
/// into `output_space`.
///
/// Iterates over input basis states via [`ExpandRefState::expand_ref_state_iter`],
/// applies the operator to each expanded state, and accumulates into `output`
/// via [`ProjectState::project`].  No intermediate allocation.
///
/// This is a matrix-free operation — no `QMatrix` is built.
///
/// # Arguments
/// - `op` — operator to apply (e.g. `HardcoreOperator`, `BondOperator`)
/// - `input_space` — basis the input vector lives in
/// - `output_space` — basis to project the result into
/// - `coeffs` — per-cindex scaling factors (length `op.num_cindices()`)
/// - `input` — input vector (length `input_space.size()`)
/// - `output` — output vector (length `output_space.size()`)
/// - `overwrite` — if true, zero `output` before accumulating
pub fn apply_and_project_to_inner<H, B, C, IS, OS>(
    op: &H,
    input_space: &IS,
    output_space: &OS,
    coeffs: &[C64],
    input: &[C64],
    output: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError>
where
    H: Operator<C> + Sync,
    B: BitInt,
    C: CIndex,
    IS: BasisSpace<B> + ExpandRefState<B, C64, C64> + Sync,
    OS: ProjectState<B> + Sync,
{
    validate_args(
        op.num_cindices(),
        coeffs.len(),
        input_space.n_sites(),
        input_space.lhss(),
        input_space.size(),
        input.len(),
        output_space.n_sites(),
        output_space.lhss(),
        output_space.size(),
        output.len(),
    )?;

    if overwrite {
        output.iter_mut().for_each(|c| *c = C64::default());
    }

    if input.len() < PARALLEL_APPLY_THRESHOLD {
        for (i, coeff) in input.iter().enumerate() {
            for (state, amp) in input_space.expand_ref_state_iter(i, coeff) {
                if amp.norm_sqr() == 0.0 {
                    continue;
                }
                op.apply(state, |cindex, op_amp, new_state| {
                    if let Some((j, scale)) = output_space.project(new_state) {
                        output[j] += coeffs[cindex.as_usize()] * op_amp * amp * scale;
                    }
                });
            }
        }
    } else {
        let output_size = output_space.size();
        let combined: Vec<C64> = input
            .par_iter()
            .enumerate()
            .fold(
                || vec![C64::default(); output_size],
                |mut local_out, (i, coeff)| {
                    for (state, amp) in input_space.expand_ref_state_iter(i, coeff) {
                        if amp.norm_sqr() == 0.0 {
                            continue;
                        }
                        op.apply(state, |cindex, op_amp, new_state| {
                            if let Some((j, scale)) = output_space.project(new_state) {
                                local_out[j] += coeffs[cindex.as_usize()] * op_amp * amp * scale;
                            }
                        });
                    }
                    local_out
                },
            )
            .reduce(
                || vec![C64::default(); output_size],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += y;
                    }
                    a
                },
            );
        for (o, c) in output.iter_mut().zip(combined.iter()) {
            *o += c;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// project_to — identity operator (expand + project, no operator)
// ---------------------------------------------------------------------------

/// Project a vector from `input_space` into `output_space` without applying
/// any operator (identity projection).
///
/// Expands the input vector via [`ExpandRefState`] and projects each state
/// into `output_space` via [`ProjectState`].  Equivalent to
/// `apply_and_project_to` with the identity operator.
pub fn project_to_inner<B, IS, OS>(
    input_space: &IS,
    output_space: &OS,
    input: &[C64],
    output: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError>
where
    B: BitInt,
    IS: BasisSpace<B> + ExpandRefState<B, C64, C64> + Sync,
    OS: ProjectState<B> + Sync,
{
    if input_space.n_sites() != output_space.n_sites() {
        return Err(QuSpinError::ValueError(format!(
            "input basis has n_sites={} but output basis has n_sites={}",
            input_space.n_sites(),
            output_space.n_sites(),
        )));
    }
    if input_space.lhss() != output_space.lhss() {
        return Err(QuSpinError::ValueError(format!(
            "input basis has lhss={} but output basis has lhss={}",
            input_space.lhss(),
            output_space.lhss(),
        )));
    }
    if input.len() != input_space.size() {
        return Err(QuSpinError::ValueError(format!(
            "input.len() = {} but input_space.size() = {}",
            input.len(),
            input_space.size(),
        )));
    }
    if output.len() != output_space.size() {
        return Err(QuSpinError::ValueError(format!(
            "output.len() = {} but output_space.size() = {}",
            output.len(),
            output_space.size(),
        )));
    }

    if overwrite {
        output.iter_mut().for_each(|c| *c = C64::default());
    }

    if input.len() < PARALLEL_APPLY_THRESHOLD {
        for (i, coeff) in input.iter().enumerate() {
            for (state, amp) in input_space.expand_ref_state_iter(i, coeff) {
                if amp.norm_sqr() == 0.0 {
                    continue;
                }
                if let Some((j, scale)) = output_space.project(state) {
                    output[j] += amp * scale;
                }
            }
        }
    } else {
        let output_size = output_space.size();
        let combined: Vec<C64> = input
            .par_iter()
            .enumerate()
            .fold(
                || vec![C64::default(); output_size],
                |mut local_out, (i, coeff)| {
                    for (state, amp) in input_space.expand_ref_state_iter(i, coeff) {
                        if amp.norm_sqr() == 0.0 {
                            continue;
                        }
                        if let Some((j, scale)) = output_space.project(state) {
                            local_out[j] += amp * scale;
                        }
                    }
                    local_out
                },
            )
            .reduce(
                || vec![C64::default(); output_size],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x += y;
                    }
                    a
                },
            );
        for (o, c) in output.iter_mut().zip(combined.iter()) {
            *o += c;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn validate_args(
    num_cindices: usize,
    coeffs_len: usize,
    input_n_sites: usize,
    input_lhss: usize,
    input_size: usize,
    input_len: usize,
    output_n_sites: usize,
    output_lhss: usize,
    output_size: usize,
    output_len: usize,
) -> Result<(), QuSpinError> {
    if coeffs_len != num_cindices {
        return Err(QuSpinError::ValueError(format!(
            "coeffs.len() = {coeffs_len} but operator has {num_cindices} cindices"
        )));
    }
    if input_n_sites != output_n_sites {
        return Err(QuSpinError::ValueError(format!(
            "input basis has n_sites={input_n_sites} but output basis has n_sites={output_n_sites}"
        )));
    }
    if input_lhss != output_lhss {
        return Err(QuSpinError::ValueError(format!(
            "input basis has lhss={input_lhss} but output basis has lhss={output_lhss}"
        )));
    }
    if input_len != input_size {
        return Err(QuSpinError::ValueError(format!(
            "input.len() = {input_len} but input_space.size() = {input_size}"
        )));
    }
    if output_len != output_size {
        return Err(QuSpinError::ValueError(format!(
            "output.len() = {output_len} but output_space.size() = {output_size}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SpaceInner dispatch
// ---------------------------------------------------------------------------

pub use quspin_basis::dispatch::SpaceInner;
use quspin_basis::dispatch::{
    SpaceInnerBit, SpaceInnerBitDefault, SpaceInnerDit, SpaceInnerDitDefault, SpaceInnerQuat,
    SpaceInnerQuatDefault, SpaceInnerTrit, SpaceInnerTritDefault,
};
#[cfg(feature = "large-int")]
use quspin_basis::dispatch::{
    SpaceInnerBitLargeInt, SpaceInnerDitLargeInt, SpaceInnerQuatLargeInt, SpaceInnerTritLargeInt,
};

/// Cross-dispatch over a per-family `Default` sub-enum (variants
/// `Full32`/`Full64`/`Sub32`/`Sub64`/`Sub128`/`Sub256`/`Sym32`/`Sym64`/
/// `Sym128`/`Sym256`). Only same-width pairs are valid; cross-width
/// pairs fall through to the catch-all error arm.
///
/// `$body` is an expression using the bound `in_s` and `out_s`
/// identifiers (closure-like syntax). The same body shape is reused by
/// `apply_and_project_to` and `project_to` with different inner-fn
/// invocations.
macro_rules! dispatch_default {
    ($Enum:ident, $input:expr, $output:expr, |$in_s:ident, $out_s:ident| $body:expr) => {
        match ($input, $output) {
            // width 32
            ($Enum::Full32($in_s), $Enum::Full32($out_s)) => $body,
            ($Enum::Full32($in_s), $Enum::Sub32($out_s)) => $body,
            ($Enum::Full32($in_s), $Enum::Sym32($out_s)) => $body,
            ($Enum::Sub32($in_s), $Enum::Full32($out_s)) => $body,
            ($Enum::Sub32($in_s), $Enum::Sub32($out_s)) => $body,
            ($Enum::Sub32($in_s), $Enum::Sym32($out_s)) => $body,
            ($Enum::Sym32($in_s), $Enum::Full32($out_s)) => $body,
            ($Enum::Sym32($in_s), $Enum::Sub32($out_s)) => $body,
            ($Enum::Sym32($in_s), $Enum::Sym32($out_s)) => $body,
            // width 64
            ($Enum::Full64($in_s), $Enum::Full64($out_s)) => $body,
            ($Enum::Full64($in_s), $Enum::Sub64($out_s)) => $body,
            ($Enum::Full64($in_s), $Enum::Sym64($out_s)) => $body,
            ($Enum::Sub64($in_s), $Enum::Full64($out_s)) => $body,
            ($Enum::Sub64($in_s), $Enum::Sub64($out_s)) => $body,
            ($Enum::Sub64($in_s), $Enum::Sym64($out_s)) => $body,
            ($Enum::Sym64($in_s), $Enum::Full64($out_s)) => $body,
            ($Enum::Sym64($in_s), $Enum::Sub64($out_s)) => $body,
            ($Enum::Sym64($in_s), $Enum::Sym64($out_s)) => $body,
            // width 128
            ($Enum::Sub128($in_s), $Enum::Sub128($out_s)) => $body,
            ($Enum::Sub128($in_s), $Enum::Sym128($out_s)) => $body,
            ($Enum::Sym128($in_s), $Enum::Sub128($out_s)) => $body,
            ($Enum::Sym128($in_s), $Enum::Sym128($out_s)) => $body,
            // width 256
            ($Enum::Sub256($in_s), $Enum::Sub256($out_s)) => $body,
            ($Enum::Sub256($in_s), $Enum::Sym256($out_s)) => $body,
            ($Enum::Sym256($in_s), $Enum::Sub256($out_s)) => $body,
            ($Enum::Sym256($in_s), $Enum::Sym256($out_s)) => $body,
            _ => Err(QuSpinError::ValueError(
                "input and output bases have incompatible state integer types".into(),
            )),
        }
    };
}

/// Cross-dispatch over a per-family `LargeInt` sub-enum (variants
/// `Sub*` / `Sym*` for widths 512..8192). See [`dispatch_default`].
#[cfg(feature = "large-int")]
macro_rules! dispatch_largeint {
    ($Enum:ident, $input:expr, $output:expr, |$in_s:ident, $out_s:ident| $body:expr) => {
        match ($input, $output) {
            // width 512
            ($Enum::Sub512($in_s), $Enum::Sub512($out_s)) => $body,
            ($Enum::Sub512($in_s), $Enum::Sym512($out_s)) => $body,
            ($Enum::Sym512($in_s), $Enum::Sub512($out_s)) => $body,
            ($Enum::Sym512($in_s), $Enum::Sym512($out_s)) => $body,
            // width 1024
            ($Enum::Sub1024($in_s), $Enum::Sub1024($out_s)) => $body,
            ($Enum::Sub1024($in_s), $Enum::Sym1024($out_s)) => $body,
            ($Enum::Sym1024($in_s), $Enum::Sub1024($out_s)) => $body,
            ($Enum::Sym1024($in_s), $Enum::Sym1024($out_s)) => $body,
            // width 2048
            ($Enum::Sub2048($in_s), $Enum::Sub2048($out_s)) => $body,
            ($Enum::Sub2048($in_s), $Enum::Sym2048($out_s)) => $body,
            ($Enum::Sym2048($in_s), $Enum::Sub2048($out_s)) => $body,
            ($Enum::Sym2048($in_s), $Enum::Sym2048($out_s)) => $body,
            // width 4096
            ($Enum::Sub4096($in_s), $Enum::Sub4096($out_s)) => $body,
            ($Enum::Sub4096($in_s), $Enum::Sym4096($out_s)) => $body,
            ($Enum::Sym4096($in_s), $Enum::Sub4096($out_s)) => $body,
            ($Enum::Sym4096($in_s), $Enum::Sym4096($out_s)) => $body,
            // width 8192
            ($Enum::Sub8192($in_s), $Enum::Sub8192($out_s)) => $body,
            ($Enum::Sub8192($in_s), $Enum::Sym8192($out_s)) => $body,
            ($Enum::Sym8192($in_s), $Enum::Sub8192($out_s)) => $body,
            ($Enum::Sym8192($in_s), $Enum::Sym8192($out_s)) => $body,
            _ => Err(QuSpinError::ValueError(
                "input and output bases have incompatible state integer types".into(),
            )),
        }
    };
}

/// Type-erased dispatch: apply operator to a vector in `input` space and
/// project the result into `output` space.
///
/// Called from each `OperatorInner::apply_and_project_to` after resolving the
/// cindex type. Errors when the input and output bases have different
/// LHSS families (no cross-LHSS apply is supported).
pub fn apply_and_project_to<H, C>(
    op: &H,
    input: &SpaceInner,
    output: &SpaceInner,
    coeffs: &[C64],
    in_vec: &[C64],
    out_vec: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError>
where
    H: Operator<C> + Sync,
    C: CIndex,
{
    match (input, output) {
        // ----------------------------------------------------------------
        // Bit family
        // ----------------------------------------------------------------
        (
            SpaceInner::Bit(SpaceInnerBit::Default(i)),
            SpaceInner::Bit(SpaceInnerBit::Default(o)),
        ) => {
            dispatch_default!(SpaceInnerBitDefault, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (
            SpaceInner::Bit(SpaceInnerBit::LargeInt(i)),
            SpaceInner::Bit(SpaceInnerBit::LargeInt(o)),
        ) => dispatch_largeint!(SpaceInnerBitLargeInt, i, o, |in_s, out_s| {
            apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
        }),

        // ----------------------------------------------------------------
        // Trit family
        // ----------------------------------------------------------------
        (
            SpaceInner::Trit(SpaceInnerTrit::Default(i)),
            SpaceInner::Trit(SpaceInnerTrit::Default(o)),
        ) => dispatch_default!(SpaceInnerTritDefault, i, o, |in_s, out_s| {
            apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
        }),
        #[cfg(feature = "large-int")]
        (
            SpaceInner::Trit(SpaceInnerTrit::LargeInt(i)),
            SpaceInner::Trit(SpaceInnerTrit::LargeInt(o)),
        ) => dispatch_largeint!(SpaceInnerTritLargeInt, i, o, |in_s, out_s| {
            apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
        }),

        // ----------------------------------------------------------------
        // Quat family
        // ----------------------------------------------------------------
        (
            SpaceInner::Quat(SpaceInnerQuat::Default(i)),
            SpaceInner::Quat(SpaceInnerQuat::Default(o)),
        ) => dispatch_default!(SpaceInnerQuatDefault, i, o, |in_s, out_s| {
            apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
        }),
        #[cfg(feature = "large-int")]
        (
            SpaceInner::Quat(SpaceInnerQuat::LargeInt(i)),
            SpaceInner::Quat(SpaceInnerQuat::LargeInt(o)),
        ) => dispatch_largeint!(SpaceInnerQuatLargeInt, i, o, |in_s, out_s| {
            apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
        }),

        // ----------------------------------------------------------------
        // Dit family
        // ----------------------------------------------------------------
        (
            SpaceInner::Dit(SpaceInnerDit::Default(i)),
            SpaceInner::Dit(SpaceInnerDit::Default(o)),
        ) => {
            dispatch_default!(SpaceInnerDitDefault, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (
            SpaceInner::Dit(SpaceInnerDit::LargeInt(i)),
            SpaceInner::Dit(SpaceInnerDit::LargeInt(o)),
        ) => dispatch_largeint!(SpaceInnerDitLargeInt, i, o, |in_s, out_s| {
            apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
        }),

        _ => Err(QuSpinError::ValueError(
            "input and output bases have incompatible LHSS families or width groups".into(),
        )),
    }
}

/// Type-erased dispatch for identity projection (no operator).
pub fn project_to(
    input: &SpaceInner,
    output: &SpaceInner,
    in_vec: &[C64],
    out_vec: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError> {
    match (input, output) {
        // Bit family
        (
            SpaceInner::Bit(SpaceInnerBit::Default(i)),
            SpaceInner::Bit(SpaceInnerBit::Default(o)),
        ) => {
            dispatch_default!(SpaceInnerBitDefault, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (
            SpaceInner::Bit(SpaceInnerBit::LargeInt(i)),
            SpaceInner::Bit(SpaceInnerBit::LargeInt(o)),
        ) => dispatch_largeint!(SpaceInnerBitLargeInt, i, o, |in_s, out_s| {
            project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
        }),

        // Trit family
        (
            SpaceInner::Trit(SpaceInnerTrit::Default(i)),
            SpaceInner::Trit(SpaceInnerTrit::Default(o)),
        ) => dispatch_default!(SpaceInnerTritDefault, i, o, |in_s, out_s| {
            project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
        }),
        #[cfg(feature = "large-int")]
        (
            SpaceInner::Trit(SpaceInnerTrit::LargeInt(i)),
            SpaceInner::Trit(SpaceInnerTrit::LargeInt(o)),
        ) => dispatch_largeint!(SpaceInnerTritLargeInt, i, o, |in_s, out_s| {
            project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
        }),

        // Quat family
        (
            SpaceInner::Quat(SpaceInnerQuat::Default(i)),
            SpaceInner::Quat(SpaceInnerQuat::Default(o)),
        ) => dispatch_default!(SpaceInnerQuatDefault, i, o, |in_s, out_s| {
            project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
        }),
        #[cfg(feature = "large-int")]
        (
            SpaceInner::Quat(SpaceInnerQuat::LargeInt(i)),
            SpaceInner::Quat(SpaceInnerQuat::LargeInt(o)),
        ) => dispatch_largeint!(SpaceInnerQuatLargeInt, i, o, |in_s, out_s| {
            project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
        }),

        // Dit family
        (
            SpaceInner::Dit(SpaceInnerDit::Default(i)),
            SpaceInner::Dit(SpaceInnerDit::Default(o)),
        ) => {
            dispatch_default!(SpaceInnerDitDefault, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (
            SpaceInner::Dit(SpaceInnerDit::LargeInt(i)),
            SpaceInner::Dit(SpaceInnerDit::LargeInt(o)),
        ) => dispatch_largeint!(SpaceInnerDitLargeInt, i, o, |in_s, out_s| {
            project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
        }),

        _ => Err(QuSpinError::ValueError(
            "input and output bases have incompatible LHSS families or width groups".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qmatrix::build::build_from_basis;
    use quspin_basis::space::{FullSpace, Subspace};
    use quspin_operator::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
    use smallvec::smallvec;

    /// S+ = (X + iY)/2 = |1⟩⟨0| (raises spin).
    /// As a HardcoreOperator: P operator (creation).
    fn sp_operator() -> HardcoreOperator<u8> {
        let terms = vec![OpEntry::new(
            0u8,
            C64::new(1.0, 0.0),
            smallvec![(HardcoreOp::P, 0u32)],
        )];
        HardcoreOperator::new(terms)
    }

    /// XX Hamiltonian on 2 sites.
    fn xx_ham() -> HardcoreOperator<u8> {
        let ops = smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, C64::new(1.0, 0.0), ops)];
        HardcoreOperator::new(terms)
    }

    #[test]
    fn apply_same_space_matches_qmatrix_dot() {
        // Compare apply() with QMatrix::dot for XX on 2-site 1-particle subspace
        let ham = xx_ham();
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0b01u32, &ham);
        assert_eq!(sub.size(), 2);

        let mat = build_from_basis::<_, u32, C64, i64, u8, _>(&ham, &sub);
        let coeffs = vec![C64::new(1.0, 0.0)];
        let psi = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];

        // QMatrix dot (Complex<f64> matrix)
        let mut out_mat = vec![C64::default(); 2];
        mat.dot(true, &coeffs, &psi, &mut out_mat).unwrap();

        // apply()
        let mut out_apply = vec![C64::default(); 2];
        apply_and_project_to_inner(&ham, &sub, &sub, &coeffs, &psi, &mut out_apply, true).unwrap();

        for i in 0..2 {
            assert!(
                (out_mat[i] - out_apply[i]).norm() < 1e-12,
                "mismatch at {i}: mat={}, apply={}",
                out_mat[i],
                out_apply[i],
            );
        }
    }

    /// Diagonal ZZ operator (conserves particle number, useful for BFS seeds).
    fn zz_ham() -> HardcoreOperator<u8> {
        let ops = smallvec![(HardcoreOp::Z, 0u32), (HardcoreOp::Z, 1u32)];
        let terms = vec![OpEntry::new(0u8, C64::new(1.0, 0.0), ops)];
        HardcoreOperator::new(terms)
    }

    #[test]
    fn project_sp_from_sz0_to_sz1() {
        // 2-site system: apply P_0 (S+ on site 0) to a vector in Sz=0 sector,
        // project into Sz=1 sector.
        //
        // Sz=0 states: {|01⟩=1, |10⟩=2} (1-particle sector)
        // Sz=1 states: {|11⟩=3} (2-particle sector)
        //
        // P_0|01⟩ = |11⟩ (site 0 was 0, now 1)
        // P_0|10⟩ = 0    (site 0 already 1)
        let sp = sp_operator();
        let xx = xx_ham();
        let zz = zz_ham();

        // Build Sz=0 subspace (1-particle) — XX connects |01⟩↔|10⟩
        let mut sz0 = Subspace::<u32>::new(2, 2, false);
        sz0.build(0b01u32, &xx);
        assert_eq!(sz0.size(), 2);

        // Build Sz=1 subspace (2-particle) — ZZ is diagonal, stays at |11⟩
        let mut sz1 = Subspace::<u32>::new(2, 2, false);
        sz1.build(0b11u32, &zz);
        assert_eq!(sz1.size(), 1);

        // Input: |ψ⟩ = |01⟩ + |10⟩ (equal superposition in Sz=0)
        // Subspace sorted ascending: state_at(0)=1=|01⟩, state_at(1)=2=|10⟩
        let psi = vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)];
        let coeffs = vec![C64::new(1.0, 0.0)];
        let mut result = vec![C64::default(); 1];

        apply_and_project_to_inner(&sp, &sz0, &sz1, &coeffs, &psi, &mut result, true).unwrap();

        // P_0|01⟩ = |11⟩ → in Sz=1, P_0|10⟩ = 0 → dropped
        // result[0] should be 1.0 (from the |01⟩ component)
        assert!(
            (result[0] - C64::new(1.0, 0.0)).norm() < 1e-12,
            "result = {}, expected 1.0",
            result[0],
        );
    }

    #[test]
    fn project_to_full_space() {
        // Apply XX to 1-particle subspace vector, project into full space
        let ham = xx_ham();
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0b01u32, &ham);
        let full = FullSpace::<u32>::new(2, 2, false);

        // |ψ⟩ = |01⟩ in subspace
        let psi = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let coeffs = vec![C64::new(1.0, 0.0)];
        let mut result = vec![C64::default(); 4];

        apply_and_project_to_inner(&ham, &sub, &full, &coeffs, &psi, &mut result, true).unwrap();

        // XX|01⟩ = |10⟩
        // FullSpace: state_at(0)=3, state_at(1)=2=|10⟩, state_at(2)=1, state_at(3)=0
        // So |10⟩ → index 1
        assert!(
            (result[1] - C64::new(1.0, 0.0)).norm() < 1e-12,
            "result[1] = {}, expected 1.0",
            result[1],
        );
        // All others zero
        for (i, r) in result.iter().enumerate() {
            if i != 1 {
                assert!(r.norm() < 1e-12, "result[{i}] = {r}, expected 0");
            }
        }
    }

    #[test]
    fn overwrite_false_accumulates() {
        let ham = xx_ham();
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0b01u32, &ham);

        let psi = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let coeffs = vec![C64::new(1.0, 0.0)];
        let mut out = vec![C64::new(5.0, 0.0); 2];

        apply_and_project_to_inner(&ham, &sub, &sub, &coeffs, &psi, &mut out, false).unwrap();

        // XX|01⟩ = |10⟩, so out[1] should be 5 + 1 = 6
        assert!(
            (out[1] - C64::new(6.0, 0.0)).norm() < 1e-12,
            "out[1] = {}, expected 6.0",
            out[1],
        );
    }

    #[test]
    fn coeffs_scale_result() {
        let ham = xx_ham();
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0b01u32, &ham);

        let psi = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let coeffs = vec![C64::new(3.0, 1.0)]; // scale by 3+i
        let mut out = vec![C64::default(); 2];

        apply_and_project_to_inner(&ham, &sub, &sub, &coeffs, &psi, &mut out, true).unwrap();

        // XX|01⟩ = |10⟩ with amp 1.0, scaled by 3+i
        assert!(
            (out[1] - C64::new(3.0, 1.0)).norm() < 1e-12,
            "out[1] = {}, expected 3+i",
            out[1],
        );
    }

    #[test]
    fn apply_symmetric_basis_matches_qmatrix_dot() {
        // Compare apply on a SymBasis vs QMatrix::dot to verify normalization.
        // 3-site system with translation symmetry (k=0), using X operator to
        // connect all states.
        use crate::qmatrix::build::build_from_symmetric;
        use quspin_basis::sym::SymBasis;
        use quspin_bitbasis::PermDitMask;

        let n_sites = 3;
        // Use same X operator as the SymBasis tests to connect all states
        let x_op = quspin_bitbasis::test_graphs::XAllSites::new(n_sites as u32);

        // XX + ZZ chain for the Hamiltonian
        let mut terms = Vec::new();
        for i in 0..n_sites {
            let j = (i + 1) % n_sites;
            let xx = smallvec![(HardcoreOp::X, i as u32), (HardcoreOp::X, j as u32)];
            terms.push(OpEntry::new(0u8, C64::new(1.0, 0.0), xx));
            let zz = smallvec![(HardcoreOp::Z, i as u32), (HardcoreOp::Z, j as u32)];
            terms.push(OpEntry::new(0u8, C64::new(1.0, 0.0), zz));
        }
        let ham = HardcoreOperator::new(terms);

        // Build symmetric basis: cyclic translation group of order n_sites,
        // k=0 representation (character = 1 for every power).
        let perm: Vec<usize> = (0..n_sites).map(|i| (i + 1) % n_sites).collect();
        let mut sym = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, n_sites, false);
        sym.add_cyclic(quspin_basis::SymElement::lattice(&perm), n_sites, |_| {
            C64::new(1.0, 0.0)
        })
        .unwrap();

        // Seed with |000⟩ = 0 — X flips connect all states
        sym.build(0u32, &x_op).unwrap();
        let dim = sym.size();
        assert!(dim > 0, "symmetric basis is empty (dim=0)");

        // Build QMatrix in the symmetric basis
        let mat = build_from_symmetric::<_, u32, PermDitMask<u32>, u32, C64, i64, u8>(&ham, &sym);

        // Test vector
        let mut psi = vec![C64::default(); dim];
        psi[0] = C64::new(1.0, 0.0);
        if dim > 1 {
            psi[1] = C64::new(0.5, 0.3);
        }
        let coeffs = vec![C64::new(1.0, 0.0)];

        // QMatrix::dot
        let mut out_mat = vec![C64::default(); dim];
        mat.dot(true, &coeffs, &psi, &mut out_mat).unwrap();

        // apply_and_project_to_inner (same input/output SymBasis)
        let mut out_apply = vec![C64::default(); dim];
        apply_and_project_to_inner(&ham, &sym, &sym, &coeffs, &psi, &mut out_apply, true).unwrap();

        for i in 0..dim {
            assert!(
                (out_mat[i] - out_apply[i]).norm() < 1e-10,
                "SymBasis mismatch at {i}: mat={}, apply={}",
                out_mat[i],
                out_apply[i],
            );
        }
    }

    #[test]
    fn wrong_coeffs_len_errors() {
        let ham = xx_ham();
        let sub = Subspace::<u32>::new(2, 2, false);
        let psi: Vec<C64> = vec![];
        let coeffs = vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)]; // 2, but only 1 cindex
        let mut out: Vec<C64> = vec![];
        assert!(
            apply_and_project_to_inner(&ham, &sub, &sub, &coeffs, &psi, &mut out, true).is_err()
        );
    }
}
