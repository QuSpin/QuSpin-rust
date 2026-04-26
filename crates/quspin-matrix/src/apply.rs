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
            (
                j,
                grp_char.conj() * C64::new((norm / group_order).sqrt(), 0.0),
            )
        })
    }
}

// ---------------------------------------------------------------------------
// apply_and_project_to_inner — fully monomorphized inner kernel
// ---------------------------------------------------------------------------

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
// project_to_inner — identity operator (expand + project, no operator)
// ---------------------------------------------------------------------------

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
// Type-erased dispatch
// ---------------------------------------------------------------------------

pub use quspin_basis::dispatch::{
    BitBasis, BitBasisDefault, DitBasis, DynDitBasis, DynDitBasisDefault, GenericBasis, QuatBasis,
    QuatBasisDefault, TritBasis, TritBasisDefault,
};
#[cfg(feature = "large-int")]
pub use quspin_basis::dispatch::{
    BitBasisLargeInt, DynDitBasisLargeInt, QuatBasisLargeInt, TritBasisLargeInt,
};

/// Cross-dispatch over a per-family `Default` sub-enum (variants
/// `Full32`/`Full64`/`Sub32`/`Sub64`/`Sub128`/`Sub256`/`Sym32`/`Sym64`/
/// `Sym128`/`Sym256`). Only same-width pairs are valid; cross-width
/// pairs fall through to the catch-all error arm.
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
/// `Sub*` / `Sym*` for widths 512..8192).
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

// ---------------------------------------------------------------------------
// apply_and_project_to_bit — bit-only entry point (FermionBasis path)
// ---------------------------------------------------------------------------

/// Apply an operator and project, restricted to the bit family
/// (LHSS = 2). Used directly by [`FermionBasis`](quspin_basis::FermionBasis)
/// so the fermion compile path never sees the dit-family monomorphizations.
pub fn apply_and_project_to_bit<H, C>(
    op: &H,
    input: &BitBasis,
    output: &BitBasis,
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
        (BitBasis::Default(i), BitBasis::Default(o)) => {
            dispatch_default!(BitBasisDefault, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (BitBasis::LargeInt(i), BitBasis::LargeInt(o)) => {
            dispatch_largeint!(BitBasisLargeInt, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        _ => Err(QuSpinError::ValueError(
            "input and output bases have incompatible state integer types".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// apply_and_project_to_dit — dit-only entry point (LHSS > 2)
// ---------------------------------------------------------------------------

/// Apply an operator and project, restricted to the dit family
/// (LHSS > 2).
pub fn apply_and_project_to_dit<H, C>(
    op: &H,
    input: &DitBasis,
    output: &DitBasis,
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
        (DitBasis::Trit(TritBasis::Default(i)), DitBasis::Trit(TritBasis::Default(o))) => {
            dispatch_default!(TritBasisDefault, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (DitBasis::Trit(TritBasis::LargeInt(i)), DitBasis::Trit(TritBasis::LargeInt(o))) => {
            dispatch_largeint!(TritBasisLargeInt, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        (DitBasis::Quat(QuatBasis::Default(i)), DitBasis::Quat(QuatBasis::Default(o))) => {
            dispatch_default!(QuatBasisDefault, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (DitBasis::Quat(QuatBasis::LargeInt(i)), DitBasis::Quat(QuatBasis::LargeInt(o))) => {
            dispatch_largeint!(QuatBasisLargeInt, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        (DitBasis::Dyn(DynDitBasis::Default(i)), DitBasis::Dyn(DynDitBasis::Default(o))) => {
            dispatch_default!(DynDitBasisDefault, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (DitBasis::Dyn(DynDitBasis::LargeInt(i)), DitBasis::Dyn(DynDitBasis::LargeInt(o))) => {
            dispatch_largeint!(DynDitBasisLargeInt, i, o, |in_s, out_s| {
                apply_and_project_to_inner(op, in_s, out_s, coeffs, in_vec, out_vec, overwrite)
            })
        }
        _ => Err(QuSpinError::ValueError(
            "input and output bases have incompatible LHSS families or width groups".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// apply_and_project_to — generic entry point
// ---------------------------------------------------------------------------

/// Type-erased dispatch: branch into [`apply_and_project_to_bit`] /
/// [`apply_and_project_to_dit`] depending on the LHSS family.
pub fn apply_and_project_to<H, C>(
    op: &H,
    input: &GenericBasis,
    output: &GenericBasis,
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
        (GenericBasis::Bit(i), GenericBasis::Bit(o)) => {
            apply_and_project_to_bit(op, i, o, coeffs, in_vec, out_vec, overwrite)
        }
        (GenericBasis::Dit(i), GenericBasis::Dit(o)) => {
            apply_and_project_to_dit(op, i, o, coeffs, in_vec, out_vec, overwrite)
        }
        _ => Err(QuSpinError::ValueError(
            "input and output bases have incompatible LHSS families".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// project_to — bit / dit / generic
// ---------------------------------------------------------------------------

/// Project a vector across two bit-family bases (LHSS = 2).
pub fn project_to_bit(
    input: &BitBasis,
    output: &BitBasis,
    in_vec: &[C64],
    out_vec: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError> {
    match (input, output) {
        (BitBasis::Default(i), BitBasis::Default(o)) => {
            dispatch_default!(BitBasisDefault, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (BitBasis::LargeInt(i), BitBasis::LargeInt(o)) => {
            dispatch_largeint!(BitBasisLargeInt, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        _ => Err(QuSpinError::ValueError(
            "input and output bases have incompatible state integer types".into(),
        )),
    }
}

/// Project a vector across two dit-family bases (LHSS > 2).
pub fn project_to_dit(
    input: &DitBasis,
    output: &DitBasis,
    in_vec: &[C64],
    out_vec: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError> {
    match (input, output) {
        (DitBasis::Trit(TritBasis::Default(i)), DitBasis::Trit(TritBasis::Default(o))) => {
            dispatch_default!(TritBasisDefault, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (DitBasis::Trit(TritBasis::LargeInt(i)), DitBasis::Trit(TritBasis::LargeInt(o))) => {
            dispatch_largeint!(TritBasisLargeInt, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        (DitBasis::Quat(QuatBasis::Default(i)), DitBasis::Quat(QuatBasis::Default(o))) => {
            dispatch_default!(QuatBasisDefault, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (DitBasis::Quat(QuatBasis::LargeInt(i)), DitBasis::Quat(QuatBasis::LargeInt(o))) => {
            dispatch_largeint!(QuatBasisLargeInt, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        (DitBasis::Dyn(DynDitBasis::Default(i)), DitBasis::Dyn(DynDitBasis::Default(o))) => {
            dispatch_default!(DynDitBasisDefault, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        #[cfg(feature = "large-int")]
        (DitBasis::Dyn(DynDitBasis::LargeInt(i)), DitBasis::Dyn(DynDitBasis::LargeInt(o))) => {
            dispatch_largeint!(DynDitBasisLargeInt, i, o, |in_s, out_s| {
                project_to_inner(in_s, out_s, in_vec, out_vec, overwrite)
            })
        }
        _ => Err(QuSpinError::ValueError(
            "input and output bases have incompatible LHSS families or width groups".into(),
        )),
    }
}

/// Type-erased identity-projection dispatch.
pub fn project_to(
    input: &GenericBasis,
    output: &GenericBasis,
    in_vec: &[C64],
    out_vec: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError> {
    match (input, output) {
        (GenericBasis::Bit(i), GenericBasis::Bit(o)) => {
            project_to_bit(i, o, in_vec, out_vec, overwrite)
        }
        (GenericBasis::Dit(i), GenericBasis::Dit(o)) => {
            project_to_dit(i, o, in_vec, out_vec, overwrite)
        }
        _ => Err(QuSpinError::ValueError(
            "input and output bases have incompatible LHSS families".into(),
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

    fn sp_operator() -> HardcoreOperator<u8> {
        let terms = vec![OpEntry::new(
            0u8,
            C64::new(1.0, 0.0),
            smallvec![(HardcoreOp::P, 0u32)],
        )];
        HardcoreOperator::new(terms)
    }

    fn xx_ham() -> HardcoreOperator<u8> {
        let ops = smallvec![(HardcoreOp::X, 0u32), (HardcoreOp::X, 1u32)];
        let terms = vec![OpEntry::new(0u8, C64::new(1.0, 0.0), ops)];
        HardcoreOperator::new(terms)
    }

    #[test]
    fn apply_same_space_matches_qmatrix_dot() {
        let ham = xx_ham();
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0b01u32, &ham);
        assert_eq!(sub.size(), 2);

        let mat = build_from_basis::<_, u32, C64, i64, u8, _>(&ham, &sub);
        let coeffs = vec![C64::new(1.0, 0.0)];
        let psi = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];

        let mut out_mat = vec![C64::default(); 2];
        mat.dot(true, &coeffs, &psi, &mut out_mat).unwrap();

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

    fn zz_ham() -> HardcoreOperator<u8> {
        let ops = smallvec![(HardcoreOp::Z, 0u32), (HardcoreOp::Z, 1u32)];
        let terms = vec![OpEntry::new(0u8, C64::new(1.0, 0.0), ops)];
        HardcoreOperator::new(terms)
    }

    #[test]
    fn project_sp_from_sz0_to_sz1() {
        let sp = sp_operator();
        let xx = xx_ham();
        let zz = zz_ham();

        let mut sz0 = Subspace::<u32>::new(2, 2, false);
        sz0.build(0b01u32, &xx);
        assert_eq!(sz0.size(), 2);

        let mut sz1 = Subspace::<u32>::new(2, 2, false);
        sz1.build(0b11u32, &zz);
        assert_eq!(sz1.size(), 1);

        let psi = vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)];
        let coeffs = vec![C64::new(1.0, 0.0)];
        let mut result = vec![C64::default(); 1];

        apply_and_project_to_inner(&sp, &sz0, &sz1, &coeffs, &psi, &mut result, true).unwrap();

        assert!(
            (result[0] - C64::new(1.0, 0.0)).norm() < 1e-12,
            "result = {}, expected 1.0",
            result[0],
        );
    }

    #[test]
    fn project_to_full_space() {
        let ham = xx_ham();
        let mut sub = Subspace::<u32>::new(2, 2, false);
        sub.build(0b01u32, &ham);
        let full = FullSpace::<u32>::new(2, 2, false);

        let psi = vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let coeffs = vec![C64::new(1.0, 0.0)];
        let mut result = vec![C64::default(); 4];

        apply_and_project_to_inner(&ham, &sub, &full, &coeffs, &psi, &mut result, true).unwrap();

        assert!(
            (result[1] - C64::new(1.0, 0.0)).norm() < 1e-12,
            "result[1] = {}, expected 1.0",
            result[1],
        );
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
        let coeffs = vec![C64::new(3.0, 1.0)];
        let mut out = vec![C64::default(); 2];

        apply_and_project_to_inner(&ham, &sub, &sub, &coeffs, &psi, &mut out, true).unwrap();

        assert!(
            (out[1] - C64::new(3.0, 1.0)).norm() < 1e-12,
            "out[1] = {}, expected 3+i",
            out[1],
        );
    }

    #[test]
    fn apply_symmetric_basis_matches_qmatrix_dot() {
        use crate::qmatrix::build::build_from_symmetric;
        use quspin_basis::sym::SymBasis;
        use quspin_bitbasis::PermDitMask;

        let n_sites = 3;
        let x_op = quspin_bitbasis::test_graphs::XAllSites::new(n_sites as u32);

        let mut terms = Vec::new();
        for i in 0..n_sites {
            let j = (i + 1) % n_sites;
            let xx = smallvec![(HardcoreOp::X, i as u32), (HardcoreOp::X, j as u32)];
            terms.push(OpEntry::new(0u8, C64::new(1.0, 0.0), xx));
            let zz = smallvec![(HardcoreOp::Z, i as u32), (HardcoreOp::Z, j as u32)];
            terms.push(OpEntry::new(0u8, C64::new(1.0, 0.0), zz));
        }
        let ham = HardcoreOperator::new(terms);

        let perm: Vec<usize> = (0..n_sites).map(|i| (i + 1) % n_sites).collect();
        let mut sym = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, n_sites, false);
        sym.add_cyclic(quspin_basis::SymElement::lattice(&perm), n_sites, |_| {
            C64::new(1.0, 0.0)
        })
        .unwrap();

        sym.build(0u32, &x_op).unwrap();
        let dim = sym.size();
        assert!(dim > 0, "symmetric basis is empty (dim=0)");

        let mat = build_from_symmetric::<_, u32, PermDitMask<u32>, u32, C64, i64, u8>(&ham, &sym);

        let mut psi = vec![C64::default(); dim];
        psi[0] = C64::new(1.0, 0.0);
        if dim > 1 {
            psi[1] = C64::new(0.5, 0.3);
        }
        let coeffs = vec![C64::new(1.0, 0.0)];

        let mut out_mat = vec![C64::default(); dim];
        mat.dot(true, &coeffs, &psi, &mut out_mat).unwrap();

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
        let coeffs = vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)];
        let mut out: Vec<C64> = vec![];
        assert!(
            apply_and_project_to_inner(&ham, &sub, &sub, &coeffs, &psi, &mut out, true).is_err()
        );
    }
}
