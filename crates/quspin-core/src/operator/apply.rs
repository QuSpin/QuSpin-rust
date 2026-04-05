use super::Operator;
use crate::basis::expand::ExpandRefState;
use crate::basis::space::{FullSpace, Subspace};
use crate::basis::sym::{NormInt, SymBasis};
use crate::basis::traits::BasisSpace;
use crate::bitbasis::{BitInt, BitStateOp};
use crate::error::QuSpinError;
use crate::qmatrix::CIndex;
use num_complex::Complex;

type C64 = Complex<f64>;

// ---------------------------------------------------------------------------
// ProjectState — zero-cost abstraction over output projection
// ---------------------------------------------------------------------------

/// Project a full-space state into a basis, returning the index and a
/// complex scaling factor.
///
/// For non-symmetric bases the scale is always `1.0`.  For `SymBasis` the
/// state is mapped to its orbit representative and scaled by
/// `conj(χ_g) / norm`.  Fully monomorphized — zero runtime dispatch.
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

impl<B: BitInt, L: BitStateOp<B>, N: NormInt> ProjectState<B> for SymBasis<B, L, N> {
    fn project(&self, state: B) -> Option<(usize, C64)> {
        let (rep, grp_char) = self.get_refstate(state);
        self.index(rep).map(|j| {
            let (_, norm) = self.entry(j);
            (j, grp_char.conj() / C64::new(norm, 0.0))
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
    H: Operator<C>,
    B: BitInt,
    C: CIndex,
    IS: BasisSpace<B> + ExpandRefState<B, C64, C64>,
    OS: ProjectState<B>,
{
    validate_args(
        op.num_cindices(),
        coeffs.len(),
        input_space.size(),
        input.len(),
        output_space.size(),
        output.len(),
    )?;

    if overwrite {
        output.iter_mut().for_each(|c| *c = C64::default());
    }

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
    IS: BasisSpace<B> + ExpandRefState<B, C64, C64>,
    OS: ProjectState<B>,
{
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

    Ok(())
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

fn validate_args(
    num_cindices: usize,
    coeffs_len: usize,
    input_size: usize,
    input_len: usize,
    output_size: usize,
    output_len: usize,
) -> Result<(), QuSpinError> {
    if coeffs_len != num_cindices {
        return Err(QuSpinError::ValueError(format!(
            "coeffs.len() = {coeffs_len} but operator has {num_cindices} cindices"
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

pub use crate::basis::dispatch::SpaceInner;

/// Handle a single input variant: dispatch over all output variants.
macro_rules! dispatch_one_input {
    (
        $op:expr, $input:expr, $output:expr,
        $coeffs:expr, $in_vec:expr, $out_vec:expr, $overwrite:expr,
        $InVar:ident,
        output = [$($OutVar:ident),*]
    ) => {
        if let SpaceInner::$InVar(in_s) = $input {
            $(
                if let SpaceInner::$OutVar(out_s) = $output {
                    return apply_and_project_to_inner($op, in_s, out_s, $coeffs, $in_vec, $out_vec, $overwrite);
                }
            )*
            return Err(QuSpinError::ValueError(
                "input and output bases have incompatible state integer types".into(),
            ));
        }
    };
}

/// Dispatch over one BitInt family via recursive tt-munching.
/// Peels off one input variant at a time to avoid repetition count conflicts.
macro_rules! dispatch_project_b {
    // Base case: no more input variants
    (
        $op:expr, $input:expr, $output:expr,
        $coeffs:expr, $in_vec:expr, $out_vec:expr, $overwrite:expr,
        input = [],
        $($out_tt:tt)*
    ) => {};
    // Recursive case: peel off first input variant
    (
        $op:expr, $input:expr, $output:expr,
        $coeffs:expr, $in_vec:expr, $out_vec:expr, $overwrite:expr,
        input = [$first:ident $(, $rest:ident)*],
        $($out_tt:tt)*
    ) => {
        dispatch_one_input!(
            $op, $input, $output,
            $coeffs, $in_vec, $out_vec, $overwrite,
            $first,
            $($out_tt)*
        );
        dispatch_project_b!(
            $op, $input, $output,
            $coeffs, $in_vec, $out_vec, $overwrite,
            input = [$($rest),*],
            $($out_tt)*
        );
    };
}

/// Type-erased dispatch: apply operator to a vector in `input` space and
/// project the result into `output` space.
///
/// Called from each `OperatorInner::apply_and_project_to` after resolving the
/// cindex type.
#[allow(unused_imports, dead_code)]
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
    H: Operator<C>,
    C: CIndex,
{
    #[allow(unused_imports)]
    use crate::bitbasis::{DynamicPermDitValues, PermDitMask, PermDitValues};

    type B128 = ruint::Uint<128, 2>;
    type B256 = ruint::Uint<256, 4>;
    #[cfg(feature = "large-int")]
    type B512 = ruint::Uint<512, 8>;
    #[cfg(feature = "large-int")]
    type B1024 = ruint::Uint<1024, 16>;
    #[cfg(feature = "large-int")]
    type B2048 = ruint::Uint<2048, 32>;
    #[cfg(feature = "large-int")]
    type B4096 = ruint::Uint<4096, 64>;
    #[cfg(feature = "large-int")]
    type B8192 = ruint::Uint<8192, 128>;

    // u32 family
    dispatch_project_b!(
        op,
        input,
        output,
        coeffs,
        in_vec,
        out_vec,
        overwrite,
        input = [Full32, Sub32, Sym32, TritSym32, QuatSym32, DitSym32],
        output = [Full32, Sub32, Sym32, TritSym32, QuatSym32, DitSym32]
    );

    // u64 family
    dispatch_project_b!(
        op,
        input,
        output,
        coeffs,
        in_vec,
        out_vec,
        overwrite,
        input = [Full64, Sub64, Sym64, TritSym64, QuatSym64, DitSym64],
        output = [Full64, Sub64, Sym64, TritSym64, QuatSym64, DitSym64]
    );

    // B128 family
    dispatch_project_b!(
        op,
        input,
        output,
        coeffs,
        in_vec,
        out_vec,
        overwrite,
        input = [Sub128, Sym128, TritSym128, QuatSym128, DitSym128],
        output = [Sub128, Sym128, TritSym128, QuatSym128, DitSym128]
    );

    // B256 family
    dispatch_project_b!(
        op,
        input,
        output,
        coeffs,
        in_vec,
        out_vec,
        overwrite,
        input = [Sub256, Sym256, TritSym256, QuatSym256, DitSym256],
        output = [Sub256, Sym256, TritSym256, QuatSym256, DitSym256]
    );

    // large-int families (B512..B8192)
    #[cfg(feature = "large-int")]
    {
        dispatch_project_b!(
            op,
            input,
            output,
            coeffs,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub512, Sym512, TritSym512, QuatSym512, DitSym512],
            output = [Sub512, Sym512, TritSym512, QuatSym512, DitSym512]
        );
        dispatch_project_b!(
            op,
            input,
            output,
            coeffs,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub1024, Sym1024, TritSym1024, QuatSym1024, DitSym1024],
            output = [Sub1024, Sym1024, TritSym1024, QuatSym1024, DitSym1024]
        );
        dispatch_project_b!(
            op,
            input,
            output,
            coeffs,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub2048, Sym2048, TritSym2048, QuatSym2048, DitSym2048],
            output = [Sub2048, Sym2048, TritSym2048, QuatSym2048, DitSym2048]
        );
        dispatch_project_b!(
            op,
            input,
            output,
            coeffs,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub4096, Sym4096, TritSym4096, QuatSym4096, DitSym4096],
            output = [Sub4096, Sym4096, TritSym4096, QuatSym4096, DitSym4096]
        );
        dispatch_project_b!(
            op,
            input,
            output,
            coeffs,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub8192, Sym8192, TritSym8192, QuatSym8192, DitSym8192],
            output = [Sub8192, Sym8192, TritSym8192, QuatSym8192, DitSym8192]
        );
    }

    Err(QuSpinError::ValueError(
        "unsupported input basis type for apply_and_project_to \
         (unsupported or mismatched state integer types)"
            .into(),
    ))
}

/// Type-erased dispatch for identity projection (no operator).
#[allow(dead_code)]
pub fn project_to(
    input: &SpaceInner,
    output: &SpaceInner,
    in_vec: &[C64],
    out_vec: &mut [C64],
    overwrite: bool,
) -> Result<(), QuSpinError> {
    // Reuse the same dispatch macros by wrapping project_to in a closure
    // that matches the dispatch_one_input_project pattern.
    macro_rules! dispatch_one_project {
        ($input:expr, $output:expr, $in_vec:expr, $out_vec:expr, $overwrite:expr,
         $InVar:ident, output = [$($OutVar:ident),*]) => {
            if let SpaceInner::$InVar(in_s) = $input {
                $(
                    if let SpaceInner::$OutVar(out_s) = $output {
                        return project_to_inner(in_s, out_s, $in_vec, $out_vec, $overwrite);
                    }
                )*
                return Err(QuSpinError::ValueError(
                    "input and output bases have incompatible state integer types".into(),
                ));
            }
        };
    }

    macro_rules! dispatch_project_family {
        ($input:expr, $output:expr, $in_vec:expr, $out_vec:expr, $overwrite:expr,
         input = [], $($out_tt:tt)*) => {};
        ($input:expr, $output:expr, $in_vec:expr, $out_vec:expr, $overwrite:expr,
         input = [$first:ident $(, $rest:ident)*], $($out_tt:tt)*) => {
            dispatch_one_project!(
                $input, $output, $in_vec, $out_vec, $overwrite,
                $first, $($out_tt)*
            );
            dispatch_project_family!(
                $input, $output, $in_vec, $out_vec, $overwrite,
                input = [$($rest),*], $($out_tt)*
            );
        };
    }

    #[allow(unused_imports)]
    use crate::bitbasis::{DynamicPermDitValues, PermDitMask, PermDitValues};

    type B128 = ruint::Uint<128, 2>;
    type B256 = ruint::Uint<256, 4>;
    #[cfg(feature = "large-int")]
    type B512 = ruint::Uint<512, 8>;
    #[cfg(feature = "large-int")]
    type B1024 = ruint::Uint<1024, 16>;
    #[cfg(feature = "large-int")]
    type B2048 = ruint::Uint<2048, 32>;
    #[cfg(feature = "large-int")]
    type B4096 = ruint::Uint<4096, 64>;
    #[cfg(feature = "large-int")]
    type B8192 = ruint::Uint<8192, 128>;

    dispatch_project_family!(
        input,
        output,
        in_vec,
        out_vec,
        overwrite,
        input = [Full32, Sub32, Sym32, TritSym32, QuatSym32, DitSym32],
        output = [Full32, Sub32, Sym32, TritSym32, QuatSym32, DitSym32]
    );
    dispatch_project_family!(
        input,
        output,
        in_vec,
        out_vec,
        overwrite,
        input = [Full64, Sub64, Sym64, TritSym64, QuatSym64, DitSym64],
        output = [Full64, Sub64, Sym64, TritSym64, QuatSym64, DitSym64]
    );
    dispatch_project_family!(
        input,
        output,
        in_vec,
        out_vec,
        overwrite,
        input = [Sub128, Sym128, TritSym128, QuatSym128, DitSym128],
        output = [Sub128, Sym128, TritSym128, QuatSym128, DitSym128]
    );
    dispatch_project_family!(
        input,
        output,
        in_vec,
        out_vec,
        overwrite,
        input = [Sub256, Sym256, TritSym256, QuatSym256, DitSym256],
        output = [Sub256, Sym256, TritSym256, QuatSym256, DitSym256]
    );
    #[cfg(feature = "large-int")]
    {
        dispatch_project_family!(
            input,
            output,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub512, Sym512, TritSym512, QuatSym512, DitSym512],
            output = [Sub512, Sym512, TritSym512, QuatSym512, DitSym512]
        );
        dispatch_project_family!(
            input,
            output,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub1024, Sym1024, TritSym1024, QuatSym1024, DitSym1024],
            output = [Sub1024, Sym1024, TritSym1024, QuatSym1024, DitSym1024]
        );
        dispatch_project_family!(
            input,
            output,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub2048, Sym2048, TritSym2048, QuatSym2048, DitSym2048],
            output = [Sub2048, Sym2048, TritSym2048, QuatSym2048, DitSym2048]
        );
        dispatch_project_family!(
            input,
            output,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub4096, Sym4096, TritSym4096, QuatSym4096, DitSym4096],
            output = [Sub4096, Sym4096, TritSym4096, QuatSym4096, DitSym4096]
        );
        dispatch_project_family!(
            input,
            output,
            in_vec,
            out_vec,
            overwrite,
            input = [Sub8192, Sym8192, TritSym8192, QuatSym8192, DitSym8192],
            output = [Sub8192, Sym8192, TritSym8192, QuatSym8192, DitSym8192]
        );
    }

    Err(QuSpinError::ValueError(
        "unsupported input basis type for project_to \
         (unsupported or mismatched state integer types)"
            .into(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::space::{FullSpace, Subspace};
    use crate::operator::pauli::{HardcoreOp, HardcoreOperator, OpEntry};
    use crate::qmatrix::build::build_from_basis;
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
        sub.build(0b01u32, |s| ham.apply_smallvec(s).into_iter());
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
        sz0.build(0b01u32, |s| xx.apply_smallvec(s).into_iter());
        assert_eq!(sz0.size(), 2);

        // Build Sz=1 subspace (2-particle) — ZZ is diagonal, stays at |11⟩
        let mut sz1 = Subspace::<u32>::new(2, 2, false);
        sz1.build(0b11u32, |s| zz.apply_smallvec(s).into_iter());
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
        sub.build(0b01u32, |s| ham.apply_smallvec(s).into_iter());
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
        sub.build(0b01u32, |s| ham.apply_smallvec(s).into_iter());

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
        sub.build(0b01u32, |s| ham.apply_smallvec(s).into_iter());

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
        use crate::basis::sym::SymBasis;
        use crate::bitbasis::PermDitMask;
        use crate::qmatrix::build::build_from_symmetric;

        let n_sites = 3;
        // Use same X operator as the SymBasis tests to connect all states
        let x_op = |state: u32| -> Vec<(C64, u32, u8)> {
            (0..n_sites as u32)
                .map(|loc| (C64::new(1.0, 0.0), state ^ (1 << loc), 0u8))
                .collect()
        };

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

        // Build symmetric basis: cyclic shift [1,2,0], character = 1 (k=0)
        let perm: Vec<usize> = (0..n_sites).map(|i| (i + 1) % n_sites).collect();
        let mut sym = SymBasis::<u32, PermDitMask<u32>, u32>::new_empty(2, n_sites, false);
        sym.add_lattice(C64::new(1.0, 0.0), &perm);

        // Seed with |000⟩ = 0 — X flips connect all states
        sym.build(0u32, x_op);
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
