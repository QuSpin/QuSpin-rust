use super::QMatrix;
/// Type-erased `QMatrixInner`, `IntoQMatrixInner`, and the `with_qmatrix!` macro.
///
/// 12 variants: 6 value types × 2 cindex types.  The index type `I` is fixed
/// to `i64` at the FFI boundary.
///
/// Naming convention: `QM` prefix, value type abbreviation, cindex type
/// abbreviation.  For example `QMf64U8` is `QMatrix<f64, i64, u8>`.
use super::build::build_from_space;
use crate::basis::dispatch::SpaceInner;
use crate::dtype::ValueDType;
use crate::error::QuSpinError;
use crate::operator::bond::BondOperatorInner;
use crate::operator::boson::BosonOperatorInner;
use crate::operator::fermion::FermionOperatorInner;
use crate::operator::pauli::HardcoreOperatorInner;
use crate::primitive::Primitive;
use crate::qmatrix::matrix::{CIndex, Index};
use ndarray::{ArrayView2, ArrayViewMut2};

/// Return type of `QMatrixInner::materialize`.
type CsrTriple = (Vec<i64>, Vec<i64>, Vec<num_complex::Complex<f64>>);

// ---------------------------------------------------------------------------
// QMatrixInner
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum QMatrixInner {
    QMi8U8(QMatrix<i8, i64, u8>),
    QMi8U16(QMatrix<i8, i64, u16>),
    QMi16U8(QMatrix<i16, i64, u8>),
    QMi16U16(QMatrix<i16, i64, u16>),
    QMf32U8(QMatrix<f32, i64, u8>),
    QMf32U16(QMatrix<f32, i64, u16>),
    QMf64U8(QMatrix<f64, i64, u8>),
    QMf64U16(QMatrix<f64, i64, u16>),
    /// `complex64` (2 × f32).
    QMc32U8(QMatrix<num_complex::Complex<f32>, i64, u8>),
    QMc32U16(QMatrix<num_complex::Complex<f32>, i64, u16>),
    /// `complex128` (2 × f64).
    QMc64U8(QMatrix<num_complex::Complex<f64>, i64, u8>),
    QMc64U16(QMatrix<num_complex::Complex<f64>, i64, u16>),
}

impl QMatrixInner {
    pub fn dim(&self) -> usize {
        match self {
            QMatrixInner::QMi8U8(m) => m.dim(),
            QMatrixInner::QMi8U16(m) => m.dim(),
            QMatrixInner::QMi16U8(m) => m.dim(),
            QMatrixInner::QMi16U16(m) => m.dim(),
            QMatrixInner::QMf32U8(m) => m.dim(),
            QMatrixInner::QMf32U16(m) => m.dim(),
            QMatrixInner::QMf64U8(m) => m.dim(),
            QMatrixInner::QMf64U16(m) => m.dim(),
            QMatrixInner::QMc32U8(m) => m.dim(),
            QMatrixInner::QMc32U16(m) => m.dim(),
            QMatrixInner::QMc64U8(m) => m.dim(),
            QMatrixInner::QMc64U16(m) => m.dim(),
        }
    }

    pub fn num_coeff(&self) -> usize {
        crate::with_qmatrix!(self, _M, _C, mat, { mat.num_coeff() })
    }

    pub fn nnz(&self) -> usize {
        match self {
            QMatrixInner::QMi8U8(m) => m.nnz(),
            QMatrixInner::QMi8U16(m) => m.nnz(),
            QMatrixInner::QMi16U8(m) => m.nnz(),
            QMatrixInner::QMi16U16(m) => m.nnz(),
            QMatrixInner::QMf32U8(m) => m.nnz(),
            QMatrixInner::QMf32U16(m) => m.nnz(),
            QMatrixInner::QMf64U8(m) => m.nnz(),
            QMatrixInner::QMf64U16(m) => m.nnz(),
            QMatrixInner::QMc32U8(m) => m.nnz(),
            QMatrixInner::QMc32U16(m) => m.nnz(),
            QMatrixInner::QMc64U8(m) => m.nnz(),
            QMatrixInner::QMc64U16(m) => m.nnz(),
        }
    }

    /// NumPy dtype name for the matrix element type (e.g. `"float64"`).
    pub fn dtype_name(&self) -> &'static str {
        match self {
            QMatrixInner::QMi8U8(_) | QMatrixInner::QMi8U16(_) => "int8",
            QMatrixInner::QMi16U8(_) | QMatrixInner::QMi16U16(_) => "int16",
            QMatrixInner::QMf32U8(_) | QMatrixInner::QMf32U16(_) => "float32",
            QMatrixInner::QMf64U8(_) | QMatrixInner::QMf64U16(_) => "float64",
            QMatrixInner::QMc32U8(_) | QMatrixInner::QMc32U16(_) => "complex64",
            QMatrixInner::QMc64U8(_) | QMatrixInner::QMc64U16(_) => "complex128",
        }
    }

    pub fn dot<V: crate::primitive::Primitive>(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        crate::with_qmatrix!(self, _M, _C, mat, {
            mat.dot(overwrite, coeff, input, output)
        })
    }

    pub fn dot_transpose<V: crate::primitive::Primitive>(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: &[V],
        output: &mut [V],
    ) -> Result<(), QuSpinError> {
        crate::with_qmatrix!(self, _M, _C, mat, {
            mat.dot_transpose(overwrite, coeff, input, output)
        })
    }

    pub fn dot_many<V: crate::primitive::Primitive>(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: ArrayView2<'_, V>,
        output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError> {
        crate::with_qmatrix!(self, _M, _C, mat, {
            mat.dot_many(overwrite, coeff, input, output)
        })
    }

    pub fn dot_transpose_many<V: crate::primitive::Primitive>(
        &self,
        overwrite: bool,
        coeff: &[V],
        input: ArrayView2<'_, V>,
        output: ArrayViewMut2<'_, V>,
    ) -> Result<(), QuSpinError> {
        crate::with_qmatrix!(self, _M, _C, mat, {
            mat.dot_transpose_many(overwrite, coeff, input, output)
        })
    }

    /// Element-wise addition.  Both operands must have the same dtype.
    pub fn try_add(self, rhs: Self) -> Result<Self, QuSpinError> {
        macro_rules! add_variant {
            ($a:expr, $b:expr, $variant:ident) => {
                match $b {
                    QMatrixInner::$variant(r) => Ok(QMatrixInner::$variant($a + r)),
                    _ => Err(QuSpinError::ValueError(
                        "QMatrix dtype mismatch in addition".to_string(),
                    )),
                }
            };
        }
        match self {
            QMatrixInner::QMi8U8(l) => add_variant!(l, rhs, QMi8U8),
            QMatrixInner::QMi8U16(l) => add_variant!(l, rhs, QMi8U16),
            QMatrixInner::QMi16U8(l) => add_variant!(l, rhs, QMi16U8),
            QMatrixInner::QMi16U16(l) => add_variant!(l, rhs, QMi16U16),
            QMatrixInner::QMf32U8(l) => add_variant!(l, rhs, QMf32U8),
            QMatrixInner::QMf32U16(l) => add_variant!(l, rhs, QMf32U16),
            QMatrixInner::QMf64U8(l) => add_variant!(l, rhs, QMf64U8),
            QMatrixInner::QMf64U16(l) => add_variant!(l, rhs, QMf64U16),
            QMatrixInner::QMc32U8(l) => add_variant!(l, rhs, QMc32U8),
            QMatrixInner::QMc32U16(l) => add_variant!(l, rhs, QMc32U16),
            QMatrixInner::QMc64U8(l) => add_variant!(l, rhs, QMc64U8),
            QMatrixInner::QMc64U16(l) => add_variant!(l, rhs, QMc64U16),
        }
    }

    /// Materialize as standard CSR: sum over cindices weighted by `coeff`.
    ///
    /// Returns `(indptr, indices, data)` where each is a `Vec` with i64/i64/Complex<f64>
    /// element type respectively.  Entries at the same `(row, col)` are merged.
    ///
    /// If `drop_zeros` is true, entries whose magnitude is below a small tolerance
    /// are omitted from the output.
    pub fn materialize(
        &self,
        coeff: &[num_complex::Complex<f64>],
        drop_zeros: bool,
    ) -> Result<CsrTriple, QuSpinError> {
        crate::with_qmatrix!(self, _M, _C, mat, {
            if coeff.len() != mat.num_coeff() {
                return Err(QuSpinError::ValueError(format!(
                    "coeff length {} != num_coeff {}",
                    coeff.len(),
                    mat.num_coeff()
                )));
            }
            let dim = mat.dim();
            let mut indptr = Vec::with_capacity(dim + 1);
            let mut indices: Vec<i64> = Vec::with_capacity(mat.nnz());
            let mut data: Vec<num_complex::Complex<f64>> = Vec::with_capacity(mat.nnz());
            indptr.push(0i64);
            for r in 0..dim {
                let row = mat.row(r);
                let mut col_it = row.iter().peekable();
                while let Some(first) = col_it.next() {
                    let col = first.col;
                    let mut acc = first.value.to_complex() * coeff[first.cindex.as_usize()];
                    while col_it.peek().map(|e| e.col == col).unwrap_or(false) {
                        let e = col_it.next().unwrap();
                        acc += e.value.to_complex() * coeff[e.cindex.as_usize()];
                    }
                    if !drop_zeros || acc.norm() > 4.0 * f64::EPSILON {
                        indices.push(col.as_usize() as i64);
                        data.push(acc);
                    }
                }
                indptr.push(indices.len() as i64);
            }
            Ok((indptr, indices, data))
        })
    }

    /// Element-wise subtraction.  Both operands must have the same dtype.
    pub fn try_sub(self, rhs: Self) -> Result<Self, QuSpinError> {
        macro_rules! sub_variant {
            ($a:expr, $b:expr, $variant:ident) => {
                match $b {
                    QMatrixInner::$variant(r) => Ok(QMatrixInner::$variant($a - r)),
                    _ => Err(QuSpinError::ValueError(
                        "QMatrix dtype mismatch in subtraction".to_string(),
                    )),
                }
            };
        }
        match self {
            QMatrixInner::QMi8U8(l) => sub_variant!(l, rhs, QMi8U8),
            QMatrixInner::QMi8U16(l) => sub_variant!(l, rhs, QMi8U16),
            QMatrixInner::QMi16U8(l) => sub_variant!(l, rhs, QMi16U8),
            QMatrixInner::QMi16U16(l) => sub_variant!(l, rhs, QMi16U16),
            QMatrixInner::QMf32U8(l) => sub_variant!(l, rhs, QMf32U8),
            QMatrixInner::QMf32U16(l) => sub_variant!(l, rhs, QMf32U16),
            QMatrixInner::QMf64U8(l) => sub_variant!(l, rhs, QMf64U8),
            QMatrixInner::QMf64U16(l) => sub_variant!(l, rhs, QMf64U16),
            QMatrixInner::QMc32U8(l) => sub_variant!(l, rhs, QMc32U8),
            QMatrixInner::QMc32U16(l) => sub_variant!(l, rhs, QMc32U16),
            QMatrixInner::QMc64U8(l) => sub_variant!(l, rhs, QMc64U8),
            QMatrixInner::QMc64U16(l) => sub_variant!(l, rhs, QMc64U16),
        }
    }
}

// ---------------------------------------------------------------------------
// Build helpers
// ---------------------------------------------------------------------------

/// Build a `QMatrixInner` from a concrete `Operator<u8>` and a `SpaceInner`,
/// selecting the element type from `dtype`.
macro_rules! dispatch_dtype_u8 {
    ($h:expr, $space:expr, $dtype:expr) => {
        match $dtype {
            ValueDType::Int8 => QMatrixInner::QMi8U8(build_from_space($h, $space)),
            ValueDType::Int16 => QMatrixInner::QMi16U8(build_from_space($h, $space)),
            ValueDType::Float32 => QMatrixInner::QMf32U8(build_from_space($h, $space)),
            ValueDType::Float64 => QMatrixInner::QMf64U8(build_from_space($h, $space)),
            ValueDType::Complex64 => QMatrixInner::QMc32U8(build_from_space($h, $space)),
            ValueDType::Complex128 => QMatrixInner::QMc64U8(build_from_space($h, $space)),
        }
    };
}

/// Build a `QMatrixInner` from a concrete `Operator<u16>` and a `SpaceInner`,
/// selecting the element type from `dtype`.
macro_rules! dispatch_dtype_u16 {
    ($h:expr, $space:expr, $dtype:expr) => {
        match $dtype {
            ValueDType::Int8 => QMatrixInner::QMi8U16(build_from_space($h, $space)),
            ValueDType::Int16 => QMatrixInner::QMi16U16(build_from_space($h, $space)),
            ValueDType::Float32 => QMatrixInner::QMf32U16(build_from_space($h, $space)),
            ValueDType::Float64 => QMatrixInner::QMf64U16(build_from_space($h, $space)),
            ValueDType::Complex64 => QMatrixInner::QMc32U16(build_from_space($h, $space)),
            ValueDType::Complex128 => QMatrixInner::QMc64U16(build_from_space($h, $space)),
        }
    };
}

impl QMatrixInner {
    /// Build from a Pauli/hardcore operator and a basis space.
    pub fn build_hardcore(
        ham: &HardcoreOperatorInner,
        space: &SpaceInner,
        dtype: ValueDType,
    ) -> Self {
        match ham {
            HardcoreOperatorInner::Ham8(h) => dispatch_dtype_u8!(h, space, dtype),
            HardcoreOperatorInner::Ham16(h) => dispatch_dtype_u16!(h, space, dtype),
        }
    }

    /// Build from a bond operator and a basis space.
    pub fn build_bond(ham: &BondOperatorInner, space: &SpaceInner, dtype: ValueDType) -> Self {
        match ham {
            BondOperatorInner::Ham8(h) => dispatch_dtype_u8!(h, space, dtype),
            BondOperatorInner::Ham16(h) => dispatch_dtype_u16!(h, space, dtype),
        }
    }

    /// Build from a boson operator and a basis space.
    pub fn build_boson(ham: &BosonOperatorInner, space: &SpaceInner, dtype: ValueDType) -> Self {
        match ham {
            BosonOperatorInner::Ham8(h) => dispatch_dtype_u8!(h, space, dtype),
            BosonOperatorInner::Ham16(h) => dispatch_dtype_u16!(h, space, dtype),
        }
    }

    /// Build from a fermion operator and a basis space.
    pub fn build_fermion(
        ham: &FermionOperatorInner,
        space: &SpaceInner,
        dtype: ValueDType,
    ) -> Self {
        match ham {
            FermionOperatorInner::Ham8(h) => dispatch_dtype_u8!(h, space, dtype),
            FermionOperatorInner::Ham16(h) => dispatch_dtype_u16!(h, space, dtype),
        }
    }
}

// ---------------------------------------------------------------------------
// IntoQMatrixInner
// ---------------------------------------------------------------------------

/// Convert a concrete `QMatrix<V, i64, C>` to the type-erased `QMatrixInner`.
pub trait IntoQMatrixInner {
    fn into_qmatrix_inner(self) -> QMatrixInner;
}

macro_rules! impl_into_qmatrix_inner {
    ($V:ty, $C:ty, $variant:ident) => {
        impl IntoQMatrixInner for QMatrix<$V, i64, $C> {
            #[inline]
            fn into_qmatrix_inner(self) -> QMatrixInner {
                QMatrixInner::$variant(self)
            }
        }
    };
}

impl_into_qmatrix_inner!(i8, u8, QMi8U8);
impl_into_qmatrix_inner!(i8, u16, QMi8U16);
impl_into_qmatrix_inner!(i16, u8, QMi16U8);
impl_into_qmatrix_inner!(i16, u16, QMi16U16);
impl_into_qmatrix_inner!(f32, u8, QMf32U8);
impl_into_qmatrix_inner!(f32, u16, QMf32U16);
impl_into_qmatrix_inner!(f64, u8, QMf64U8);
impl_into_qmatrix_inner!(f64, u16, QMf64U16);
impl_into_qmatrix_inner!(num_complex::Complex<f32>, u8, QMc32U8);
impl_into_qmatrix_inner!(num_complex::Complex<f32>, u16, QMc32U16);
impl_into_qmatrix_inner!(num_complex::Complex<f64>, u8, QMc64U8);
impl_into_qmatrix_inner!(num_complex::Complex<f64>, u16, QMc64U16);

// ---------------------------------------------------------------------------
// with_qmatrix! macro
// ---------------------------------------------------------------------------

/// Match on a `&QMatrixInner`, injecting type aliases `$M` (stored element
/// type) and `$C` (cindex type), and binding `$mat` to the inner `QMatrix`
/// reference.
#[macro_export]
macro_rules! with_qmatrix {
    ($inner:expr, $M:ident, $C:ident, $mat:ident, $body:block) => {
        match $inner {
            $crate::qmatrix::dispatch::QMatrixInner::QMi8U8($mat) => {
                type $M = i8;
                type $C = u8;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMi8U16($mat) => {
                type $M = i8;
                type $C = u16;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMi16U8($mat) => {
                type $M = i16;
                type $C = u8;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMi16U16($mat) => {
                type $M = i16;
                type $C = u16;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMf32U8($mat) => {
                type $M = f32;
                type $C = u8;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMf32U16($mat) => {
                type $M = f32;
                type $C = u16;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMf64U8($mat) => {
                type $M = f64;
                type $C = u8;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMf64U16($mat) => {
                type $M = f64;
                type $C = u16;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMc32U8($mat) => {
                type $M = ::num_complex::Complex<f32>;
                type $C = u8;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMc32U16($mat) => {
                type $M = ::num_complex::Complex<f32>;
                type $C = u16;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMc64U8($mat) => {
                type $M = ::num_complex::Complex<f64>;
                type $C = u8;
                $body
            }
            $crate::qmatrix::dispatch::QMatrixInner::QMc64U16($mat) => {
                type $M = ::num_complex::Complex<f64>;
                type $C = u16;
                $body
            }
        }
    };
}
