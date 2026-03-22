use pyo3::PyErr;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use quspin_core::error::QuSpinError as CoreError;

/// Local newtype wrapper around `quspin-core`'s error type.
///
/// Required by Rust's orphan rules: `From<CoreError> for PyErr` cannot be
/// implemented directly because both types are foreign to this crate.
/// Wrapping in a local type breaks the orphan restriction.
///
/// Usage at PyO3 call sites:
/// ```rust,ignore
/// core_function().map_err(Error::from)?
/// ```
pub struct Error(pub CoreError);

impl From<CoreError> for Error {
    fn from(e: CoreError) -> Self {
        Error(e)
    }
}

impl From<Error> for PyErr {
    fn from(e: Error) -> Self {
        match e.0 {
            CoreError::RuntimeError(msg) => PyRuntimeError::new_err(msg),
            CoreError::ValueError(msg) => PyValueError::new_err(msg),
            CoreError::IndexError(msg) => PyIndexError::new_err(msg),
        }
    }
}
