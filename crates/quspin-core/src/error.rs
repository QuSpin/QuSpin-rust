use thiserror::Error;

/// Library-wide error type.  Maps directly to Python exception types at the
/// PyO3 boundary via `impl From<QuSpinError> for PyErr` in `quspin-py`.
#[derive(Error, Debug, Clone)]
pub enum QuSpinError {
    #[error("RuntimeError: {0}")]
    RuntimeError(String),

    #[error("ValueError: {0}")]
    ValueError(String),

    #[error("IndexError: {0}")]
    IndexError(String),
}
