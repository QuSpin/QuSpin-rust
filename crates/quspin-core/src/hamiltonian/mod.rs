pub mod hardcore;

pub use hardcore::{HardcoreHamiltonian, HardcoreHamiltonianInner, HardcoreOp, OpEntry};

use crate::error::QuSpinError;

/// Operator types that can be parsed from a single ASCII character.
///
/// Implement this trait on an operator enum to plug into the shared
/// `parse_term` / `parse_coupling` / `parse_ops` pipeline in any FFI crate.
pub trait ParseOp: Sized {
    /// Convert a single character to `Self`.
    ///
    /// Returns `QuSpinError::ValueError` for unrecognised characters.  The
    /// error message should name the valid characters for this operator type.
    fn from_char(ch: char) -> Result<Self, QuSpinError>;
}
