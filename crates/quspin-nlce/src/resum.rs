//! Resummation of the bare partial sums.
//!
//! Phase 1 provides only the trait and the identity ([`Bare`]); Wynn-ε and
//! Euler transformations implement the same trait later.

use crate::error::NlceError;
use crate::property::Property;

/// Accelerates convergence of a sequence of partial sums (ascending order).
pub trait Resummation<P: Property> {
    /// Estimate of the limit of `partial_sums`.
    ///
    /// # Errors
    /// `InvalidInput` if the sequence is too short for the method.
    fn resum(&self, partial_sums: &[P]) -> Result<P, NlceError>;
}

/// No resummation: returns the highest-order partial sum.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bare;

impl<P: Property> Resummation<P> for Bare {
    fn resum(&self, partial_sums: &[P]) -> Result<P, NlceError> {
        partial_sums
            .last()
            .cloned()
            .ok_or_else(|| NlceError::InvalidInput("no partial sums to resum".into()))
    }
}
