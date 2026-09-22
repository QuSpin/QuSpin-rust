//! Error type for the NLCE pipeline.

use quspin_core::QuSpinError;
use std::fmt;

/// Errors raised while generating, solving, or combining clusters.
#[derive(Debug)]
pub enum NlceError {
    /// An error propagated from QuSpin-rust (basis, operator, matrix, ED).
    QuSpin(QuSpinError),
    /// Invalid user input (bad temperature grid, malformed graph, …).
    InvalidInput(String),
    /// The cluster DAG is inconsistent (e.g. a sub-cluster referenced before
    /// it was generated, or a missing cluster property).
    InconsistentClusters(String),
    /// A combination of model, solver, and observables that is not supported.
    Unsupported(String),
}

impl fmt::Display for NlceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QuSpin(e) => write!(f, "QuSpin error: {e}"),
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::InconsistentClusters(s) => write!(f, "inconsistent clusters: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for NlceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::QuSpin(e) => Some(e),
            _ => None,
        }
    }
}

impl From<QuSpinError> for NlceError {
    fn from(e: QuSpinError) -> Self {
        Self::QuSpin(e)
    }
}
