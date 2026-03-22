pub mod build;
pub mod dispatch;
pub mod matrix;
pub mod ops;

pub use dispatch::{IntoQMatrixInner, QMatrixInner};
pub use matrix::{CIndex, Entry, Index, QMatrix};
