pub mod build;
pub mod dispatch;
pub mod matrix;
pub mod ops;
// Internal: `RowSource` / `RawEntry` and the `&dyn RowSource` boundary are a
// code-size optimisation, not an interface. Keeping the module crate-private
// means external callers cannot come to depend on the runtime-dispatch shape
// — only the typed builders in `build` are public.
pub(crate) mod rowsource;

pub use build::build_from_space;
pub use dispatch::{IntoQMatrixInner, QMatrixInner};
pub use matrix::{CIndex, Entry, Index, QMatrix};
