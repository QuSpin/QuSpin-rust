//! Row-range CSR materialisation that bypasses `QMatrix`.
//!
//! Each call walks rows `[row_start, row_end)` of the operator + basis pair
//! directly, returning `(indptr, indices, data)` in the layout petsc4py's
//! `Mat.setValuesCSR(I, J, V)` expects.  Memory is bounded by the slab —
//! useful when each MPI rank only needs its locally-owned rows of a matrix
//! that's otherwise too large to materialise globally.
//!
//! See `docs/superpowers/specs/2026-05-03-petsc4py-csr-slab-design.md`.
