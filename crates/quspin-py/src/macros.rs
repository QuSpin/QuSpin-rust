// `with_value_dtype!`, `with_cindex_dtype!`, and `select_b_for_n_sites!` now
// live in `quspin-core` and are re-exported here for convenience.
//
// `select_b_for_n_sites!` requires a caller-supplied overflow expression as
// its third argument.  In `quspin-py` this is always a `PyValueError`:
//
//   select_b_for_n_sites!(n, B,
//       return Err(pyo3::exceptions::PyValueError::new_err(format!(
//           "n_sites={n} exceeds the maximum supported value of 8192"
//       ))),
//       { ... }
//   );
