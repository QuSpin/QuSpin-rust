use super::matrix::PARALLEL_DIM_THRESHOLD;
use super::{CIndex, Entry, Index, QMatrix};
use crate::primitive::Primitive;
use rayon::prelude::*;
use std::ops::{Add, Sub};

// ---------------------------------------------------------------------------
// Binary operations (add / sub)
// ---------------------------------------------------------------------------

/// Sorted row-merge of two `QMatrix` instances under a binary operator.
///
/// Both matrices must have the same `dim`.  Within each row, entries are
/// assumed sorted by (col, cindex) — as guaranteed by `QMatrix::from_csr`.
/// The merge walks both rows simultaneously (like a merge-step in merge sort),
/// combining entries at the same (col, cindex) key via `op` and dropping
/// zeros.
///
/// Mirrors `binary_op` from `qmatrix.hpp` (sequential single-thread branch).
/// Merge a single row from `lhs` and `rhs` under `op`, returning the entries.
fn merge_row<M, I, C, Op>(
    op: &Op,
    lhs_row: &[Entry<M, I, C>],
    rhs_row: &[Entry<M, I, C>],
) -> Vec<Entry<M, I, C>>
where
    M: Primitive + PartialEq,
    I: Index,
    C: CIndex,
    Op: Fn(M, M) -> M,
{
    let zero = M::default();
    let mut entries = Vec::new();
    let mut li = lhs_row.iter().peekable();
    let mut ri = rhs_row.iter().peekable();

    while let (Some(le), Some(re)) = (li.peek(), ri.peek()) {
        let result = if le.lt_col_cindex(re) {
            let e = *le;
            li.next();
            let v = op(e.value, zero);
            if v != zero {
                Some(Entry::new(v, e.col, e.cindex))
            } else {
                None
            }
        } else if re.lt_col_cindex(le) {
            let e = *re;
            ri.next();
            let v = op(zero, e.value);
            if v != zero {
                Some(Entry::new(v, e.col, e.cindex))
            } else {
                None
            }
        } else {
            let le = *le;
            let re = *re;
            li.next();
            ri.next();
            let v = op(le.value, re.value);
            if v != zero {
                Some(Entry::new(v, le.col, le.cindex))
            } else {
                None
            }
        };
        if let Some(e) = result {
            entries.push(e);
        }
    }

    for e in li {
        let v = op(e.value, zero);
        if v != zero {
            entries.push(Entry::new(v, e.col, e.cindex));
        }
    }
    for e in ri {
        let v = op(zero, e.value);
        if v != zero {
            entries.push(Entry::new(v, e.col, e.cindex));
        }
    }
    entries
}

fn binary_op<M, I, C, Op>(
    op: Op,
    lhs: &QMatrix<M, I, C>,
    rhs: &QMatrix<M, I, C>,
) -> QMatrix<M, I, C>
where
    M: Primitive + PartialEq,
    I: Index,
    C: CIndex,
    Op: Fn(M, M) -> M + Sync,
{
    assert_eq!(lhs.dim(), rhs.dim(), "QMatrix dimensions must match");

    let dim = lhs.dim();
    let build_row = |r: usize| merge_row(&op, lhs.row(r), rhs.row(r));

    let rows: Vec<Vec<Entry<M, I, C>>> = if dim >= PARALLEL_DIM_THRESHOLD {
        (0..dim).into_par_iter().map(build_row).collect()
    } else {
        (0..dim).map(build_row).collect()
    };

    let total_nnz: usize = rows.iter().map(|r| r.len()).sum();
    let mut indptr = Vec::with_capacity(dim + 1);
    let mut data = Vec::with_capacity(total_nnz);
    indptr.push(I::from_usize(0));
    for row in rows {
        data.extend_from_slice(&row);
        indptr.push(I::from_usize(data.len()));
    }
    QMatrix::from_csr(indptr, data)
}

impl<M, I, C> Add for QMatrix<M, I, C>
where
    M: Primitive + PartialEq + Add<Output = M>,
    I: Index,
    C: CIndex,
{
    type Output = QMatrix<M, I, C>;

    fn add(self, rhs: Self) -> Self::Output {
        binary_op(|a, b| a + b, &self, &rhs)
    }
}

impl<M, I, C> Sub for QMatrix<M, I, C>
where
    M: Primitive + PartialEq + Sub<Output = M>,
    I: Index,
    C: CIndex,
{
    type Output = QMatrix<M, I, C>;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_op(|a, b| a - b, &self, &rhs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mat_a() -> QMatrix<f64, i64, u8> {
        // [[1, 2], [3, 0]]
        let indptr = vec![0i64, 2, 3];
        let data = vec![
            Entry::new(1.0f64, 0i64, 0u8),
            Entry::new(2.0f64, 1i64, 0u8),
            Entry::new(3.0f64, 0i64, 0u8),
        ];
        QMatrix::from_csr(indptr, data)
    }

    fn mat_b() -> QMatrix<f64, i64, u8> {
        // [[0, 1], [1, 4]]
        let indptr = vec![0i64, 1, 3];
        let data = vec![
            Entry::new(1.0f64, 1i64, 0u8),
            Entry::new(1.0f64, 0i64, 0u8),
            Entry::new(4.0f64, 1i64, 0u8),
        ];
        QMatrix::from_csr(indptr, data)
    }

    #[test]
    fn add_matrices() {
        // A + B = [[1,3],[4,4]]
        let a = mat_a();
        let b = mat_b();
        let c = a + b;
        assert_eq!(c.dim(), 2);

        let r0 = c.row(0);
        assert_eq!(r0.len(), 2);
        assert!((r0[0].value - 1.0).abs() < 1e-12); // col 0
        assert!((r0[1].value - 3.0).abs() < 1e-12); // col 1

        let r1 = c.row(1);
        assert_eq!(r1.len(), 2);
        assert!((r1[0].value - 4.0).abs() < 1e-12); // col 0
        assert!((r1[1].value - 4.0).abs() < 1e-12); // col 1
    }

    #[test]
    fn sub_matrices() {
        // A - B = [[1,1],[2,-4]]
        let a = mat_a();
        let b = mat_b();
        let c = a - b;

        let r0 = c.row(0);
        assert_eq!(r0.len(), 2);
        assert!((r0[0].value - 1.0).abs() < 1e-12);
        assert!((r0[1].value - 1.0).abs() < 1e-12);

        let r1 = c.row(1);
        assert_eq!(r1.len(), 2);
        assert!((r1[0].value - 2.0).abs() < 1e-12);
        assert!((r1[1].value - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn add_cancels_to_zero_drops_entry() {
        // A + (-A) should have no non-zero entries
        let indptr = vec![0i64, 1];
        let data_a = vec![Entry::new(1.0f64, 0i64, 0u8)];
        let data_neg = vec![Entry::new(-1.0f64, 0i64, 0u8)];
        let a = QMatrix::<f64, i64, u8>::from_csr(indptr.clone(), data_a);
        let neg_a = QMatrix::<f64, i64, u8>::from_csr(indptr, data_neg);
        let sum = a + neg_a;
        assert_eq!(sum.nnz(), 0);
    }

    #[test]
    fn dot_after_add() {
        // (A+B)|v⟩ = A|v⟩ + B|v⟩
        let _a = mat_a();
        let _b = mat_b();
        let coeff = vec![1.0f64];
        let v = vec![1.0f64, 1.0];

        let mut av = vec![0.0f64; 2];
        let mut bv = vec![0.0f64; 2];
        QMatrix::from_csr(
            vec![0i64, 2, 3],
            vec![
                Entry::new(1.0f64, 0i64, 0u8),
                Entry::new(2.0f64, 1i64, 0u8),
                Entry::new(3.0f64, 0i64, 0u8),
            ],
        )
        .dot(true, &coeff, &v, &mut av)
        .unwrap();
        QMatrix::from_csr(
            vec![0i64, 1, 3],
            vec![
                Entry::new(1.0f64, 1i64, 0u8),
                Entry::new(1.0f64, 0i64, 0u8),
                Entry::new(4.0f64, 1i64, 0u8),
            ],
        )
        .dot(true, &coeff, &v, &mut bv)
        .unwrap();

        let c = mat_a() + mat_b();
        let mut cv = vec![0.0f64; 2];
        c.dot(true, &coeff, &v, &mut cv).unwrap();

        for i in 0..2 {
            assert!((cv[i] - (av[i] + bv[i])).abs() < 1e-12);
        }
    }

    #[test]
    fn binary_op_parallel_sub_cancellation() {
        // Build two identical tridiagonal matrices with dim=300, subtract.
        let dim = 300;
        let mut indptr = Vec::with_capacity(dim + 1);
        let mut data = Vec::new();
        indptr.push(0i64);
        for r in 0..dim {
            if r > 0 {
                data.push(Entry::new(1.0f64, (r - 1) as i64, 0u8));
            }
            if r + 1 < dim {
                data.push(Entry::new(1.0f64, (r + 1) as i64, 0u8));
            }
            indptr.push(data.len() as i64);
        }
        let a = QMatrix::<f64, i64, u8>::from_csr(indptr.clone(), data.clone());
        let b = QMatrix::<f64, i64, u8>::from_csr(indptr, data);
        let c = a - b;
        assert_eq!(c.nnz(), 0, "A - A should be zero matrix");
    }
}
