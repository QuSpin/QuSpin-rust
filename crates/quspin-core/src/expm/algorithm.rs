//! Core Taylor-series partition algorithm for computing `exp(a·A)·f`.
//!
//! Ports the C++ `expm_multiply` / `expm_multiply_batch` kernels from
//! `parallel-sparse-tools`, which implement the partitioned Taylor method of
//! Al-Mohy & Higham (2011).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};

use ndarray::{ArrayViewMut1, ArrayViewMut2, Axis};

use crate::error::QuSpinError;

use super::compute::ExpmComputation;
use super::linear_operator::LinearOperator;

/// Minimum dimension for the persistent-thread parallel path.
///
/// Matrices smaller than this are handled by [`expm_multiply`] to avoid
/// thread-pool wake-up overhead dominating the matvec cost.
pub const PAR_THRESHOLD: usize = 256;

// ---------------------------------------------------------------------------
// RawSlice — raw-pointer slice wrapper for cross-thread sharing
// ---------------------------------------------------------------------------

/// Wraps a raw mutable slice pointer for sharing across scoped threads.
///
/// # Safety invariant
/// The caller guarantees that:
/// - Write accesses from different threads target **disjoint** index ranges.
/// - `Barrier::wait()` separates every write from any subsequent read by
///   another thread (provides the required memory ordering).
struct RawSlice<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: `RawSlice` is just a raw pointer + length — no ownership semantics.
// The caller upholds the aliasing invariants described in the struct's doc.
unsafe impl<T: Send> Send for RawSlice<T> {}
unsafe impl<T: Sync> Sync for RawSlice<T> {}

// `Copy` is safe: copying a raw pointer doesn't affect ownership or aliasing.
impl<T> Copy for RawSlice<T> {}
impl<T> Clone for RawSlice<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> RawSlice<T> {
    fn new(s: &mut [T]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }

    /// Return an immutable view of the full slice.
    ///
    /// # Safety
    /// No other thread may be writing to any index at the time of the call.
    #[inline]
    unsafe fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Return a mutable view of `range` (absolute indices into the full slice).
    ///
    /// # Safety
    /// No other thread may access `range` concurrently.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    unsafe fn subslice_mut(&self, range: std::ops::Range<usize>) -> &mut [T] {
        debug_assert!(range.end <= self.len);
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.end - range.start)
        }
    }

    /// Write `val` to absolute index `idx`.
    ///
    /// # Safety
    /// No other thread may access `idx` concurrently.
    #[inline]
    unsafe fn write(&self, idx: usize, val: T) {
        unsafe { self.ptr.add(idx).write(val) };
    }

    /// Read from absolute index `idx`.
    ///
    /// # Safety
    /// No other thread may be writing to `idx` at the time of the call.
    #[inline]
    unsafe fn read(&self, idx: usize) -> T
    where
        T: Copy,
    {
        unsafe { self.ptr.add(idx).read() }
    }
}

// ---------------------------------------------------------------------------
// Scalar variant
// ---------------------------------------------------------------------------

/// Compute `exp(a·A) · f` in-place (single vector).
///
/// Uses the partitioned Taylor expansion:
///
/// ```text
/// exp(a·A) = exp(a·μ) · [exp(a·(A−μI)/s)]^s
///          ≈ exp(a·μ) · [Σ_{j=0}^{m_star} (a·(A−μI)/(j·s))^j]^s
/// ```
///
/// # Arguments
/// - `op`      — linear operator implementing `A · x`
/// - `a`       — global scalar factor
/// - `mu`      — diagonal shift μ (usually `trace(A)/n`)
/// - `s`       — partition count (scaling factor)
/// - `m_star`  — Taylor truncation order per partition
/// - `tol`     — convergence tolerance (typically `V::machine_eps()`)
/// - `f`       — input/output vector, length = `op.dim()`
/// - `work`    — scratch buffer, length ≥ `2 * op.dim()`
///
/// # Errors
/// Returns `ValueError` if buffer lengths are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn expm_multiply<V, Op>(
    op: &Op,
    a: V,
    mu: V,
    s: usize,
    m_star: usize,
    tol: V::Real,
    mut f: ArrayViewMut1<'_, V>,
    work: &mut [V],
) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    let n = op.dim();

    if f.len() != n {
        return Err(QuSpinError::ValueError(format!(
            "f.len()={} must equal op.dim()={}",
            f.len(),
            n
        )));
    }
    if work.len() < 2 * n {
        return Err(QuSpinError::ValueError(format!(
            "work.len()={} must be >= 2 * op.dim()={}",
            work.len(),
            2 * n
        )));
    }

    if n == 0 || s == 0 {
        return Ok(());
    }

    // Split scratch buffer: b1 = current Taylor term, tmp = matvec output.
    let (b1, tmp) = work[..2 * n].split_at_mut(n);

    // η = exp(a·μ / s) — applied once per outer partition.
    let inv_s = V::real_from_f64(1.0 / s as f64);
    let eta = (a * mu * V::from_real(inv_s)).exp_val();

    // B = F = f  (initial: Taylor term B1 = accumulated sum F)
    for (b, &fk) in b1.iter_mut().zip(f.iter()) {
        *b = fk;
    }

    for _i in 0..s {
        // c1 = ‖B‖_∞  (inf-norm of the current Taylor term / starting vector)
        let mut c1 = inf_norm(b1);

        'taylor: for j in 1..=m_star {
            // tmp = A · B  (overwrite=true zeros tmp first)
            op.dot(true, b1, tmp)?;

            // scale = a / (j · s)
            let scale = a * V::from_real(V::real_from_f64(1.0 / (j * s) as f64));

            let mut c2 = V::Real::default();
            let mut c3 = V::Real::default();

            // tmp[k] = scale · (tmp[k] − μ · B[k])      (new Taylor term)
            // f[k]  += tmp[k]
            for k in 0..n {
                tmp[k] = scale * (tmp[k] - mu * b1[k]);
                f[k] += tmp[k];
                let abs_tmp = tmp[k].abs_val();
                let abs_f = f[k].abs_val();
                if abs_tmp > c2 {
                    c2 = abs_tmp;
                }
                if abs_f > c3 {
                    c3 = abs_f;
                }
            }

            // Convergence: latest term negligible relative to accumulated sum.
            if c1 + c2 <= tol * c3 {
                break 'taylor;
            }

            c1 = c2;
            // B = tmp (advance to next Taylor term)
            b1.copy_from_slice(tmp);
        }

        // F *= η  then  B = F  (reset starting point for next partition)
        for fk in f.iter_mut() {
            *fk *= eta;
        }
        for (b, &fk) in b1.iter_mut().zip(f.iter()) {
            *b = fk;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Parallel scalar variant (persistent-thread pool)
// ---------------------------------------------------------------------------

/// Compute `exp(a·A) · f` in-place using a persistent thread pool.
///
/// Spawns `n_threads` threads once via [`std::thread::scope`].  Each thread
/// owns a static row range `[begin, end)` and communicates via a shared
/// [`Barrier`].  This eliminates per-Taylor-step rayon fork/join overhead.
///
/// The Taylor loop within each outer partition (`s`) runs four phases per
/// step `j`:
///
/// 1. **Matvec** — each thread calls `op.dot_chunk(input=b1, output=tmp, rows=begin..end)`.
/// 2. **Element update** — each thread transforms `tmp[begin..end]` and
///    accumulates local inf-norms `local_c2`, `local_c3`.
/// 3. **Convergence check** (thread 0 only) — reduces per-thread norms and
///    sets the shared `exit_flag` atomically.
/// 4. **State transfer** — each thread copies `tmp[begin..end] → b1[begin..end]`,
///    or breaks if `exit_flag` is set.
///
/// All phases are separated by `barrier.wait()` calls; the `Barrier` also
/// provides the required memory ordering between writes and subsequent reads.
///
/// # Errors
/// Returns `ValueError` if buffer lengths are inconsistent.
///
/// # Panics
/// Panics if an internal `dot_chunk` call fails (only possible on dimension
/// mismatch, which is validated before threads are spawned).
#[allow(clippy::too_many_arguments)]
pub fn expm_multiply_par<V, Op>(
    op: &Op,
    a: V,
    mu: V,
    s: usize,
    m_star: usize,
    tol: V::Real,
    f: ArrayViewMut1<'_, V>,
    work: &mut [V],
) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    let n = op.dim();

    if f.len() != n {
        return Err(QuSpinError::ValueError(format!(
            "f.len()={} must equal op.dim()={}",
            f.len(),
            n
        )));
    }
    if work.len() < 2 * n {
        return Err(QuSpinError::ValueError(format!(
            "work.len()={} must be >= 2 * op.dim()={}",
            work.len(),
            2 * n
        )));
    }

    if n == 0 || s == 0 {
        return Ok(());
    }

    // Split scratch: b1 = current Taylor term, tmp = matvec output.
    let (b1_buf, tmp_buf) = work[..2 * n].split_at_mut(n);

    // b1 = f  (initialise the Taylor accumulator from the input)
    for (b, &fk) in b1_buf.iter_mut().zip(f.iter()) {
        *b = fk;
    }

    let inv_s = V::real_from_f64(1.0 / s as f64);
    let eta = (a * mu * V::from_real(inv_s)).exp_val();

    // Clamp thread count to n so every thread gets at least one row.
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(n);
    let chunk = n.div_ceil(n_threads);

    // Shared synchronization primitives.
    let barrier = Arc::new(Barrier::new(n_threads));
    let exit_flag = Arc::new(AtomicBool::new(false));

    // Per-thread norm slots — thread `tid` writes slot `tid`; thread 0 reads
    // all slots after the Phase-3 barrier.  These are small (n_threads elements)
    // and accessed under barrier discipline, so RawSlice is appropriate.
    let mut norms_c2 = vec![V::Real::default(); n_threads];
    let mut norms_c3 = vec![V::Real::default(); n_threads];
    let raw_c2 = RawSlice::new(&mut norms_c2);
    let raw_c3 = RawSlice::new(&mut norms_c3);

    // b1 transitions between "all threads read" (Phase 1) and "each thread
    // writes its chunk" (Phase 4), separated by barriers.  Rust cannot reason
    // about barrier-based sequencing statically, so RawSlice is used here.
    let raw_b1 = RawSlice::new(b1_buf);

    // Build exactly `n_threads` disjoint row ranges so the number of spawned
    // workers always matches the Barrier size.  Using chunks_mut(chunk) can
    // yield fewer than n_threads chunks when n is not a multiple of chunk
    // (e.g. n=5, n_threads=4, chunk=2 → 3 chunks), which would deadlock the
    // barrier.  Explicit (begin, end) ranges guarantee exactly n_threads
    // partitions; the last few may be empty when n < n_threads * chunk.
    let ranges: Vec<(usize, usize)> = (0..n_threads)
        .map(|tid| {
            let begin = (tid * chunk).min(n);
            let end = ((tid + 1) * chunk).min(n);
            (begin, end)
        })
        .collect();

    // tmp and f are split into per-thread chunks using safe Rust: each thread
    // receives exclusive ownership of its contiguous slice (tmp) or strided
    // sub-view (f) with no aliasing.
    let mut tmp_chunks: Vec<&mut [V]> = Vec::with_capacity(n_threads);
    {
        let mut remaining = tmp_buf;
        for &(begin, end) in &ranges {
            let len = end - begin;
            let (lo, hi) = remaining.split_at_mut(len);
            tmp_chunks.push(lo);
            remaining = hi;
        }
    }

    let mut f_parts: Vec<ArrayViewMut1<'_, V>> = Vec::with_capacity(n_threads);
    {
        let mut remaining = f;
        for &(begin, end) in &ranges {
            let len = end - begin;
            let (lo, hi) = remaining.split_at(Axis(0), len);
            f_parts.push(lo);
            remaining = hi;
        }
    }

    std::thread::scope(|scope| {
        for (tid, ((begin, end), (tmp_chunk, mut f_chunk))) in ranges
            .into_iter()
            .zip(tmp_chunks.into_iter().zip(f_parts))
            .enumerate()
        {
            let n_local = end - begin; // rows owned by this thread

            let barrier = Arc::clone(&barrier);
            let exit_flag = Arc::clone(&exit_flag);

            scope.spawn(move || {
                // tmp_chunk : &mut [V]            — safe, exclusive, contiguous
                // f_chunk   : ArrayViewMut1<'_,V> — safe, exclusive, may be strided
                //
                // raw_b1 : RawSlice — b1 read/write duality (barrier-sequenced)
                // raw_c2 / raw_c3 : RawSlice — per-thread norm slots (disjoint writes,
                //                              barrier-sequenced reads by thread 0)

                let mut c1 = if tid == 0 {
                    let b1 = unsafe { raw_b1.as_slice() };
                    inf_norm(b1)
                } else {
                    V::Real::default()
                };

                for _outer in 0..s {
                    // -----------------------------------------------------------
                    // Taylor loop
                    // -----------------------------------------------------------
                    for j in 1..=m_star {
                        // Phase 1: parallel matvec — each thread fills tmp_chunk.
                        // barrier ensures b1 is fully written before any thread reads it.
                        barrier.wait();
                        {
                            let b1 = unsafe { raw_b1.as_slice() };
                            op.dot_chunk(true, b1, tmp_chunk, begin)
                                .expect("expm_multiply_par: dot_chunk failed");
                        }

                        // Phase 2: element-wise update + local norm accumulation.
                        // barrier ensures all tmp_chunk writes are visible.
                        barrier.wait();
                        let scale = a * V::from_real(V::real_from_f64(1.0 / (j * s) as f64));
                        let mut lc2 = V::Real::default();
                        let mut lc3 = V::Real::default();
                        {
                            let b1 = unsafe { raw_b1.as_slice() };
                            for k_local in 0..n_local {
                                let t = scale * (tmp_chunk[k_local] - mu * b1[begin + k_local]);
                                tmp_chunk[k_local] = t;
                                f_chunk[k_local] += t;
                                let at = t.abs_val();
                                let af = f_chunk[k_local].abs_val();
                                if at > lc2 {
                                    lc2 = at;
                                }
                                if af > lc3 {
                                    lc3 = af;
                                }
                            }
                        }
                        unsafe {
                            raw_c2.write(tid, lc2);
                            raw_c3.write(tid, lc3);
                        }

                        // Phase 3: thread 0 reduces norms and sets exit_flag.
                        // barrier ensures all norm slots are written.
                        barrier.wait();
                        if tid == 0 {
                            let c2 = (0..n_threads)
                                .map(|t| unsafe { raw_c2.read(t) })
                                .fold(V::Real::default(), |acc, x| if x > acc { x } else { acc });
                            let c3 = (0..n_threads)
                                .map(|t| unsafe { raw_c3.read(t) })
                                .fold(V::Real::default(), |acc, x| if x > acc { x } else { acc });
                            exit_flag.store(c1 + c2 <= tol * c3, Ordering::Release);
                            c1 = c2;
                        }

                        // Phase 4: break or copy tmp → b1.
                        // barrier broadcasts exit_flag.
                        barrier.wait();
                        if exit_flag.load(Ordering::Acquire) {
                            break;
                        }
                        {
                            let b1_chunk = unsafe { raw_b1.subslice_mut(begin..begin + n_local) };
                            b1_chunk.copy_from_slice(tmp_chunk);
                        }
                        // No trailing barrier: Phase 1 of j+1 provides it.
                    }
                    // end Taylor loop

                    // -----------------------------------------------------------
                    // Post-partition: f *= η;  b1 = f
                    // -----------------------------------------------------------
                    // barrier: all threads have exited the j loop.
                    barrier.wait();
                    for fk in f_chunk.iter_mut() {
                        *fk *= eta;
                    }
                    // barrier: all f[*] *= η before b1 = f.
                    barrier.wait();
                    {
                        let b1_chunk = unsafe { raw_b1.subslice_mut(begin..begin + n_local) };
                        for (b, &fk) in b1_chunk.iter_mut().zip(f_chunk.iter()) {
                            *b = fk;
                        }
                    }

                    // Compute c1 = inf_norm(b1) for the next outer iteration.
                    {
                        let b1_chunk = unsafe { raw_b1.subslice_mut(begin..begin + n_local) };
                        let local_max = b1_chunk
                            .iter()
                            .map(|x| x.abs_val())
                            .fold(V::Real::default(), |acc, x| if x > acc { x } else { acc });
                        unsafe { raw_c2.write(tid, local_max) };
                    }
                    // barrier: all c2 slots written before thread 0 reduces.
                    barrier.wait();
                    if tid == 0 {
                        c1 = (0..n_threads)
                            .map(|t| unsafe { raw_c2.read(t) })
                            .fold(V::Real::default(), |acc, x| if x > acc { x } else { acc });
                        exit_flag.store(false, Ordering::Release);
                    }
                    // Phase-1 barrier of the next outer iteration's j=1 provides
                    // the final synchronisation before threads read b1 again.
                }
            });
        }
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Batch variant
// ---------------------------------------------------------------------------

/// Compute `exp(a·A) · F` in-place for multiple column vectors simultaneously.
///
/// `f` and `work` have shape `(dim, n_vecs)`.  Parameters `a`, `mu`, `s`,
/// `m_star`, `tol` are the same for all columns; only `f` differs.
///
/// The convergence check aggregates norms across **all** columns (joint
/// termination: the Taylor loop stops only when every column has converged).
///
/// # Errors
/// Returns `ValueError` if array shapes are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn expm_multiply_many<V, Op>(
    op: &Op,
    a: V,
    mu: V,
    s: usize,
    m_star: usize,
    tol: V::Real,
    mut f: ArrayViewMut2<'_, V>,
    mut work: ArrayViewMut2<'_, V>,
) -> Result<(), QuSpinError>
where
    V: ExpmComputation,
    Op: LinearOperator<V>,
{
    let n = op.dim();
    let n_vecs = f.ncols();

    if f.nrows() != n {
        return Err(QuSpinError::ValueError(format!(
            "f.nrows()={} must equal op.dim()={}",
            f.nrows(),
            n
        )));
    }
    if work.nrows() < 2 * n || work.ncols() < n_vecs {
        return Err(QuSpinError::ValueError(format!(
            "work shape ({},{}) must be (>= 2*{}, >= {})",
            work.nrows(),
            work.ncols(),
            n,
            n_vecs
        )));
    }

    if n == 0 || n_vecs == 0 || s == 0 {
        return Ok(());
    }

    let inv_s = V::real_from_f64(1.0 / s as f64);
    let eta = (a * mu * V::from_real(inv_s)).exp_val();

    // Views into the work array: b1 = work[0..n, :], tmp = work[n..2n, :]
    let (mut b1_view, mut tmp_view) = work.view_mut().split_at(ndarray::Axis(0), n);

    // B = F (copy f into b1)
    b1_view.assign(&f);

    for _i in 0..s {
        // c1 = max over all (k, col) of |B[k, col]|
        let mut c1 = inf_norm_2d(b1_view.view());

        'taylor: for j in 1..=m_star {
            // tmp = A · B  (batch matvec)
            op.dot_many(true, b1_view.view(), tmp_view.view_mut())?;

            let scale = a * V::from_real(V::real_from_f64(1.0 / (j * s) as f64));

            let mut c2 = V::Real::default();
            let mut c3 = V::Real::default();

            for k in 0..n {
                for col in 0..n_vecs {
                    let b = b1_view[[k, col]];
                    let t = scale * (tmp_view[[k, col]] - mu * b);
                    tmp_view[[k, col]] = t;
                    let fval = f[[k, col]] + t;
                    f[[k, col]] = fval;
                    let abs_t = t.abs_val();
                    let abs_f = fval.abs_val();
                    if abs_t > c2 {
                        c2 = abs_t;
                    }
                    if abs_f > c3 {
                        c3 = abs_f;
                    }
                }
            }

            if c1 + c2 <= tol * c3 {
                break 'taylor;
            }

            c1 = c2;
            // B = tmp
            b1_view.assign(&tmp_view);
        }

        // F *= η;  B = F
        for fk in f.iter_mut() {
            *fk *= eta;
        }
        b1_view.assign(&f);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Vector infinity norm: `max_k |v[k]|`.
fn inf_norm<V: ExpmComputation>(v: &[V]) -> V::Real {
    v.iter()
        .map(|x| x.abs_val())
        .fold(V::Real::default(), |acc, x| if x > acc { x } else { acc })
}

/// Matrix infinity norm over all elements: `max_{k,col} |M[k,col]|`.
fn inf_norm_2d<V: ExpmComputation>(m: ndarray::ArrayView2<'_, V>) -> V::Real {
    m.iter()
        .map(|x| x.abs_val())
        .fold(V::Real::default(), |acc, x| if x > acc { x } else { acc })
}
