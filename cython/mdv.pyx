# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import math
import time
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport round, ceil, sqrt, log, log1p, pow, INFINITY
from libcpp.algorithm cimport lower_bound, upper_bound
from cython.parallel cimport prange

cdef extern from *:
    """
    #ifdef HAVE_OPENMP
    #include <omp.h>
    #else
    static inline int omp_get_max_threads(void) { return 1; }
    static inline int omp_get_thread_num(void)  { return 0; }
    #endif
    """
    int omp_get_max_threads() noexcept nogil
    int omp_get_thread_num()  noexcept nogil

import quant_cython_k_center as kc

# Ensure numpy types are initialized
cnp.import_array()

# -------------------------------------------------------------------------
# Core C Logic
# -------------------------------------------------------------------------

cdef Py_ssize_t _subdivide_intervals_c(
    double* w,
    Py_ssize_t n_w,
    double* interval_starts,
    double* interval_ends,
    Py_ssize_t m,
    Py_ssize_t K,
    double* out,
    Py_ssize_t out_size,
    double L,
    double target_gap,
    bint run_in_parallel,
) noexcept nogil:
    """
    Pure-C subdivision of m intervals into fine-grained sub-bucket (min, max) pairs.

    Each of the m intervals is split into K_i sub-buckets. Data points in w are
    assigned to sub-buckets and per-sub-bucket (min, max) values are recorded.
    The output is a flat sorted array of the non-empty sub-bucket endpoints,
    suitable for use as quantization candidates.

    Sub-bucket count K_i per interval is chosen as follows:
      - target_gap > 0 (gap-based): K_i = max(K, ceil(bin_span / target_gap)),
        guaranteeing each sub-bucket has width <= target_gap.
      - target_gap <= 0, L > 0 (density-proportional): K_i = max(K, round(K*m*c_i/n))
        where c_i is the point count in interval i.
      - L == 0 (binary-search path): fixed K sub-buckets per interval.

    When run_in_parallel=True and OpenMP is available, uses parallel counting and
    parallel assignment with per-thread output buffers to avoid write conflicts.

    Parameters
    ----------
    w : double*
        Pointer to n_w data points.
    n_w : Py_ssize_t
        Number of data points.
    interval_starts : double*
        Pointer to m interval start values (left endpoints), sorted ascending.
    interval_ends : double*
        Pointer to m interval end values (right endpoints).
    m : Py_ssize_t
        Number of intervals.
    K : Py_ssize_t
        Minimum number of sub-buckets per interval (baseline resolution).
    out : double*
        Output buffer. Must be pre-allocated with at least out_size doubles.
    out_size : Py_ssize_t
        Allocated length of out.
    L : double
        Uniform interval width (interval_starts[i+1] - interval_starts[i]).
        When L > 0, interval lookup uses O(1) arithmetic. When L == 0, uses
        binary search.
    target_gap : double
        Maximum allowed sub-bucket width. When > 0, overrides density-proportional
        allocation and sets K_i = ceil(bin_span / target_gap).
    run_in_parallel : bool
        Enable OpenMP parallelism for counting and assignment loops.

    Returns
    -------
    Py_ssize_t
        Number of elements written to out (the length of the output array).
        Returns -1 on memory allocation failure.
    """
    cdef double* inv_interval_sizes = <double*>malloc(m * sizeof(double))
    if inv_interval_sizes == NULL:
        return -1

    cdef Py_ssize_t i
    cdef double int_start, int_end
    cdef Py_ssize_t idx, local_b, global_b
    cdef double val, w_min
    cdef double base = interval_starts[0]
    cdef double inv_L = 1.0 / L if L > 0 else 0.0
    cdef Py_ssize_t total_count, running_total, used_out_size, K_i, K_i_gap
    cdef double bin_span
    cdef Py_ssize_t* counts = NULL
    cdef Py_ssize_t* K_per_bin = NULL
    cdef Py_ssize_t* prefix_sum = NULL

    # Parallel-specific vars
    cdef int nthreads = 1
    cdef int t, tid
    cdef Py_ssize_t* t_counts = NULL
    cdef double* thread_out = NULL
    cdef Py_ssize_t thread_stride, t_slot, base_offset, count_pad, count_stride
    cdef double g_min, g_max, t_min, t_max

    if run_in_parallel:
        nthreads = omp_get_max_threads()

    if L > 0:
        K_per_bin  = <Py_ssize_t*>malloc(m * sizeof(Py_ssize_t))
        prefix_sum = <Py_ssize_t*>malloc(m * sizeof(Py_ssize_t))
        if K_per_bin == NULL or prefix_sum == NULL:
            free(inv_interval_sizes)
            if K_per_bin  != NULL: free(K_per_bin)
            if prefix_sum != NULL: free(prefix_sum)
            return -1

        if target_gap <= 0.0:
            counts = <Py_ssize_t*>malloc(m * sizeof(Py_ssize_t))
            if counts == NULL:
                free(inv_interval_sizes)
                free(K_per_bin)
                free(prefix_sum)
                return -1

            # Opt B: parallel counting with per-thread arrays.
            # Merge cost O(nthreads*m) is negligible vs O(n).
            if nthreads > 1:
                count_pad = 8 # 64 bits / 8 bytes per Py_ssize_t
                count_stride = (m + count_pad - 1) & ~(count_pad - 1) # align to 8
                t_counts = <Py_ssize_t*>malloc(nthreads * count_stride * sizeof(Py_ssize_t))
            if t_counts != NULL:
                for i in prange(<Py_ssize_t>nthreads * count_stride, schedule='static'):
                    t_counts[i] = 0
                for i in prange(n_w, schedule='static'):
                    tid = omp_get_thread_num()
                    idx = <Py_ssize_t>((w[i] - base) * inv_L)
                    if idx >= m:
                        idx = m - 1
                    t_counts[<Py_ssize_t>tid * count_stride + idx] += 1
                for i in range(m):
                    counts[i] = 0
                    for t in range(nthreads):
                        counts[i] += t_counts[<Py_ssize_t>t * count_stride + i]
                free(t_counts)
                t_counts = NULL
            else:
                # Serial fallback (nthreads == 1 or malloc failed)
                for i in range(m):
                    counts[i] = 0
                for i in range(n_w):
                    idx = <Py_ssize_t>((w[i] - base) * inv_L)
                    if idx >= m:
                        idx = m - 1
                    counts[idx] += 1

        running_total = 0
        for i in range(m):
            K_i = K
            if target_gap > 0.0:
                bin_span = interval_ends[i] - interval_starts[i]
                if bin_span > 0.0:
                    K_i_gap = <Py_ssize_t>ceil(bin_span / target_gap)
                    if K_i_gap > K_i:
                        K_i = K_i_gap
                else:
                    K_i = 0  # empty bin: inv_interval_sizes[i]==0 forces local_b=0, only slot 0 used
            else:
                K_i = <Py_ssize_t>(round(<double>K * m * counts[i] / n_w))
                if K_i < K:
                    K_i = K
            K_per_bin[i]  = K_i
            prefix_sum[i] = running_total
            running_total += K_i + 1
        used_out_size = 2 * running_total

        if counts != NULL:
            free(counts)
            counts = NULL

        for i in range(m):
            int_start = interval_starts[i]
            int_end   = interval_ends[i]
            if int_end <= int_start:
                inv_interval_sizes[i] = 0.0
            else:
                inv_interval_sizes[i] = K_per_bin[i] / (int_end - int_start)
    else:
        for i in range(m):
            int_start = interval_starts[i]
            int_end   = interval_ends[i]
            if int_end <= int_start:
                inv_interval_sizes[i] = 0.0
            else:
                inv_interval_sizes[i] = K / (int_end - int_start)
        used_out_size = out_size

    # Opt A: parallel main assignment loop (L>0 only).
    # Cost gate: merge work (nthreads * used_out_size) must be < half the loop work (n_w).
    thread_stride = 2 * used_out_size
    if nthreads > 1 and L > 0 and <Py_ssize_t>nthreads * thread_stride < n_w // 2:
        thread_out = <double*>malloc(<Py_ssize_t>nthreads * thread_stride * sizeof(double))

    if thread_out != NULL:
        # Init per-thread output buffers in parallel
        for i in prange(<Py_ssize_t>nthreads * used_out_size, schedule='static'):
            thread_out[2 * i]     = INFINITY
            thread_out[2 * i + 1] = -INFINITY

        # Parallel assignment: each thread writes only to its own buffer section
        for i in prange(n_w, schedule='static'):
            val = w[i]
            tid = omp_get_thread_num()
            base_offset = <Py_ssize_t>tid * thread_stride

            idx = <Py_ssize_t>((val - base) * inv_L)
            if idx >= m:
                idx = m - 1

            w_min   = interval_starts[idx]
            local_b = <Py_ssize_t>((val - w_min) * inv_interval_sizes[idx])
            global_b = 2 * (prefix_sum[idx] + local_b)

            t_slot = base_offset + global_b
            if val < thread_out[t_slot]:
                thread_out[t_slot] = val
            if val > thread_out[t_slot + 1]:
                thread_out[t_slot + 1] = val

        # Serial merge: min-of-mins and max-of-maxes across threads
        for global_b in range(0, used_out_size, 2):
            g_min = INFINITY
            g_max = -INFINITY
            for t in range(nthreads):
                t_min = thread_out[<Py_ssize_t>t * thread_stride + global_b]
                t_max = thread_out[<Py_ssize_t>t * thread_stride + global_b + 1]
                if t_min < g_min:
                    g_min = t_min
                if t_max > g_max:
                    g_max = t_max
            out[global_b]     = g_min
            out[global_b + 1] = g_max

        free(thread_out)
        thread_out = NULL
    else:
        # Serial path: init sentinels then assign (also handles L==0 binary-search)
        for i in range(0, used_out_size, 2):
            out[i]     = INFINITY
            out[i + 1] = -INFINITY

        for i in range(n_w):
            val = w[i]
            if L > 0:
                idx = <Py_ssize_t>((val - base) * inv_L)
                if idx >= m:
                    idx = m - 1
            else:
                idx = (upper_bound(interval_starts, interval_starts + m, val) - interval_starts) - 1
                if idx < 0:
                    continue
                elif idx >= m:
                    idx = m - 1

            w_min   = interval_starts[idx]
            local_b = <Py_ssize_t>((val - w_min) * inv_interval_sizes[idx])
            if L > 0:
                global_b = 2 * (prefix_sum[idx] + local_b)
            else:
                global_b = 2 * ((K + 1) * idx + local_b)

            if val < out[global_b]:
                out[global_b] = val
            if val > out[global_b + 1]:
                out[global_b + 1] = val

    # make sure first element is always min, i.e. interval_starts[0]
    if out[0] != interval_starts[0]:
        out[0] = interval_starts[0]
        if out[1] == -INFINITY:
            out[1] = interval_starts[0]

    total_count = 0
    for i in range(0, used_out_size - 1, 2):
        if out[i] != INFINITY:
            if out[i] == out[i+1]:
                out[total_count] = out[i]
                total_count += 1
            else:
                out[total_count]     = out[i]
                out[total_count + 1] = out[i + 1]
                total_count += 2

    if out[total_count - 1] != interval_ends[m - 1]:
        out[total_count] = interval_ends[m - 1]
        total_count += 1

    free(inv_interval_sizes)
    if K_per_bin  != NULL: free(K_per_bin)
    if prefix_sum != NULL: free(prefix_sum)
    return total_count


# -------------------------------------------------------------------------
# Python Wrapper
# -------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def subdivide_intervals_vectorized_fast(w, intervals, double gamma, double L=0.0,
                                        double target_gap=0.0, bint run_in_parallel=False):
    """
    Subdivide a set of intervals into fine sub-buckets and return the non-empty
    endpoint values as a sorted 1D array.

    Each interval is split into K = round(1/gamma) sub-buckets. Data points from w
    are assigned to sub-buckets, and the (min, max) of each non-empty sub-bucket is
    recorded. The resulting endpoint values serve as quantization candidates for mdv.

    Parameters
    ----------
    w : array-like, float64
        1D array of data points to assign into sub-buckets.
    intervals : array-like, shape (m, 2), float64
        Array of m intervals, where intervals[i] = [start_i, end_i].
    gamma : float
        Subdivision resolution parameter. Sets the base sub-bucket count
        K = round(1/gamma). Typically gamma = sqrt(eps)/2.
    L : float, default 0.0
        Uniform interval width. When L > 0, uses O(1) arithmetic for interval
        lookup; when L == 0, uses binary search.
    target_gap : float, default 0.0
        Maximum allowed sub-bucket width. When > 0, overrides the fixed-K
        allocation and sets K_i = ceil(bin_span / target_gap) per interval.
    run_in_parallel : bool, default False
        Enable OpenMP parallelism for counting and assignment loops.

    Returns
    -------
    ndarray, shape (p,), float64
        Sorted 1D array of p non-empty sub-bucket endpoint values (each non-empty
        sub-bucket contributes its min and/or max). Returns an empty array if w or
        intervals is empty.
    """
    if w is None or not len(w) or intervals is None or not len(intervals):
        return np.array([], dtype=np.float64)

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] w_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] intervals_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] starts_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] ends_arr

    try:
        w_arr = np.ascontiguousarray(w, dtype=np.float64)
        intervals_arr = np.ascontiguousarray(intervals, dtype=np.float64)
        starts_arr = np.ascontiguousarray(intervals_arr[:, 0], dtype=np.float64)
        ends_arr = np.ascontiguousarray(intervals_arr[:, 1], dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.float64)

    cdef Py_ssize_t n_w = w_arr.shape[0]
    cdef Py_ssize_t m = intervals_arr.shape[0]
    if n_w == 0 or m == 0:
        return np.array([], dtype=np.float64)

    cdef Py_ssize_t K = <Py_ssize_t>round(1.0 / gamma) if gamma > 0 else 1
    if K <= 0:
        K = 1

    cdef Py_ssize_t K_max = K
    cdef Py_ssize_t K_gap
    cdef Py_ssize_t max_out_size
    cdef Py_ssize_t n_nonempty
    cdef Py_ssize_t i
    cdef double K_gap_float
    if target_gap > 0.0 and L > 0.0:
        K_gap_float = L / target_gap
        # Use C ceil (not Python math.ceil) to avoid Python int overflow.
        # Guard against values that would overflow ssize_t (~9.2e18).
        if K_gap_float >= 4.6e18:
            K_gap = K  # safety fallback: shouldn't occur with valid v_star
        else:
            K_gap = <Py_ssize_t>ceil(K_gap_float)
        if K_gap > K_max:
            K_max = K_gap
        # Non-empty bins use K_max subdivisions; empty bins use only 1 slot.
        # Avoids massive over-allocation when most bins are empty.
        n_nonempty = 0
        for i in range(m):
            if starts_arr[i] < ends_arr[i]:
                n_nonempty += 1
        max_out_size = n_nonempty * (K_max + 1) * 2 + (m - n_nonempty) * 2 + 2
    else:
        max_out_size = m * (2 * K + 1) * 2 + 2

    cdef Py_ssize_t result_len
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] result_arr = np.empty(max_out_size, dtype=np.float64)

    with nogil:
        result_len = _subdivide_intervals_c(
            &w_arr[0], n_w,
            &starts_arr[0], &ends_arr[0], m,
            K, &result_arr[0],
            max_out_size, L, target_gap, run_in_parallel
        )

    if result_len < 0:
        raise MemoryError("Failed to allocate internal C buffers.")

    return result_arr[:result_len]

@cython.boundscheck(False)
@cython.wraparound(False)
def subdivide_intervals(w, intervals, double gamma, double L=0.0, double target_gap=0.0,
                        bint run_in_parallel=False):
    """
    Alias for subdivide_intervals_vectorized_fast. See that function for full documentation.

    Parameters
    ----------
    w : array-like, float64
        1D array of data points.
    intervals : array-like, shape (m, 2), float64
        Array of m intervals [start, end].
    gamma : float
        Subdivision resolution; K = round(1/gamma) sub-buckets per interval.
    L : float, default 0.0
        Uniform interval width for O(1) lookup. 0.0 uses binary search.
    target_gap : float, default 0.0
        Maximum sub-bucket width; overrides fixed-K when > 0.
    run_in_parallel : bool, default True
        Enable OpenMP parallelism.

    Returns
    -------
    ndarray, shape (p,), float64
        Sorted array of non-empty sub-bucket endpoint values.
    """
    return subdivide_intervals_vectorized_fast(w, intervals, gamma, L, target_gap, run_in_parallel)


cdef inline double _max_variance_c(
    double* w, Py_ssize_t n,
    double* Q, Py_ssize_t m
) noexcept nogil:
    """
    Pure-C maximum variance for a fixed quantizer set Q.

    For each data point w[i], identifies its enclosing quantizer interval [lo, hi]
    via binary search and computes the variance (w[i] - lo) * (hi - w[i]). Returns
    the maximum over all points. Points outside [Q[0], Q[m-1]] have variance 0.

    Parameters
    ----------
    w : double*
        Pointer to n data points.
    n : Py_ssize_t
        Number of data points.
    Q : double*
        Pointer to m sorted quantizer values.
    m : Py_ssize_t
        Number of quantizer values.

    Returns
    -------
    double
        Maximum variance across all data points.
    """
    cdef double max_v = 0.0
    cdef double val, lo, hi, var
    cdef Py_ssize_t i, left, right, mid, idx

    for i in range(n):
        val = w[i]
        # Binary search: first idx where Q[idx] >= val (searchsorted left)
        left = 0
        right = m
        while left < right:
            mid = left + (right - left) // 2
            if Q[mid] < val:
                left = mid + 1
            else:
                right = mid
        idx = left

        if idx == 0:
            lo = Q[0]
            hi = Q[0]
        elif idx == m:
            lo = Q[m - 1]
            hi = Q[m - 1]
        else:
            lo = Q[idx - 1]
            hi = Q[idx]

        var = (val - lo) * (hi - val)
        if var > max_v:
            max_v = var

    return max_v


@cython.boundscheck(False)
@cython.wraparound(False)
def max_variance(w, Q):
    """
    Compute the maximum variance for a fixed sorted quantizer set Q over data w.

    For each data point, finds its enclosing quantizer interval [lo, hi] and
    computes (point - lo) * (hi - point). Returns the maximum over all points.

    Parameters
    ----------
    w : array-like, float64
        1D array of data points.
    Q : array-like, float64
        1D array of quantizer values. Will be sorted internally.

    Returns
    -------
    float
        Maximum variance. Returns 0.0 if w is empty or None.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] w_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] Q_arr

    if w is None:
        return 0.0

    w_arr = np.ascontiguousarray(w, dtype=np.float64)
    if w_arr.size == 0:
        return 0.0

    if Q is None or len(Q) == 0:
        raise ValueError("Q must be non-empty")

    Q_arr = np.ascontiguousarray(np.sort(np.asarray(Q, dtype=np.float64)))

    return _max_variance_c(&w_arr[0], w_arr.shape[0], &Q_arr[0], Q_arr.shape[0])


cdef inline int _bisect_left_ptr(double* arr, double target, int lo, int hi_excl) noexcept nogil:
    cdef int mid
    while lo < hi_excl:
        mid = (lo + hi_excl) >> 1
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi_excl = mid
    return lo


cdef inline int _exp_search_ptr(double* arr, Py_ssize_t n, double target, int start_idx) noexcept nogil:
    cdef int bound, lo, hi

    if start_idx >= <int>n:
        return <int>n
    if arr[start_idx] >= target:
        return start_idx

    bound = 1
    while start_idx + bound < <int>n and arr[start_idx + bound] < target:
        bound <<= 1

    lo = start_idx + (bound >> 1)
    hi = start_idx + bound + 1
    if hi > <int>n:
        hi = <int>n
    return _bisect_left_ptr(arr, target, lo, hi)


cdef inline int _min_s_count_ptr(double* w, Py_ssize_t n, double v) noexcept nogil:
    cdef int count, current_idx, i
    cdef double q, w_d, sqrt_v, target, w_i, term1, term2, x

    if v == 0.0:
        return <int>n

    count = 1
    q = w[0]
    w_d = w[n - 1]
    sqrt_v = sqrt(v)
    current_idx = 0

    while q < w_d:
        target = q + sqrt_v
        i = _exp_search_ptr(w, n, target, current_idx)
        if i >= <int>n:
            break

        w_i = w[i]
        term1 = w_i + v / (w_i - q)

        if i > 0 and w[i - 1] > q:
            term2 = w[i - 1] + v / (w[i - 1] - q)
            x = term1 if term1 < term2 else term2
        else:
            x = term1

        count += 1
        q = x
        current_idx = i

    if q < w_d:
        count += 1

    return count


cdef double _min_v_c(
    double* w, Py_ssize_t n, int s,
    double lower_bound, double upper_bound, double eps
) noexcept nogil:
    """Binary search on a geometric variance grid entirely in C."""
    cdef double base = 1.0 + eps
    cdef double log_base = log1p(eps)
    cdef double ratio = upper_bound / lower_bound
    cdef Py_ssize_t k_hi, lo, hi, mid
    cdef double v

    if ratio <= 1.0:
        k_hi = 0
    else:
        k_hi = <Py_ssize_t>ceil(log(ratio) / log_base)

    lo = 0
    hi = k_hi

    while lo < hi:
        mid = lo + (hi - lo) // 2
        v = lower_bound * pow(base, <double>mid)
        if _min_s_count_ptr(w, n, v) <= s:
            hi = mid
        else:
            lo = mid + 1

    return lower_bound * pow(base, <double>lo)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _bisect_left_arr(cnp.ndarray[cnp.float64_t, ndim=1] arr, double target, int lo, int hi_excl):
    cdef int mid
    while lo < hi_excl:
        mid = (lo + hi_excl) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi_excl = mid
    return lo


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _exp_search(cnp.ndarray[cnp.float64_t, ndim=1] arr, double target, int start_idx):
    cdef int n = arr.shape[0]
    cdef int bound, lo, hi

    if start_idx >= n:
        return n

    if arr[start_idx] >= target:
        return start_idx

    bound = 1
    while start_idx + bound < n and arr[start_idx + bound] < target:
        bound *= 2

    lo = start_idx + (bound // 2)
    hi = min(start_idx + bound + 1, n)
    return _bisect_left_arr(arr, target, lo, hi)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _min_s_count(cnp.ndarray[cnp.float64_t, ndim=1] w, double v):
    return _min_s_count_ptr(&w[0], w.shape[0], v)


class SortedObj1:
    """
    Pre-sorted quantization candidate set with methods for variance optimization.

    Holds a sorted float64 array w of quantization candidates and provides
    efficient greedy and search-based methods for the 1D max-variance objective.
    All methods assume w is sorted; pass w_sorted=True to skip the sort.
    """

    def __init__(self, w, bint w_sorted=False):
        """
        Parameters
        ----------
        w : array-like, float64
            1D array of quantization candidate values.
        w_sorted : bool, default False
            If True, assume w is already sorted ascending and skip the sort.

        Raises
        ------
        ValueError
            If the resulting array has fewer than 2 elements.
        """
        cdef cnp.ndarray arr = np.asarray(w, dtype=np.float64)
        if not w_sorted:
            arr = np.sort(arr)
        self.w = arr
        self.n = arr.shape[0]

        if self.n < 2:
            raise ValueError("Input array must have at least 2 elements.")

    def min_s(self, double v, bint return_set=False):
        """
        Find the minimum number of quantizer points needed so that max-variance <= v.

        Uses a greedy forward scan: places each successive quantizer at the optimal
        position to cover as many points as possible within variance budget v.
        Exponential search accelerates finding the next uncovered point.

        Parameters
        ----------
        v : float
            Variance budget. Must be >= 0. If 0, returns n (all points needed).
        return_set : bool, default False
            If False, return only the count (fast path, no list allocation).
            If True, return the list of quantizer values.

        Returns
        -------
        int
            Minimum number of quantizer points, when return_set=False.
        list of float
            The quantizer values achieving the minimum count, when return_set=True.
        """
        cdef list Q
        cdef double q, w_d, sqrt_v, target, w_i, term1, term2, x
        cdef int current_idx, i

        if v == 0:
            return self.n

        # Fast count-only path avoids Python list construction in hot loops.
        if not return_set:
            return _min_s_count(self.w, v)

        Q = [float(self.w[0])]
        q = self.w[0]
        w_d = self.w[self.n - 1]
        sqrt_v = math.sqrt(v)
        current_idx = 0

        while q < w_d:
            target = q + sqrt_v
            i = _exp_search(self.w, target, current_idx)

            if i >= self.n:
                break

            w_i = self.w[i]
            term1 = w_i + v / (w_i - q)

            if i > 0 and self.w[i - 1] > q:
                term2 = self.w[i - 1] + v / (self.w[i - 1] - q)
                x = term1 if term1 < term2 else term2
            else:
                x = term1

            Q.append(float(x))
            q = x
            current_idx = i

        last_q = Q[len(Q) - 1]
        if last_q < w_d:
            Q.append(float(w_d))
        elif last_q > w_d:
            Q[len(Q) - 1] = float(w_d)

        return Q if return_set else len(Q)

    def min_v(self, int s, double lower_bound, double upper_bound, double eps, bint return_set=False):
        """
        Find the minimum variance achievable with exactly s quantizer points.

        Binary-searches a geometric grid of variance values in [lower_bound, upper_bound]
        with ratio (1+eps) between steps, entirely in C via _min_v_c (nogil).

        Parameters
        ----------
        s : int
            Target number of quantizer points. Must be >= 2.
        lower_bound : float
            Lower bound on the variance search range. Must be > 0.
        upper_bound : float
            Upper bound on the variance search range. Must be >= lower_bound.
        eps : float
            Grid resolution. Adjacent grid points differ by a factor of (1+eps).
        return_set : bool, default False
            If False, return the minimum variance value.
            If True, return the quantizer set achieving that variance.

        Returns
        -------
        float
            Minimum variance on the geometric grid, when return_set=False.
        list of float
            Quantizer values achieving the minimum variance, when return_set=True.

        Raises
        ------
        ValueError
            If s < 2, eps <= 0, lower_bound <= 0, upper_bound < lower_bound, or
            no feasible variance exists in [lower_bound, upper_bound].
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] w_arr = self.w
        cdef Py_ssize_t n = w_arr.shape[0]
        cdef double* w_ptr = &w_arr[0]
        cdef double v_star

        if s < 2:
            raise ValueError("s must be at least 2")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if lower_bound <= 0:
            raise ValueError("lower_bound must be positive")
        if upper_bound < lower_bound:
            raise ValueError("upper_bound must be >= lower_bound")

        if _min_s_count_ptr(w_ptr, n, lower_bound) <= s:
            return self.min_s(lower_bound, return_set=True) if return_set else lower_bound

        if _min_s_count_ptr(w_ptr, n, upper_bound) > s:
            raise ValueError("No feasible v in the provided [lower_bound, upper_bound] range")

        with nogil:
            v_star = _min_v_c(w_ptr, n, s, lower_bound, upper_bound, eps)

        return self.min_s(v_star, return_set=True) if return_set else v_star


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray get_unique_c(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] intervals):
    """
    Return a sorted 1-D array of the non-empty interval endpoints.

    k_center_heuristic fills empty bins with [dummy_val, dummy_val] where
    dummy_val == intervals[0, 0] (the global minimum).  For any bin i > 0
    the actual data values are strictly > dummy_val (they live in a higher
    grid cell), so checking start == dummy_val is a reliable empty-bin test.
    Bin 0 is always non-empty and is always included.
    """
    cdef Py_ssize_t m = intervals.shape[0]
    cdef Py_ssize_t i, out_idx
    cdef double dummy_val = intervals[0, 0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(2 * m, dtype=np.float64)

    result[0] = intervals[0, 0]
    result[1] = intervals[0, 1]
    out_idx = 2

    for i in range(1, m):
        if intervals[i, 0] != dummy_val:
            result[out_idx]     = intervals[i, 0]
            result[out_idx + 1] = intervals[i, 1]
            out_idx += 2

    return result[:out_idx]


@cython.boundscheck(False)
@cython.wraparound(False)
def mdv(*args, **kwargs):
    """
    Find the optimal quantizer set minimizing maximum variance for s points over data w.

    Main entry point for the 1D quantization objective. Runs a pipeline of:
    1. k-center heuristic (grid-based coreset) to generate interval candidates,
    2. rough variance estimate from the interval structure,
    3. interval subdivision into fine sub-buckets,
    4. iterative refinement of v* via min_v on the sub-bucket candidates.

    Parameters
    ----------
    w : array-like, float64
        1D array of data points to quantize. Must have at least 2 elements.
    s : int
        Number of quantizer points.
    eps : float
        Approximation parameter. The returned solution has variance at most
        (1+eps) * v*, where v* is the true optimum. Must be > 0.
    return_set : bool, default True
        If True, return the quantizer point values.
        If False, return the achieved maximum variance.
    k_center_alg : callable or None, default None
        Custom k-center algorithm. When None, uses k_center_heuristic internally.
    run_k_heuristic_in_parallel : bool, default True
        Enable OpenMP parallelism in the k-center heuristic step.
    approx_factor : float, default 2.0
        Unused (kept for API compatibility).
    allow_relaxed : bool, default True
        Passed to k_center_heuristic; returns relaxed coreset when True.
    interval_count_factor : float, default 2.0
        Divisor for gamma = sqrt(eps) / interval_count_factor.
    run_subdivide_in_parallel : bool, default True
        Enable OpenMP parallelism in the subdivision step.
    B_factor : int, default 10
        Multiplier for the number of k-center buckets: B = B_factor * s * 2/sqrt(eps) + s.

    Returns
    -------
    list of float
        Optimal quantizer point values, when return_set=True.
    float
        Maximum variance achieved by the returned quantizer set, when return_set=False.
    """
    result, _, _, _, _ = mdv_time(*args, **kwargs)
    return result


def mdv_time(w, int s, double eps, bint return_set=True, k_center_alg=None, bint run_k_heuristic_in_parallel=False, double approx_factor=2.0, bint allow_relaxed = True, double interval_count_factor = 2.0, bint run_subdivide_in_parallel=False, int B_factor = 10):
    """
    Run mdv and return both the result and per-phase timing breakdowns.

    Identical to mdv but also records wall-clock time spent in each pipeline phase.
    Useful for profiling and performance analysis.

    Parameters
    ----------
    w : array-like, float64
        1D array of data points to quantize. Must have at least 2 elements.
    s : int
        Number of quantizer points.
    eps : float
        Approximation parameter. Must be > 0.
    return_set : bool, default True
        If True, return the quantizer point values; if False, return max variance.
    k_center_alg : callable or None, default None
        Custom k-center algorithm. When None, uses k_center_heuristic.
    run_k_heuristic_in_parallel : bool, default True
        Enable OpenMP parallelism in the k-center heuristic step.
    approx_factor : float, default 2.0
        Unused (kept for API compatibility).
    allow_relaxed : bool, default True
        Passed to k_center_heuristic.
    interval_count_factor : float, default 2.0
        Divisor for gamma = sqrt(eps) / interval_count_factor.
    run_subdivide_in_parallel : bool, default True
        Enable OpenMP parallelism in the subdivision step.
    B_factor : int, default 10
        Multiplier for the number of k-center buckets.

    Returns
    -------
    result : list of float or float
        Quantizer point values (return_set=True) or max variance (return_set=False).
    t_kcenter_endpoints : float
        Seconds spent in the k-center / interval generation phase.
    t_rough_sortedmdv : float
        Seconds spent computing the rough variance estimate.
    t_subdivide : float
        Seconds spent in the interval subdivision phase.
    t_remainder : float
        Seconds spent in the min_v refinement and final greedy scan.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w_arr
    cdef object interval_endpoints
    cdef object intervals_arr, endpoints
    cdef double rough_v_obj, gamma, lower, upper
    cdef double t_kcenter_endpoints, t_rough_sortedmdv, t_subdivide, t_remainder
    cdef object subdivided_points, refined, q_star, result
    cdef tuple subdivide_phase_times

    w_arr = np.asarray(w, dtype=np.float64)
    if w_arr.shape[0] < 2:
        raise ValueError("Input array must have at least 2 elements.")
    if eps <= 0.0:
        raise ValueError("Must have eps > 0")

    cdef double L = 0.0
    cdef double min_w, max_w
    _t0 = time.perf_counter()
    min_w = np.min(w_arr)
    max_w = np.max(w_arr)
    if k_center_alg is None:
        # compute B to include the first subdivision by \sqrt{eps}/2, to skip the first pass of subdivide_intervals
        B = int(B_factor * s * 2 / math.sqrt(eps)) + s # + s to ensure always at least s
        interval_endpoints = kc.k_center_heuristic(w_arr, s, B=B, return_clusters=True, allow_relaxed=True,
                                                    given_min=min_w, given_max=max_w, has_minmax=True, run_in_parallel=run_k_heuristic_in_parallel)
        L = (max_w - min_w) / B
    else:
        interval_endpoints = k_center_alg(w_arr, s)
    intervals_arr = np.asarray(interval_endpoints, dtype=np.float64)
    endpoints = intervals_arr.reshape(-1)
    t_kcenter_endpoints = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    # Compute tight rough_v_obj from the B-bin interval data by grouping the
    # B bins into s equal groups and taking max(group_span²/4).  This gives
    # an achievable s-center solution (one center per group midpoint), so
    # rough_v_obj ≥ v*.  Setting lower = rough_v_obj / 16 keeps k_hi ≈ 51
    # for eps=0.1, avoiding the O(n/√v) bottleneck in _min_s_count.
    cdef int _gs = intervals_arr.shape[0] // s
    if _gs >= 1:
        _n_g = s * _gs
        _r = intervals_arr[:_n_g].reshape(s, _gs, 2)
        _real = _r[:, :, 0] < _r[:, :, 1]
        _gmx = np.where(_real, _r[:, :, 1], -np.inf).max(axis=1)
        _gmn = np.where(_real, _r[:, :, 0], np.inf).min(axis=1)
        _spans = np.where(np.isfinite(_gmx) & np.isfinite(_gmn), _gmx - _gmn, 0.0)
        rough_v_obj = float((_spans ** 2).max())
        if rough_v_obj <= 0.0:
            rough_v_obj = (max_w - min_w) ** 2
    else:
        rough_v_obj = (max_w - min_w) ** 2
    # Floor rough_v_obj with uniform placement variance to handle skewed distributions
    # where the bin-grouping heuristic underestimates v_opt (e.g. Zipf with large max).
    if s > 1:
        uniform_v = (max_w - min_w) ** 2 / (4.0 * (s - 1) ** 2)
        if uniform_v > rough_v_obj:
            rough_v_obj = uniform_v
    t_rough_sortedmdv = time.perf_counter() - _t0

    gamma = math.sqrt(eps) / interval_count_factor
    _t0 = time.perf_counter()
    if k_center_alg is not None:
        subdivided_points = subdivide_intervals(w_arr, interval_endpoints, gamma, L, run_in_parallel=run_subdivide_in_parallel)
    else:
        # don't subdivide, this was already accounted for in the increased value of B
        # interval_endpoints has dummy [min,min] entries for empty bins; strip them.
        subdivided_points = get_unique_c(intervals_arr)
        # Fallback: if too few candidates (e.g. heavy-tailed data with few non-empty
        # bins), subdivide with baseline K so the lower-bound search converges properly.
        if len(subdivided_points) <= s:
            subdivided_points = subdivide_intervals(w_arr, interval_endpoints, gamma, L, 0.0, run_in_parallel=run_subdivide_in_parallel)
    t_subdivide = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    refined = SortedObj1(subdivided_points, w_sorted=True)
    upper = rough_v_obj

    # lower = rough_v_obj / 16 gives ratio = 128 → k_hi ≈ 51.
    # Safety fallback: if lower already satisfies min_s(lower) ≤ s (meaning
    # lower > v*), halve repeatedly until it is a true lower bound.
    lower = rough_v_obj / 16.0
    if lower <= 0:
        lower = np.finfo(float).tiny
    while lower > np.finfo(float).tiny and _min_s_count(refined.w, lower) <= s:
        lower /= 16.0

    # Iterative refinement: resubdivide until the max k-center bin span is ≤ the
    # target gap sqrt(eps * v_star) / 4 for the current v_star estimate.
    cdef double v_star = refined.min_v(s, lower_bound=lower, upper_bound=upper, eps=eps / 2, return_set=False)
    cdef double target_gap = 0.0
    cdef double max_bin_span = 0.0
    cdef int K_val = <int>round(1.0 / gamma) if gamma > 0 else 1
    if K_val <= 0:
        K_val = 1
    cdef object current_refined = refined
    cdef double lower_curr, upper_curr

    if k_center_alg is None and L > 0.0:
        _spans_arr = intervals_arr[:, 1] - intervals_arr[:, 0]
        max_bin_span = float(np.max(_spans_arr))

    while k_center_alg is None and L > 0.0 and v_star > 0.0:
        target_gap = math.sqrt(eps * v_star) / 2
        if target_gap <= 0.0 or max_bin_span <= target_gap: # removed * K_val since no longer immediately subdividing
            break
        # Resubdivide with tighter gap to guarantee coreset quality for current v_star.
        subdivided_points = subdivide_intervals(w_arr, interval_endpoints, gamma, L, target_gap / 8.0, run_in_parallel=run_subdivide_in_parallel)
        current_refined = SortedObj1(subdivided_points, w_sorted=True)
        lower_curr = v_star / 16.0
        if lower_curr <= 0:
            lower_curr = np.finfo(float).tiny
        while lower_curr > np.finfo(float).tiny and _min_s_count(current_refined.w, lower_curr) <= s:
            lower_curr /= 4.0
        upper_curr = v_star * 2.0
        # v_star from a coarse surrogate can underestimate the fine grid's v*
        # (coarser grid = fewer points = easier to cover = lower apparent v*).
        # Double until feasible, bounded by upper (rough_v_obj), which is always
        # a valid upper bound because current_refined.w ⊆ w_arr.
        while upper_curr < upper and _min_s_count(current_refined.w, upper_curr) > s:
            upper_curr = min(upper_curr * 2.0, upper)
        v_star = current_refined.min_v(s, lower_bound=lower_curr, upper_bound=upper_curr, eps=eps / 2, return_set=False)
        # After subdividing with target_gap/4, sub-buckets have width ≤ target_gap/4.
        # Update max_bin_span so the next iteration checks the current coreset's
        # sub-bucket width rather than the fixed original k-center bin span.
        max_bin_span = target_gap / 8.0

    q_star = current_refined.min_s(v_star, return_set=True)

    result = q_star if return_set else max_variance(w_arr, q_star)
    t_remainder = time.perf_counter() - _t0

    return result, t_kcenter_endpoints, t_rough_sortedmdv, t_subdivide, t_remainder


@cython.boundscheck(False)
@cython.wraparound(False)
def exact_mdv(w, int s, double tol=1e-2):
    """
    Find the minimum variance achievable with s quantizer points over data w.

    Sorts w using np.sort, then binary-searches a geometric variance grid via
    SortedObj1.min_v (which runs entirely in C/nogil).

    Parameters
    ----------
    w : array-like, float64
        1D array of data points to quantize. Must have at least 2 elements.
    s : int
        Number of quantizer points. Must be >= 2.
    tol : float, default 1e-2
        Relative tolerance for the geometric grid search (passed as eps to min_v).

    Returns
    -------
    list of float
        Quantizer point values achieving the minimum variance.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w_sorted = np.sort(np.asarray(w, dtype=np.float64))
    cdef Py_ssize_t n = w_sorted.shape[0]
    cdef double* w_ptr
    cdef double upper, lower, v_star

    if n < 2:
        raise ValueError("Input must have at least 2 elements")
    if s < 2:
        raise ValueError("s must be at least 2")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    w_ptr = &w_sorted[0]

    # Upper bound: variance of a single optimal quantizer covering all data.
    # (w[-1]-w[0])^2/4 >= C(0,n-1) >= v*, so min_s(upper) = 1 <= s always.
    upper = (w_sorted[n - 1] - w_sorted[0]) ** 2 / 4.0

    if upper <= 0.0:
        # All points identical; zero variance trivially achievable.
        return [float(w_sorted[0])] * s

    # Find a lower bound that is infeasible (min_s(lower) > s).
    lower = upper / 16.0
    while lower > 1e-300 and _min_s_count_ptr(w_ptr, n, lower) <= s:
        lower /= 16.0

    if lower <= 1e-300:
        return list(w_sorted[:s])

    with nogil:
        v_star = _min_v_c(w_ptr, n, s, lower, upper, tol)

    return SortedObj1(w_sorted, w_sorted=True).min_s(v_star, return_set=True)
