# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fabs, INFINITY
from libc.stdlib cimport malloc, free
from libcpp.utility cimport pair
from libcpp.algorithm cimport sort
from cython.parallel cimport prange

# Cython's standard library doesn't always expose minmax_element by default,
# so we declare it manually from the C++ <algorithm> header.
cdef extern from "<algorithm>" namespace "std" nogil:
    pair[double*, double*] minmax_element(double* first, double* last)

# Provide OpenMP stubs when compiled without -fopenmp so the parallel function
# compiles and degrades gracefully to single-threaded execution.
cdef extern from *:
    """
    #ifdef HAVE_OPENMP
    #include <omp.h>
    #else
    static inline int omp_get_max_threads(void) { return 1; }
    static inline int omp_get_thread_num(void) { return 0; }
    #endif
    """
    int omp_get_max_threads() noexcept nogil
    int omp_get_thread_num() noexcept nogil

def std_sort_numpy(double[::1] arr):
    """
    Sort a 1D float64 array in-place using C++ std::sort.

    Parameters
    ----------
    arr : ndarray, shape (n,), float64, C-contiguous
        Array to sort. Modified in-place.

    Returns
    -------
    None
    """
    cdef size_t num_elements = arr.shape[0]

    if num_elements <= 1:
        return  # Nothing to sort

    # std::sort takes memory addresses (pointers act as iterators)
    # We pass the pointer to the start, and the pointer to the end
    sort(&arr[0], &arr[0] + num_elements)

cdef inline Py_ssize_t c_bisect_left(double[::1] arr, double val, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = n
    cdef Py_ssize_t mid
    while lo < hi:
        mid = lo + (hi - lo) / 2
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def _to_cluster_ranges(points, centers, bint sort_centers=True):
    """
    Map each cluster center to the min and max of its assigned data points.

    Each point is assigned to its nearest center (ties broken toward the lower center
    when sort_centers=True, using binary search for O(n log k) performance).

    Parameters
    ----------
    points : array-like, float64
        1D array of data points to assign.
    centers : array-like, float64
        1D array of cluster centers.
    sort_centers : bool, default True
        If True, sort centers before assignment and use O(n log k) binary-search
        assignment. If False, use O(n*k) brute-force assignment (only useful when
        centers are known to be unsorted).

    Returns
    -------
    list of tuple[float, float], length len(centers)
        Each element (min_val, max_val) is the range of data points assigned to the
        corresponding center. For centers with no assigned points, returns (center, center).
    """
    if sort_centers:
        centers = np.sort(centers)

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pts_np = np.ascontiguousarray(points, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] ctrs_np = np.ascontiguousarray(centers, dtype=np.float64)
    
    cdef double[::1] pts = pts_np
    cdef double[::1] ctrs = ctrs_np

    cdef Py_ssize_t n_pts = pts.shape[0]
    cdef Py_ssize_t n_centers = ctrs.shape[0]
    cdef Py_ssize_t i, j, best_i, idx
    cdef double p, d, best_d, c, d1, d2

    if n_centers == 0:
        return []

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] mins_np = np.empty(n_centers, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] maxs_np = np.empty(n_centers, dtype=np.float64)
    # Replaced np.zeros with np.empty to bypass Python memset overhead completely
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] seen_np = np.empty(n_centers, dtype=np.uint8)

    cdef double[::1] mins = mins_np
    cdef double[::1] maxs = maxs_np
    cdef cnp.uint8_t[::1] seen = seen_np
    cdef double c_inf = np.inf

    with nogil:
        # Initialize natively in C
        for i in range(n_centers):
            mins[i] = c_inf
            maxs[i] = -c_inf
            seen[i] = 0

        # O(N log K) fast path utilizing binary search
        if sort_centers:
            for i in range(n_pts):
                p = pts[i]
                idx = c_bisect_left(ctrs, p, n_centers)
                
                # Check adjacent centers to find absolute closest
                if idx == 0:
                    best_i = 0
                elif idx == n_centers:
                    best_i = n_centers - 1
                else:
                    d1 = p - ctrs[idx - 1]
                    d2 = ctrs[idx] - p
                    if d1 < d2:
                        best_i = idx - 1
                    else:
                        best_i = idx

                if seen[best_i] == 0:
                    mins[best_i] = p
                    maxs[best_i] = p
                    seen[best_i] = 1
                else:
                    if p < mins[best_i]:
                        mins[best_i] = p
                    if p > maxs[best_i]:
                        maxs[best_i] = p
        else:
            # O(N * K) slow path only if sort_centers=False is enforced
            for i in range(n_pts):
                p = pts[i]
                best_i = 0
                best_d = fabs(p - ctrs[0])

                for j in range(1, n_centers):
                    d = fabs(p - ctrs[j])
                    if d < best_d:
                        best_d = d
                        best_i = j

                if seen[best_i] == 0:
                    mins[best_i] = p
                    maxs[best_i] = p
                    seen[best_i] = 1
                else:
                    if p < mins[best_i]:
                        mins[best_i] = p
                    if p > maxs[best_i]:
                        maxs[best_i] = p

    out = []
    for i in range(n_centers):
        c = ctrs[i]
        if seen[i]:
            out.append((mins[i], maxs[i]))
        else:
            out.append((c, c))

    return out


cdef inline void k_center_gonzalez_c(
    double* pts,
    Py_ssize_t n,
    double* centers_out,
    double* dist,
    int k
) noexcept nogil:
    """
    Pure-C Gonzalez greedy 2-approximation for 1D k-center (serial).

    Iteratively picks the point farthest from all current centers. Fills exactly
    k slots in centers_out. If all points coincide with an existing center before k
    centers are picked, remaining slots are filled with the last picked point.

    Parameters
    ----------
    pts : double*
        Pointer to n data points (need not be sorted).
    n : Py_ssize_t
        Number of data points.
    centers_out : double*
        Output buffer of length k. Filled with the k selected centers.
    dist : double*
        Working buffer of length n. Stores the current min-distance from each point
        to the nearest selected center; must be allocated by the caller.
    k : int
        Number of centers to select.
    """
    cdef Py_ssize_t i, farthest_idx
    cdef Py_ssize_t picked
    cdef double last, d, max_d

    for i in range(n):
        dist[i] = INFINITY

    centers_out[0] = pts[0]

    for picked in range(1, <Py_ssize_t>k):
        last = centers_out[picked - 1]
        farthest_idx = 0
        max_d = -1.0

        for i in range(n):
            d = fabs(pts[i] - last)
            if d < dist[i]:
                dist[i] = d
            if dist[i] > max_d:
                max_d = dist[i]
                farthest_idx = i

        centers_out[picked] = pts[farthest_idx]

        # All points coincide with an existing center; remaining picks are duplicates.
        if max_d == 0.0:
            for i in range(picked + 1, <Py_ssize_t>k):
                centers_out[i] = pts[farthest_idx]
            return


cdef inline void k_center_gonzalez_parallel_c(
    double* pts,
    Py_ssize_t n,
    double* centers_out,
    double* dist,
    int k
) noexcept nogil:
    """
    OpenMP-parallel Gonzalez greedy 2-approximation for 1D k-center.

    Same algorithm as k_center_gonzalez_c but parallelizes each distance-update
    and argmax pass using prange. Per-thread argmax results are stored in
    cache-line-padded arrays (stride 8) to prevent false sharing, then reduced
    serially. Falls back to serial k_center_gonzalez_c on malloc failure.

    Parameters
    ----------
    pts : double*
        Pointer to n data points (need not be sorted).
    n : Py_ssize_t
        Number of data points.
    centers_out : double*
        Output buffer of length k. Filled with the k selected centers.
    dist : double*
        Working buffer of length n. Stores per-point min-distance to nearest
        selected center; must be allocated by the caller.
    k : int
        Number of centers to select.
    """
    cdef int nthreads = omp_get_max_threads()
    # Stride-8 layout: each thread slot is 64 bytes apart (1 cache line),
    # preventing false sharing. Works for both double (8 bytes) and
    # Py_ssize_t (8 bytes on 64-bit), so stride * sizeof == 64.
    cdef double* t_max = <double*>malloc(nthreads * 8 * sizeof(double))
    cdef Py_ssize_t* t_idx = <Py_ssize_t*>malloc(nthreads * 8 * sizeof(Py_ssize_t))
    if not t_max or not t_idx:
        free(t_max)
        free(t_idx)
        k_center_gonzalez_c(pts, n, centers_out, dist, k)
        return

    cdef Py_ssize_t i, t, farthest_idx, picked
    cdef double last, d, max_d
    cdef int tid

    for i in prange(n, schedule='static'):
        dist[i] = INFINITY

    centers_out[0] = pts[0]

    for picked in range(1, <Py_ssize_t>k):
        last = centers_out[picked - 1]

        for t in range(nthreads):
            t_max[t * 8] = -1.0
            t_idx[t * 8] = 0

        # Parallel pass: update min-distances and accumulate per-thread argmax.
        # Each thread writes only to t_max[tid*8] and t_idx[tid*8], which are
        # on separate cache lines, so no data races or false sharing.
        for i in prange(n, schedule='static'):
            tid = omp_get_thread_num()
            d = fabs(pts[i] - last)
            if d < dist[i]:
                dist[i] = d
            if dist[i] > t_max[tid * 8]:
                t_max[tid * 8] = dist[i]
                t_idx[tid * 8] = i

        # Serial reduction over O(num_threads) thread-local results.
        max_d = -1.0
        farthest_idx = 0
        for t in range(nthreads):
            if t_max[t * 8] > max_d:
                max_d = t_max[t * 8]
                farthest_idx = t_idx[t * 8]

        centers_out[picked] = pts[farthest_idx]

        if max_d == 0.0:
            for i in range(picked + 1, <Py_ssize_t>k):
                centers_out[i] = pts[farthest_idx]
            free(t_max)
            free(t_idx)
            return

    free(t_max)
    free(t_idx)


DEF N_PARALLEL_THRESHOLD = 50000


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def k_center_gonzalez(points, int k, bint return_clusters=False, bint parallel=True):
    """
    Gonzalez greedy 2-approximation for 1D k-center.

    Iteratively selects the point farthest from all current centers, yielding a
    solution whose max-distance is at most 2× optimal. Runs in O(n*k) time.

    Parameters
    ----------
    points : array-like, float64
        1D array of data points (need not be sorted).
    k : int
        Number of centers to select. Clamped to len(points) if larger.
    return_clusters : bool, default False
        If False, return a list of center values.
        If True, return cluster ranges instead (see Returns).
    parallel : bool, default True
        Use OpenMP parallelism when n >= 50000. Falls back to serial otherwise.

    Returns
    -------
    list of float, length k
        The selected center values, when return_clusters=False.
    list of tuple[float, float], length k
        Each (min_val, max_val) is the range of points assigned to that center,
        when return_clusters=True.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pts_np = np.ascontiguousarray(points, dtype=np.float64)
    cdef double[::1] pts = pts_np
    cdef Py_ssize_t n = pts.shape[0]

    if n == 0 or k <= 0:
        return []
    if k > <int>n:
        k = <int>n

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] centers_arr_np = np.empty(k, dtype=np.float64)
    cdef double[::1] centers_arr = centers_arr_np

    cdef double* dist = <double*>malloc(n * sizeof(double))
    if not dist:
        raise MemoryError("Failed to allocate dist buffer")

    try:
        if parallel and n >= N_PARALLEL_THRESHOLD:
            with nogil:
                k_center_gonzalez_parallel_c(&pts[0], n, &centers_arr[0], dist, k)
        else:
            k_center_gonzalez_c(&pts[0], n, &centers_arr[0], dist, k)
    finally:
        free(dist)

    if return_clusters:
        return _to_cluster_ranges(pts_np, centers_arr_np, sort_centers=True)
    return centers_arr_np.tolist()


cdef struct Candidate:
    double dist
    double p
    double L
    double R

cdef inline void heap_push(Candidate* heap, int* heap_size, double dist, double p, double L, double R) noexcept nogil:
    cdef int i = heap_size[0]
    heap_size[0] += 1
    cdef int parent
    cdef Candidate temp

    heap[i].dist = dist
    heap[i].p = p
    heap[i].L = L
    heap[i].R = R

    while i > 0:
        parent = (i - 1) / 2
        if heap[parent].dist < heap[i].dist:
            temp = heap[parent]
            heap[parent] = heap[i]
            heap[i] = temp
            i = parent
        else:
            break

cdef inline Candidate heap_pop(Candidate* heap, int* heap_size) noexcept nogil:
    cdef Candidate result = heap[0]
    heap_size[0] -= 1
    if heap_size[0] == 0:
        return result

    heap[0] = heap[heap_size[0]]
    cdef int i = 0
    cdef int left, right, largest
    cdef Candidate temp

    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i

        if left < heap_size[0] and heap[left].dist > heap[largest].dist:
            largest = left
        if right < heap_size[0] and heap[right].dist > heap[largest].dist:
            largest = right

        if largest != i:
            temp = heap[i]
            heap[i] = heap[largest]
            heap[largest] = temp
            i = largest
        else:
            break

    return result


cdef inline void push_best_candidate_c(double[::1] pts, Py_ssize_t n, double L, double R, Candidate* heap, int* heap_size) noexcept nogil:
    cdef double mid = L + (R - L) / 2.0
    cdef Py_ssize_t idx = c_bisect_left(pts, mid, n)
    cdef double best_p = 0.0
    cdef double max_d = -1.0
    cdef bint found = 0
    cdef double p1, d1, p2, d2

    if idx < n:
        p1 = pts[idx]
        if L < p1 and p1 < R:
            d1 = p1 - L
            if R - p1 < d1: d1 = R - p1
            if d1 > max_d:
                max_d = d1
                best_p = p1
                found = 1

    if idx > 0:
        p2 = pts[idx - 1]
        if L < p2 and p2 < R:
            d2 = p2 - L
            if R - p2 < d2: d2 = R - p2
            if d2 > max_d:
                max_d = d2
                best_p = p2
                found = 1

    if found:
        heap_push(heap, heap_size, max_d, best_p, L, R)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def k_center_sort(points, int k, bint return_clusters=False):
    """
    2-approximation for 1D k-center optimized for sorted (or sortable) input.

    Seeds centers at the first and last point, then repeatedly splits the largest
    gap by inserting the data point farthest from the gap's endpoints, using a
    max-heap. Runs in O(k log k) after sorting.

    Parameters
    ----------
    points : array-like, float64
        1D array of data points. Must be sorted (or will be treated as if sorted
        after conversion to a contiguous array).
    k : int
        Number of centers to select.
    return_clusters : bool, default False
        If False, return a list of center values.
        If True, return cluster ranges instead (see Returns).

    Returns
    -------
    list of float, length <= k
        The selected center values, when return_clusters=False.
    list of tuple[float, float], length <= k
        Each (min_val, max_val) is the range of points assigned to that center,
        when return_clusters=True.
    """
    cdef Py_ssize_t n = len(points)
    if n == 0:
        return []

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pts_np = np.ascontiguousarray(points, dtype=np.float64)
    cdef double[::1] pts = pts_np

    if k >= n:
        centers_unique = sorted(list(set(points)))
        return _to_cluster_ranges(pts_np, centers_unique, sort_centers=False) if return_clusters else centers_unique

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] centers_arr_np = np.empty(k, dtype=np.float64)
    cdef double[::1] centers_arr = centers_arr_np

    centers_arr[0] = pts[0]
    if k == 1:
        return _to_cluster_ranges(pts_np, centers_arr_np[:1], sort_centers=False) if return_clusters else [centers_arr[0]]

    centers_arr[1] = pts[n - 1]
    if k == 2:
        return _to_cluster_ranges(pts_np, centers_arr_np[:2], sort_centers=False) if return_clusters else [centers_arr[0], centers_arr[1]]

    cdef int capacity = 2 * k
    cdef Candidate* heap = <Candidate*>malloc(capacity * sizeof(Candidate))
    if not heap:
        raise MemoryError("Failed to allocate heap memory")

    cdef int heap_size = 0
    cdef int num_centers = 2
    cdef Candidate best
    cdef int _step

    try:
        with nogil:
            push_best_candidate_c(pts, n, pts[0], pts[n - 1], heap, &heap_size)

            for _step in range(k - 2):
                if heap_size == 0:
                    break

                best = heap_pop(heap, &heap_size)
                centers_arr[num_centers] = best.p
                num_centers += 1

                push_best_candidate_c(pts, n, best.L, best.p, heap, &heap_size)
                push_best_candidate_c(pts, n, best.p, best.R, heap, &heap_size)

    finally:
        free(heap)

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] final_ctrs_np = centers_arr_np[:num_centers]
    if return_clusters:
        return _to_cluster_ranges(pts_np, final_ctrs_np, sort_centers=True)
    return [centers_arr[i] for i in range(num_centers)]

# min_val and inv_bucket_len are pre-computed by the wrapper (single pass).
# output_ptr must be pre-initialised to INFINITY by the caller.
cdef inline void k_center_heuristic_c(
    double* data_ptr,
    Py_ssize_t n,
    int B,
    double min_val,
    double inv_bucket_len,
    double* output_ptr,
    int* len_ptr,
    bint resize,
) noexcept nogil:
    cdef int i, bucket_idx
    cdef double curr_val

    for i in range(n):
        curr_val = data_ptr[i]
        bucket_idx = 2 * <int>((curr_val - min_val) * inv_bucket_len)
        if bucket_idx < 0:
            bucket_idx = 0

        if output_ptr[bucket_idx] == INFINITY:
            output_ptr[bucket_idx] = curr_val
            output_ptr[bucket_idx + 1] = curr_val
        elif curr_val < output_ptr[bucket_idx]:
            output_ptr[bucket_idx] = curr_val
        elif curr_val > output_ptr[bucket_idx + 1]:
            output_ptr[bucket_idx + 1] = curr_val

    if resize:
        for i in range(0, 2 * B + 1, 2):
            if output_ptr[i] != INFINITY:
                output_ptr[len_ptr[0]] = output_ptr[i]
                output_ptr[len_ptr[0] + 1] = output_ptr[i + 1]
                len_ptr[0] += 2


cdef inline void k_center_heuristic_parallel_c(
    double* data_ptr,
    Py_ssize_t n,
    int B,
    double min_val,
    double inv_bucket_len,
    double* output_ptr,   # 2*(B+1) elements; fully written by this function
) noexcept nogil:
    cdef int nthreads = omp_get_max_threads()
    # Each thread gets a contiguous 2*(B+1)-element slice of thread_buf.
    cdef int stride = 2 * (B + 1)
    cdef Py_ssize_t total_buf = <Py_ssize_t>nthreads * stride
    cdef double* thread_buf = <double*>malloc(total_buf * sizeof(double))
    cdef int dummy_len = 0

    if not thread_buf:
        # Fallback: serial with caller-allocated output_ptr (already init'd to INFINITY)
        k_center_heuristic_c(data_ptr, n, B, min_val, inv_bucket_len, output_ptr, &dummy_len, False)
        return

    cdef Py_ssize_t i
    cdef int t, b, bucket_idx, tid, base
    cdef double curr_val, t_min, t_max, g_min, g_max

    # Initialise per-thread buffers to INFINITY (parallel)
    for i in prange(total_buf, schedule='static'):
        thread_buf[i] = INFINITY

    # Parallel bucketing: thread tid writes to thread_buf[tid*stride + bucket_idx]
    for i in prange(n, schedule='static'):
        tid = omp_get_thread_num()
        curr_val = data_ptr[i]
        bucket_idx = 2 * <int>((curr_val - min_val) * inv_bucket_len)
        if bucket_idx < 0:
            bucket_idx = 0
        base = tid * stride + bucket_idx

        if thread_buf[base] == INFINITY:
            thread_buf[base] = curr_val
            thread_buf[base + 1] = curr_val
        elif curr_val < thread_buf[base]:
            thread_buf[base] = curr_val
        elif curr_val > thread_buf[base + 1]:
            thread_buf[base + 1] = curr_val

    # Serial merge: for each bucket take min-of-mins and max-of-maxes
    for b in range(B + 1):
        g_min = INFINITY
        g_max = -INFINITY
        for t in range(nthreads):
            t_min = thread_buf[t * stride + 2 * b]
            if t_min != INFINITY:
                if t_min < g_min:
                    g_min = t_min
                t_max = thread_buf[t * stride + 2 * b + 1]
                if t_max > g_max:
                    g_max = t_max
        output_ptr[2 * b] = g_min
        output_ptr[2 * b + 1] = g_min if g_min == INFINITY else g_max

    free(thread_buf)


def k_center_heuristic(double[::1] data, int k, int B = -1, bint return_clusters = True,
                       bint allow_relaxed = True,
                       double given_min = 0.0, double given_max = 0.0,
                       bint has_minmax = False,
                       bint run_in_parallel = True):
    """
    Grid-based coreset heuristic for large 1D k-center instances.

    Divides [min, max] into B uniform buckets and records the (min, max) of data
    points in each bucket. When allow_relaxed=True (default), the B bucket ranges
    are returned directly as a coreset — this is the fast path used by mdv.
    When allow_relaxed=False, k_center_sort is run on the compacted bucket endpoints
    to produce exactly k centers.

    Parameters
    ----------
    data : ndarray, shape (n,), float64, C-contiguous
        Input data points.
    k : int
        Number of centers (used to set default B and passed to k_center_sort when
        allow_relaxed=False).
    B : int, default -1
        Number of grid buckets. Defaults to 10*k when -1.
    return_clusters : bool, default True
        If True (and allow_relaxed=False), return cluster ranges via _to_cluster_ranges.
        Has no effect when allow_relaxed=True.
    allow_relaxed : bool, default True
        If True, return all B bucket ranges as a (B, 2) numpy array without running
        k_center_sort (the relaxed coreset path).
        If False, compact non-empty buckets and run k_center_sort to get k centers.
    given_min : float, default 0.0
        Pre-computed minimum of data. Used only when has_minmax=True.
    given_max : float, default 0.0
        Pre-computed maximum of data. Used only when has_minmax=True.
    has_minmax : bool, default False
        If True, use given_min/given_max instead of computing them from data.
    run_in_parallel : bool, default True
        Use OpenMP parallelism for bucketing when n >= 50000.

    Returns
    -------
    ndarray, shape (B, 2), float64
        Each row is (bucket_min, bucket_max) for one grid bucket. Empty buckets
        are filled with the global minimum as a dummy value. Returned when
        allow_relaxed=True.
    list of float or list of tuple[float, float]
        k centers (or cluster ranges) from k_center_sort. Returned when
        allow_relaxed=False.
    """
    cdef Py_ssize_t n, i
    cdef double* data_ptr
    cdef double* output_ptr
    cdef double min_val, max_val, range_val, inv_bucket_len, dummy_val
    cdef int output_length
    cdef cnp.ndarray[double, ndim=2, mode="c"] result_np
    cdef double[:, ::1] result_view
    cdef cnp.ndarray[double, ndim=1, mode="c"] compact_arr
    cdef double[::1] compact_view

    if B < 0:
        B = 10 * k

    n = data.shape[0]
    if n == 0:
        return np.array([], dtype=np.double)

    data_ptr = &data[0]

    if has_minmax:
        min_val = given_min
        max_val = given_max
    else:
        min_val = data_ptr[0]
        max_val = data_ptr[0]
        with nogil:
            for i in range(1, n):
                if data_ptr[i] < min_val:
                    min_val = data_ptr[i]
                elif data_ptr[i] > max_val:
                    max_val = data_ptr[i]

    range_val = max_val - min_val
    if range_val > 0.0:
        inv_bucket_len = B / range_val
    else:
        inv_bucket_len = 0.0

    # Size 2*(B+1): regular buckets 0..B-1 plus overflow slot B
    output_ptr = <double*>malloc((2 * B + 2) * sizeof(double))
    if not output_ptr:
        raise MemoryError("Failed to allocate output buffer")

    output_length = 0

    try:
        if allow_relaxed:
            if run_in_parallel and n >= N_PARALLEL_THRESHOLD:
                with nogil:
                    k_center_heuristic_parallel_c(data_ptr, n, B, min_val, inv_bucket_len, output_ptr)
            else:
                # Serial path — must pre-initialise sentinel before calling
                with nogil:
                    for i in range(2 * B + 2):
                        output_ptr[i] = INFINITY
                    k_center_heuristic_c(data_ptr, n, B, min_val, inv_bucket_len,
                                         output_ptr, &output_length, False)

            # Fold overflow bucket (index B) into bucket B-1
            if output_ptr[2 * B] != INFINITY:
                if output_ptr[2 * (B - 1)] == INFINITY:
                    output_ptr[2 * (B - 1)]     = output_ptr[2 * B]
                    output_ptr[2 * (B - 1) + 1] = output_ptr[2 * B + 1]
                else:
                    if output_ptr[2 * B] < output_ptr[2 * (B - 1)]:
                        output_ptr[2 * (B - 1)] = output_ptr[2 * B]
                    if output_ptr[2 * B + 1] > output_ptr[2 * (B - 1) + 1]:
                        output_ptr[2 * (B - 1) + 1] = output_ptr[2 * B + 1]

            # global_min is always in bucket 0; use as dummy for empty buckets
            dummy_val = output_ptr[0]

            result_np = np.empty((B, 2), dtype=np.float64)
            result_view = result_np
            with nogil:
                for i in range(B):
                    if output_ptr[2 * i] == INFINITY:
                        result_view[i, 0] = dummy_val
                        result_view[i, 1] = dummy_val
                    else:
                        result_view[i, 0] = output_ptr[2 * i]
                        result_view[i, 1] = output_ptr[2 * i + 1]
            return result_np

        else:
            # allow_relaxed=False: compact non-empty buckets, then run k_center_sort
            if run_in_parallel and n >= N_PARALLEL_THRESHOLD:
                # Parallel bucketing — fully writes output_ptr (B+1 pairs)
                with nogil:
                    k_center_heuristic_parallel_c(data_ptr, n, B, min_val, inv_bucket_len, output_ptr)
                # Compact non-empty buckets (including overflow slot B) in-place
                with nogil:
                    for i in range(B + 1):
                        if output_ptr[2 * i] != INFINITY:
                            output_ptr[output_length] = output_ptr[2 * i]
                            output_ptr[output_length + 1] = output_ptr[2 * i + 1]
                            output_length += 2
            else:
                with nogil:
                    for i in range(2 * B + 2):
                        output_ptr[i] = INFINITY
                    k_center_heuristic_c(data_ptr, n, B, min_val, inv_bucket_len,
                                         output_ptr, &output_length, True)

            compact_arr = np.empty(output_length, dtype=np.float64)
            compact_view = compact_arr
            with nogil:
                for i in range(output_length):
                    compact_view[i] = output_ptr[i]
            return k_center_sort(compact_arr, k, return_clusters=return_clusters)

    finally:
        free(output_ptr)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def k_center_heuristic_org(data, int k, B=None, bint return_clusters=False, bint allow_relaxed=True):
    """
    Original grid-based coreset heuristic for large 1D k-center instances.

    Reference implementation of k_center_heuristic. Divides [min, max] into B
    bins and collects per-bin (min, max) endpoints. When k >= number of non-empty
    endpoints or allow_relaxed=True, returns the coreset directly; otherwise runs
    k_center_sort. Prefer k_center_heuristic for production use (faster, returns
    a numpy array).

    Parameters
    ----------
    data : array-like, float64
        1D array of input data points.
    k : int
        Number of centers. Used to set default B and passed to k_center_sort.
    B : int or None, default None
        Number of grid bins. Defaults to 10*k when None.
    return_clusters : bool, default False
        If True, return cluster ranges from _to_cluster_ranges instead of center values.
    allow_relaxed : bool, default True
        If True, return the coreset endpoints directly when k >= coreset size.
        If False, always run k_center_sort to reduce to exactly k centers.

    Returns
    -------
    list of float
        Coreset endpoint values or k center values, when return_clusters=False.
    list of tuple[float, float]
        Cluster ranges (min, max) per center, when return_clusters=True.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] arr_np = np.ascontiguousarray(data, dtype=np.float64)
    cdef double[::1] arr = arr_np
    cdef Py_ssize_t n = arr.shape[0]

    if n == 0:
        return []

    cdef int B_int
    if B is None:
        B_int = 10 * k
    else:
        B_int = int(B)
        
    if B_int <= 0:
        B_int = max(1, 10 * max(1, k))

    cdef Py_ssize_t i, idx
    cdef double x
    cdef double global_min = arr[0]
    cdef double global_max = arr[0]

    with nogil:
        for i in range(1, n):
            x = arr[i]
            if x < global_min:
                global_min = x
            elif x > global_max:
                global_max = x

    if global_min == global_max:
        centers = [global_min]
        if return_clusters:
            return _to_cluster_ranges(arr_np, centers, sort_centers=False)
        return centers

    if k == 1:
        centers = [(global_min + global_max) / 2.0]
        if return_clusters:
            return _to_cluster_ranges(arr_np, centers, sort_centers=False)
        return centers

    cdef double bin_width = (global_max - global_min) / B_int
    cdef double c_inf = np.inf

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] bin_min_np = np.empty(B_int, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] bin_max_np = np.empty(B_int, dtype=np.float64)
    cdef double[::1] bin_min = bin_min_np
    cdef double[::1] bin_max = bin_max_np

    with nogil:
        for i in range(B_int):
            bin_min[i] = c_inf
            bin_max[i] = -c_inf

        bin_min[0] = global_min
        bin_max[B_int - 1] = global_max

        for i in range(n):
            x = arr[i]
            idx = <Py_ssize_t>((x - global_min) / bin_width)
            if idx >= B_int:
                idx = B_int - 1
            elif idx < 0:
                idx = 0

            if x < bin_min[idx]:
                bin_min[idx] = x
            if x > bin_max[idx]:
                bin_max[idx] = x

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] P_arr_np = np.empty(2 * B_int, dtype=np.float64)
    cdef double[::1] P_arr = P_arr_np
    cdef int p_count = 0

    with nogil:
        for i in range(B_int):
            if bin_min[i] != c_inf:
                P_arr[p_count] = bin_min[i]
                p_count += 1
                if bin_max[i] > bin_min[i]:
                    P_arr[p_count] = bin_max[i]
                    p_count += 1

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] P_final = P_arr_np[:p_count]

    if k >= p_count or allow_relaxed:
        if return_clusters:
            return _to_cluster_ranges(arr_np, P_final, sort_centers=False)
        return [P_final[i] for i in range(p_count)]

    cdef list final_centers = k_center_sort(P_final, k, return_clusters=False)
    if return_clusters:
        return _to_cluster_ranges(arr_np, final_centers, sort_centers=True)
    return final_centers
