# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import math
import time
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport fmax
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
import quant_cython_k_center as kc
from quant_cython_mdv import subdivide_intervals, max_variance
from quant_cython_kmeans import kmeans_wilber_1d
from quant_cython_asq_wilber import kmeans_asq_1d

import quiver_cpp

cnp.import_array()

cdef extern from "kmeans/kmeans.h":
    cdef cppclass kmeans_result:
        double cost
        vector[size_t] path

cdef extern from "asq_weight_fns.h":
    cdef cppclass ASQSubsetWeightFn:
        ASQSubsetWeightFn(const vector[double]& x,
                          const vector[double]& beta_in,
                          const vector[double]& gamma_in,
                          const vector[double]& alpha_in) nogil

    cdef cppclass kmeans_asq_subset_t:
        kmeans_asq_subset_t(size_t n, ASQSubsetWeightFn wfn) nogil
        unique_ptr[kmeans_result] compute(size_t k, double lower_bound, double upper_bound) nogil
        unique_ptr[kmeans_result] compute_with_binary_search(size_t k, double lower_bound, double upper_bound) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _asq_objective_c(
    double* w, Py_ssize_t n,
    double* Q, Py_ssize_t m
) noexcept nogil:
    """
    Adaptive stochastic quantization (ASQ) objective.

    w must be sorted ascending. Q must be sorted ascending.
    Uses a two-pointer sweep (O(n + m)) rather than per-element binary search.

    For each w[i], finds the enclosing quantizer interval [a, b] where
    a = largest Q value <= w[i] and b = smallest Q value >= w[i], then
    accumulates (w[i] - a) * (b - w[i]).  Returns the total sum.
    """
    cdef double total = 0.0
    cdef double val, lo, hi
    cdef Py_ssize_t i, q_idx

    if m == 0 or n == 0:
        return 0.0

    # Two-pointer: q_idx is the first index where Q[q_idx] >= w[i].
    # Because w is sorted, q_idx only ever advances forward.
    q_idx = 0
    for i in range(n):
        val = w[i]
        # Advance q_idx to first position where Q[q_idx] >= val
        while q_idx < m and Q[q_idx] < val:
            q_idx += 1

        if q_idx == 0:
            lo = Q[0]
            hi = Q[0]
        elif q_idx == m:
            lo = Q[m - 1]
            hi = Q[m - 1]
        else:
            lo = Q[q_idx - 1]
            hi = Q[q_idx]

        # Branchless: ARM64 FMAXNM maps to a single instruction, avoiding
        # branch misprediction when many points land on quantizer boundaries.
        total += fmax(0.0, (val - lo) * (hi - val))

    return total


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _asq_objective_weighted_c(
    double* w, Py_ssize_t n,
    double* Q, Py_ssize_t m,
    double* weights
) noexcept nogil:
    """
    Weighted ASQ objective.

    Same two-pointer sweep as _asq_objective_c, but each term is multiplied
    by weights[i] before accumulation.  w and Q must be sorted ascending.
    """
    cdef double total = 0.0
    cdef double val, lo, hi
    cdef Py_ssize_t i, q_idx

    if m == 0 or n == 0:
        return 0.0

    q_idx = 0
    for i in range(n):
        val = w[i]
        while q_idx < m and Q[q_idx] < val:
            q_idx += 1

        if q_idx == 0:
            lo = Q[0]
            hi = Q[0]
        elif q_idx == m:
            lo = Q[m - 1]
            hi = Q[m - 1]
        else:
            lo = Q[q_idx - 1]
            hi = Q[q_idx]

        total += weights[i] * fmax(0.0, (val - lo) * (hi - val))

    return total


@cython.boundscheck(False)
@cython.wraparound(False)
def asq_objective(w, Q, not_sorted=False, tol=1e-10, weights=None):
    """
    Adaptive stochastic quantization (ASQ) objective for a fixed quantizer set Q.

    Computes the sum over all data points w[i] of weights[i] * (w[i] - a) * (b - w[i]),
    where a is the largest quantizer value <= w[i] and b is the smallest >= w[i].
    If weights is None, all weights default to 1 (unweighted sum).

    w and Q may be unsorted (both are sorted internally) _only if not_sorted = True_
    Returns a float >= 0.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] w_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] Q_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weights_arr

    if w is None:
        return 0.0

    if not_sorted:
        w_arr = np.ascontiguousarray(np.sort(np.asarray(w, dtype=np.float64)))
    else:
        w_arr = np.ascontiguousarray(w, dtype=np.float64)
    if w_arr.size == 0:
        return 0.0

    if Q is None or len(Q) < 2:
        raise ValueError("Q must have at least 2 elements")

    if not_sorted:
        Q_arr = np.ascontiguousarray(np.sort(np.asarray(Q, dtype=np.float64)))
    else:
        Q_arr = np.ascontiguousarray(Q, dtype=np.float64)

    if Q_arr[0] > w_arr[0] + tol or Q_arr[len(Q_arr) - 1] < w_arr[len(w_arr) - 1] - tol:
        q_min = np.min(Q_arr)
        q_max = np.max(Q_arr)
        w_min = np.min(w_arr)
        w_max = np.max(w_arr)
        raise ValueError(f"Q must contain min and max of w; Q_min = {q_min}, Q_max = {q_max}, w_min = {w_min}, w_max = {w_max}")

    if weights is None:
        return _asq_objective_c(&w_arr[0], w_arr.shape[0], &Q_arr[0], Q_arr.shape[0])

    if not_sorted:
        raise ValueError("not_sorted=True with weights is not supported; sort w and weights together before calling")

    weights_arr = np.ascontiguousarray(weights, dtype=np.float64)
    if weights_arr.shape[0] != w_arr.shape[0]:
        raise ValueError(f"weights length {weights_arr.shape[0]} must match w length {w_arr.shape[0]}")

    return _asq_objective_weighted_c(
        &w_arr[0], w_arr.shape[0],
        &Q_arr[0], Q_arr.shape[0],
        &weights_arr[0],
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray _get_unique_c(cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] intervals):
    """Return a flat sorted array of non-empty interval endpoints."""
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


#################
#   Exact ADV   #
#################
def exact_adv(w_sorted, weights, int s, str estimate_method='vmix', vmix_estimate_scale=None, bint binary_search=True, int m_factor=10, double vmix_approx_scale=2.0, double vmix_approx_tol=1e-1):
    """
    Returns optimal (weighted) ADV from sorted w_sorted.  Returns both the set and the cost, as a tuple in that order.
    estimate_method: one of 'none', 'vmix', 'vmix_approx', 'interval_approx', 'vmix_aggresive'
    vmix_approx_scale: multiplier applied to (costs - costs1) when estimate_method='vmix_approx';
        upper_bound = vmix_approx_scale * (costs - costs1), lower_bound = (costs - costs1) / vmix_approx_scale
    vmix_approx_tol: tol passed to vmix_approx when estimate_method='vmix_approx'
    """
    cdef double lower_bound = 0.0
    cdef Py_ssize_t d = len(w_sorted)

    if s <= 2:
        raise ValueError("s must be at least 2")
    if s >= d:
        return w_sorted, 0

    cdef double upper_bound = -1.0
    cdef int scaling_factor = s # default scaling factor is s

    cdef double costs, costs1
    if estimate_method == 'vmix_aggresive':
        costs, costs1 = vmix(w_sorted, s, weights_in=weights, return_set=False, next_s=True)
        if costs > costs1:
            upper_bound = vmix_approx_scale * (costs - costs1)
            lower_bound = (costs - costs1) / vmix_approx_scale
        else:
            upper_bound = costs
    elif estimate_method == 'vmix_approx':
        costs1, costs = vmix_approx(w_sorted, s, m_factor*s, weights=weights, tol=vmix_approx_tol, return_set=False, next_s=True)
        if costs > costs1:
            upper_bound = vmix_approx_scale * s * (costs - costs1)
            lower_bound = s * (costs - costs1) / vmix_approx_scale
        else:
            upper_bound = costs
    elif estimate_method == 'interval_approx':
        costs1, costs = interval_approx_sorted(w_sorted, s, m_factor*s, weights=weights, return_set=False, next_s=True)
        if costs > costs1:
            upper_bound = vmix_approx_scale * (costs - costs1)
            lower_bound = (costs - costs1) / vmix_approx_scale
        else:
            upper_bound = costs
    elif estimate_method == 'vmix':
        if vmix_estimate_scale is not None:
            scaling_factor = vmix_estimate_scale
        upper_bound = scaling_factor * vmix(w_sorted, s, weights_in=weights, return_set=False)[0]
    elif estimate_method != 'none':
        raise ValueError(f"estimate_method must be one of 'none', 'vmix', 'vmix_approx', 'interval_approx', 'vmix_aggresive'; got {estimate_method!r}")

    cdef double cost
    cdef cnp.ndarray[cnp.int64_t, ndim=1] splits
    splits, cost = kmeans_asq_1d(w_sorted, weights, s-1, binary_search=binary_search, lower_bound=lower_bound, upper_bound=upper_bound)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] Q = np.zeros(s, dtype=np.float64)
    cdef Py_ssize_t i
    for i in range(splits.shape[0] - 1):
        Q[i] = w_sorted[splits[i]]
    Q[s-1] = w_sorted[d-1]
    if Q[0] != w_sorted[0]:
        raise ValueError(f"Error in kmeans_asq_1d call, did not return first element of w_sorted. w_sorted[0]={w_sorted[0]}, splits[0]={splits[0]} so Q[0] = {Q[0]}")
    return Q, cost


##################
# VMIX Functions #
##################

# AoS (Array of Structs) layout: 6 doubles = 48 bytes.
# C(prev, j) accesses all 6 prefix-sum fields at index j; packing them into one
# struct means 1-2 cache-line loads instead of 6 from SoA's discontiguous arrays.
cdef struct PrefixEntry:
    double w
    double beta
    double beta_c
    double gamma
    double gamma_c
    double alpha

cdef extern from "stdlib.h" nogil:
    int posix_memalign(void** memptr, size_t alignment, size_t size)


cdef inline void init_sums_aos(double* w, double* weights,
                                PrefixEntry* E, Py_ssize_t d) noexcept nogil:
    cdef Py_ssize_t i
    cdef double bs, bkc, by, bt, bv
    cdef double gs, gkc, gy, gt, gv

    E[0].w = w[0]
    E[0].alpha = weights[0]
    bs = weights[0] * w[0]; bkc = 0.0
    E[0].beta = bs; E[0].beta_c = 0.0
    gs = weights[0] * w[0] * w[0]; gkc = 0.0
    E[0].gamma = gs; E[0].gamma_c = 0.0

    for i in range(1, d):
        E[i].w = w[i]
        E[i].alpha = E[i-1].alpha + weights[i]

        bv = weights[i] * w[i]
        by = bv - bkc
        bt = bs + by
        bkc = (bt - bs) - by
        bs = bt
        E[i].beta = bs
        E[i].beta_c = -bkc

        gv = weights[i] * w[i] * w[i]
        gy = gv - gkc
        gt = gs + gy
        gkc = (gt - gs) - gy
        gs = gt
        E[i].gamma = gs
        E[i].gamma_c = -gkc


cdef inline double C_aos(PrefixEntry* E, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef double beta_diff = (E[j].beta - E[i].beta) + (E[j].beta_c - E[i].beta_c)
    cdef double gamma_diff = (E[j].gamma - E[i].gamma) + (E[j].gamma_c - E[i].gamma_c)
    cdef double alpha_diff = E[j].alpha - E[i].alpha
    return (E[i].w + E[j].w) * beta_diff - gamma_diff - E[i].w * E[j].w * alpha_diff


cdef inline double C_with_prev(double w_i, double beta_i, double beta_c_i,
                                double gamma_i, double gamma_c_i, double alpha_i,
                                PrefixEntry* E, Py_ssize_t j) noexcept nogil:
    cdef double beta_diff = (E[j].beta - beta_i) + (E[j].beta_c - beta_c_i)
    cdef double gamma_diff = (E[j].gamma - gamma_i) + (E[j].gamma_c - gamma_c_i)
    cdef double alpha_diff = E[j].alpha - alpha_i
    return (w_i + E[j].w) * beta_diff - gamma_diff - w_i * E[j].w * alpha_diff


cdef inline int check_vmix_aos(PrefixEntry* E, int s, double v, Py_ssize_t d,
                                Py_ssize_t* result_ptr) noexcept nogil:
    cdef int count = 1
    cdef Py_ssize_t prev = 0, lo, hi, mid, offset, last_prev
    cdef Py_ssize_t step = d / s  # improvement 3: adaptive initial step
    cdef double cost
    cdef double w_prev, beta_prev, beta_c_prev, gamma_prev, gamma_c_prev, alpha_prev

    if step < 1:
        step = 1

    result_ptr[0] = 0
    while count < s:
        last_prev = prev

        w_prev = E[prev].w
        beta_prev = E[prev].beta
        beta_c_prev = E[prev].beta_c
        gamma_prev = E[prev].gamma
        gamma_c_prev = E[prev].gamma_c
        alpha_prev = E[prev].alpha

        lo = prev
        offset = step
        hi = prev + offset

        # Exponential search
        while hi < d:
            if C_with_prev(w_prev, beta_prev, beta_c_prev,
                           gamma_prev, gamma_c_prev, alpha_prev, E, hi) >= v:
                break
            lo = hi
            offset *= 2
            hi = prev + offset

        if hi >= d:
            if C_with_prev(w_prev, beta_prev, beta_c_prev,
                           gamma_prev, gamma_c_prev, alpha_prev, E, d-1) <= v:
                result_ptr[count] = d-1
                return count + 1
            hi = d - 1

        while hi > lo + 1:
            mid = lo + (hi - lo) / 2
            cost = C_with_prev(w_prev, beta_prev, beta_c_prev,
                               gamma_prev, gamma_c_prev, alpha_prev, E, mid)
            if cost > v:
                hi = mid
            elif cost < v:
                lo = mid
            else:
                prev = mid
                result_ptr[count] = prev
                break
        else:
            prev = lo
            result_ptr[count] = prev

        step = prev - last_prev
        if step < 1:
            step = 1

        count += 1
    return s + 1


cdef inline double vmix_c_aos(PrefixEntry* E, int s, Py_ssize_t d, double tol,
                               Py_ssize_t* result_ptr, bint return_set) noexcept nogil:
    cdef Py_ssize_t i
    if s >= d:
        for i in range(d):
            result_ptr[i] = i
        return 0

    cdef double lo = 0, hi, mid, chunk_cost
    cdef Py_ssize_t k, j_end, j_start
    cdef int count_check
    cdef int hi_count = 1
    cdef double mult_factor = 1.0 / 16.0

    if s > 2:

        k = 1 + (d - 1) // (s - 1)
        hi = 0.0
        j_start = 0
        j_end = 0
        while j_start < d - 1:
            j_end = j_start + k
            if j_end >= d:
                j_end = d - 1
            chunk_cost = C_aos(E, j_start, j_end)
            if chunk_cost > hi:
                hi = chunk_cost
            j_start = j_end
    else:
        hi = C_aos(E, 0, d-1)

    cdef Py_ssize_t* best_result = <Py_ssize_t*>malloc(s * sizeof(Py_ssize_t))
    cdef bint has_best_result = False

    while hi_count < s - 1 and hi - lo > tol * hi:
        mid = lo + (hi - lo) * mult_factor
        count_check = check_vmix_aos(E, s, mid, d, result_ptr)
        if count_check <= s:
            hi = mid
            hi_count = count_check
            memcpy(best_result, result_ptr, s * sizeof(Py_ssize_t))
            has_best_result = True
        else:
            lo = mid
            mult_factor = 0.5

    if return_set:
        if has_best_result:
            memcpy(result_ptr, best_result, s * sizeof(Py_ssize_t))
        else:
            check_vmix_aos(E, s, hi, d, result_ptr)

    free(best_result)
    return hi


cdef inline double C(int i, int j, double* w,
                     double* beta, double* beta_c,
                     double* gamma, double* gamma_c,
                     double* alpha) noexcept nogil:
    """
    Compute C[i,j] using compensated differences on the prefix sums.
    beta_c[k] and gamma_c[k] hold Kahan compensation terms so that
    true_sum[k] ≈ beta[k] + beta_c[k] (resp. gamma[k] + gamma_c[k]).
    Taking diffs via (main[j]-main[i]) + (comp[j]-comp[i]) preserves ~double-double
    precision, avoiding the catastrophic cancellation that naive differences suffer
    when w values are large relative to the interval span.
    """
    cdef double beta_diff = (beta[j] - beta[i]) + (beta_c[j] - beta_c[i])
    cdef double gamma_diff = (gamma[j] - gamma[i]) + (gamma_c[j] - gamma_c[i])
    cdef double alpha_diff = alpha[j] - alpha[i]
    return (w[i] + w[j])*beta_diff - gamma_diff - w[i]*w[j]*alpha_diff


cdef inline int check_vmix(double* w, int s, double v,
                            double* beta, double* beta_c,
                            double* gamma, double* gamma_c,
                            double* alpha, Py_ssize_t d,
                            Py_ssize_t* result_ptr) noexcept nogil:
    """
    Return number of points needed if vmix(w, s) <= v, s+1 otherwise
    Assumes the values of w are sorted and unique
    """
    cdef int count = 1
    cdef Py_ssize_t prev = 0, lo, hi, mid
    cdef double cost

    result_ptr[0] = 0
    while count < s:
        lo = prev
        hi = prev + 1
        while hi < d and C(prev, hi, w, beta, beta_c, gamma, gamma_c, alpha) < v:
            lo = hi
            hi *= 2
        if hi >= d:
            # check if can feasibly include everything else, and if not truncate hi
            if C(prev, d-1, w, beta, beta_c, gamma, gamma_c, alpha) <= v:
                result_ptr[count] = d-1
                return count + 1
            else:
                hi = d - 1

        while hi > lo + 1:
            mid = lo + (hi - lo) / 2 # Overflow-safe midpoint calculation
            cost = C(prev, mid, w, beta, beta_c, gamma, gamma_c, alpha)
            if cost > v:
                hi = mid
            elif cost < v:
                lo = mid
            elif cost == v:
                # C is non-decreasing in j, so mid is the largest valid position here
                prev = mid
                result_ptr[count] = prev
                break
        else:
            # normal exit (no break): lo is the largest j where C(prev, j) <= v
            prev = lo
            result_ptr[count] = prev
        count += 1
    # reached budget of s points without reaching end, return False (i.e. s + 1)
    return s + 1

cdef inline int check_vmix_linear(double* w, int s, double v,
                                   double* beta, double* beta_c,
                                   double* gamma, double* gamma_c,
                                   double* alpha, Py_ssize_t d,
                                   Py_ssize_t* result_ptr) noexcept nogil:
    cdef int count = 1
    cdef Py_ssize_t prev = 0, i

    result_ptr[0] = 0
    while count < s:
        if C(prev, d-1, w, beta, beta_c, gamma, gamma_c, alpha) <= v:
            result_ptr[count] = d-1
            return count + 1
        i = prev + 1
        while i < d and C(prev, i, w, beta, beta_c, gamma, gamma_c, alpha) <= v:
            i += 1
        prev = i - 1
        result_ptr[count] = prev
        count += 1
    return s + 1


cdef inline double vmix_c(double* w, int s,
                          double* beta, double* beta_c,
                          double* gamma, double* gamma_c,
                          double* alpha, Py_ssize_t d, double tol,
                          Py_ssize_t* result_ptr, bint return_set) noexcept nogil:
    """
    Main implementation of vmix
    s MUST be less than d, or may infinite loop
    """

    cdef Py_ssize_t i
    if s >= d:
        for i in range(d):
            result_ptr[i] = i
        return 0

    cdef double lo = 0
    cdef double hi
    cdef double mid
    cdef double chunk_cost
    cdef Py_ssize_t k, j_end, j_start, i_lb
    cdef int count_check
    cdef int it = 0
    cdef int hi_count = 1
    cdef double mult_factor = 1 / 16.0

    if s > 2:
        k = 1 + (d - 1) // (s - 1)  # ceil((d-1) / (s-1))
        hi = 0.0
        j_start = 0
        j_end = 0
        while j_start < d - 1:
            j_end = j_start + k
            if j_end >= d:
                j_end = d-1
            chunk_cost = C(j_start, j_end, w, beta, beta_c, gamma, gamma_c, alpha)
            if chunk_cost > hi:
                hi = chunk_cost
            j_start = j_end

    else:
        hi = C(0, d-1, w, beta, beta_c, gamma, gamma_c, alpha)

    while hi_count < s - 1 and hi - lo > tol * hi: # some slack
        mid = lo +  (hi - lo) * mult_factor
        count_check = check_vmix(w, s, mid, beta, beta_c, gamma, gamma_c, alpha, d, result_ptr)
        if count_check <= s:
            hi = mid
            hi_count = count_check
        else:
            lo = mid
            mult_factor = 0.5
        it += 1

    if return_set:
        # make sure result_ptr is properly filled by running check_vmix again
        check_vmix(w, s, hi, beta, beta_c, gamma, gamma_c, alpha, d, result_ptr)
    return hi


cdef inline void init_sums(double* w, double* weights,
                           double* beta, double* beta_c,
                           double* gamma, double* gamma_c,
                           double* alpha, Py_ssize_t d) noexcept nogil:
    """
    Build prefix sums with Kahan compensation for beta and gamma.
    alpha is an ordinary prefix sum — weights are typically small and free of
    catastrophic cancellation risk. Storing the compensation term lets C() compute
    (beta[j]-beta[i]) and (gamma[j]-gamma[i]) as compensated differences, which
    retains full precision even when the raw prefix sums are huge compared to the
    result of the subtraction.
    """
    cdef Py_ssize_t i
    cdef double bs, bkc, by, bt, bv
    cdef double gs, gkc, gy, gt, gv

    alpha[0] = weights[0]
    for i in range(1, d):
        alpha[i] = alpha[i-1] + weights[i]

    # Kahan accumulators. Stored convention: beta_c[i] = -bkc so that
    # true_sum[i] ≈ beta[i] + beta_c[i].
    bs = weights[0] * w[0]
    bkc = 0.0
    beta[0] = bs
    beta_c[0] = 0.0

    gs = weights[0] * w[0] * w[0]
    gkc = 0.0
    gamma[0] = gs
    gamma_c[0] = 0.0

    for i in range(1, d):
        bv = weights[i] * w[i]
        by = bv - bkc
        bt = bs + by
        bkc = (bt - bs) - by
        bs = bt
        beta[i] = bs
        beta_c[i] = -bkc

        gv = weights[i] * w[i] * w[i]
        gy = gv - gkc
        gt = gs + gy
        gkc = (gt - gs) - gy
        gs = gt
        gamma[i] = gs
        gamma_c[i] = -gkc


@cython.boundscheck(False)
@cython.wraparound(False)
def vmix(w_in, int s, weights_in=None, double tol=1e-1, bint return_set = False, bint next_s = False):
    """
    Find the vmix objective on _sorted_ vector w and s
    If next_s = True, returns the cost at s and s+1 (regardless of the value of return_set)
    """
    cdef Py_ssize_t d = len(w_in)

    if s <= 2:
        raise ValueError("s must be at least 2")
    if s >= d:
        # when s >= d, get 0 cost
        if next_s:
            return 0, 0
        else:
            return 0, w_in

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] w = np.ascontiguousarray(w_in, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] weights
    if weights_in is None:
        weights = np.ones(d)
    else:
        weights = np.ascontiguousarray(weights_in, dtype=np.float64)

    cdef PrefixEntry* E = NULL
    if posix_memalign(<void**>&E, 64, d * sizeof(PrefixEntry)) != 0:
        raise MemoryError()
    init_sums_aos(&w[0], &weights[0], E, d)

    # Allocate s+1 slots: the next_s path calls vmix_c_aos with s+1, which writes
    # up to result_ptr[s] via check_vmix_aos, so s slots would overflow.
    cdef Py_ssize_t* result_ptr = <Py_ssize_t*>malloc((s + 1) * sizeof(Py_ssize_t))
    cdef double value = vmix_c_aos(E, s, d, tol, result_ptr, return_set)
    cdef double value_next
    if next_s:
        value_next = vmix_c_aos(E, s + 1, d, tol, result_ptr, False)
        free(E)
        free(result_ptr)
        return value, value_next

    free(E)

    cdef Py_ssize_t last_index
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] res_set
    if return_set:
        last_index = 0
        res_set = np.empty(s, dtype=np.float64)
        while last_index < s and result_ptr[last_index] < d:
            res_set[last_index] = w[result_ptr[last_index]]
            last_index += 1
            if result_ptr[last_index - 1] == d - 1:
                break

        free(result_ptr)
        return value, res_set[:last_index]
    else:
        free(result_ptr)
        return value, None


cdef inline double vmix_approx_c(double* w_sorted, double* weights, int s, Py_ssize_t m, Py_ssize_t d, Py_ssize_t* result_ptr, double tol, bint next_s, double* next_s_cost, bint interpolation_search, bint vmix_estimate) noexcept nogil:
    cdef PrefixEntry* E_full = NULL
    if posix_memalign(<void**>&E_full, 64, d * sizeof(PrefixEntry)) != 0:
        E_full = <PrefixEntry*>malloc(d * sizeof(PrefixEntry))
    init_sums_aos(w_sorted, weights, E_full, d)

    # get the initial vmix solution with size m
    cdef Py_ssize_t* vmix_sol = <Py_ssize_t*>malloc(m * sizeof(Py_ssize_t))
    vmix_c_aos(E_full, m, d, tol, vmix_sol, True)

    # vmix_c_aos may terminate early (fewer than m points) when it reaches index d-1.
    # check_vmix_aos always writes d-1 as the final entry when returning True, so scan
    # for that sentinel to find how many entries were actually written.
    cdef Py_ssize_t actual_m = 0
    while actual_m < m:
        actual_m += 1
        if vmix_sol[actual_m - 1] == d - 1:
            break

    cdef vector[double] x, alpha, beta, beta_c, gamma, gamma_c
    x.resize(actual_m)
    alpha.resize(actual_m)
    beta.resize(actual_m)
    beta_c.resize(actual_m)
    gamma.resize(actual_m)
    gamma_c.resize(actual_m)

    cdef Py_ssize_t i
    cdef Py_ssize_t curr_indx
    for i in range(actual_m):
        curr_indx = vmix_sol[i]
        x[i] = E_full[curr_indx].w
        alpha[i] = E_full[curr_indx].alpha
        beta[i] = E_full[curr_indx].beta
        beta_c[i] = E_full[curr_indx].beta_c
        gamma[i] = E_full[curr_indx].gamma
        gamma_c[i] = E_full[curr_indx].gamma_c

    free(E_full)

    if actual_m <= s:
        for i in range(actual_m):
            result_ptr[i] = vmix_sol[i]
        for i in range(actual_m, s):
            result_ptr[i] = vmix_sol[actual_m - 1]
        free(vmix_sol)
        return 0.0

    cdef double lower_bound = 0.0
    cdef double upper_bound = -1.0
    cdef PrefixEntry* E_compact = NULL
    if vmix_estimate:
        # Build compact AoS from the extracted vectors for vmix_c_aos
        if posix_memalign(<void**>&E_compact, 64, actual_m * sizeof(PrefixEntry)) != 0:
            E_compact = <PrefixEntry*>malloc(actual_m * sizeof(PrefixEntry))
        for i in range(actual_m):
            E_compact[i].w = x[i]
            E_compact[i].alpha = alpha[i]
            E_compact[i].beta = beta[i]
            E_compact[i].beta_c = beta_c[i]
            E_compact[i].gamma = gamma[i]
            E_compact[i].gamma_c = gamma_c[i]
        upper_bound = s * vmix_c_aos(E_compact, s, actual_m, tol, result_ptr, False)
        free(E_compact)

    # Solve on points from vmix_sol
    cdef ASQSubsetWeightFn* wfn = new ASQSubsetWeightFn(x, beta, gamma, alpha)
    cdef kmeans_asq_subset_t* solver = new kmeans_asq_subset_t(<size_t>actual_m, wfn[0])
    del wfn

    cdef unique_ptr[kmeans_result] km_result
    cdef unique_ptr[kmeans_result] kmeans_result2
    cdef kmeans_result* res_ptr2
    if not interpolation_search:
        km_result = solver.compute_with_binary_search(<size_t>(s - 1), lower_bound, upper_bound)
    else:
        km_result = solver.compute(<size_t>(s - 1), lower_bound, upper_bound)
    if next_s:
        if not interpolation_search:
            kmeans_result2 = solver.compute_with_binary_search(<size_t>s, lower_bound, upper_bound)
        else:
            kmeans_result2 = solver.compute(<size_t>s, lower_bound, upper_bound)
        res_ptr2 = kmeans_result2.get()
        next_s_cost[0] = res_ptr2.cost
    del solver

    cdef kmeans_result* res_ptr = km_result.get()
    cdef double cost = res_ptr.cost
    cdef vector[size_t]* pv = &res_ptr.path
    cdef Py_ssize_t npath = <Py_ssize_t>pv.size()

    cdef Py_ssize_t write_n = npath if npath <= s else s
    for i in range(write_n):
        result_ptr[i] = vmix_sol[pv[0][npath - 1 - i]]
    for i in range(write_n, s):
        result_ptr[i] = vmix_sol[actual_m - 1]

    free(vmix_sol)
    return cost


#####################
# Improved Approxes #
#####################

cdef inline double interval_approx_c(double* w, double* weights, int s, double wmin, double wmax, Py_ssize_t m, Py_ssize_t d, Py_ssize_t* result_ptr, double tol, bint next_s, double* next_s_cost) noexcept nogil:
    cdef double L = (wmax - wmin) / m
    cdef double inv_L = m / (wmax - wmin)
    cdef Py_ssize_t m_size = m + 1
    cdef Py_ssize_t i

    cdef vector[double] alpha, beta, beta_c, gamma, gamma_c, bucket_a, bucket_b, bucket_g
    alpha.resize(m_size)
    beta.resize(m_size)
    beta_c.resize(m_size)
    gamma.resize(m_size)
    gamma_c.resize(m_size)
    bucket_a.resize(m_size)
    bucket_b.resize(m_size)
    bucket_g.resize(m_size)
    for i in range(m_size):
        bucket_a[i] = 0.0

    cdef double wi, weights_i
    cdef Py_ssize_t idx
    for i in range(d):
        wi = w[i]
        weights_i = weights[i]
        idx = <Py_ssize_t>((wi - wmin) * inv_L) + 1
        if wmin == wi:
            idx = 0
        if idx >= m + 1:
            idx = m
        bucket_a[idx] += weights_i
        bucket_b[idx] += weights_i * wi
        bucket_g[idx] += weights_i * wi * wi

    # Build full prefix sums. alpha is naive (weights are modest).
    # beta and gamma use Kahan compensation; the residual is stored as
    # beta_c[i]/gamma_c[i] so true_sum ≈ main + comp. This protects C()'s
    # subtractions from catastrophic cancellation when w is large.
    alpha[0] = bucket_a[0]
    cdef double bs = bucket_b[0], bkc = 0.0, by, bt
    cdef double gs = bucket_g[0], gkc = 0.0, gy, gt
    beta[0] = bs
    beta_c[0] = 0.0
    gamma[0] = gs
    gamma_c[0] = 0.0
    for i in range(1, m_size):
        alpha[i] = alpha[i - 1] + bucket_a[i]

        by = bucket_b[i] - bkc
        bt = bs + by
        bkc = (bt - bs) - by
        bs = bt
        beta[i] = bs
        beta_c[i] = -bkc

        gy = bucket_g[i] - gkc
        gt = gs + gy
        gkc = (gt - gs) - gy
        gs = gt
        gamma[i] = gs
        gamma_c[i] = -gkc

    cdef vector[Py_ssize_t] full_indices  # maps compacted idx → full grid idx
    cdef vector[double] cx, calpha, cbeta, cbeta_c, cgamma, cgamma_c  # compacted arrays

    # Bucket 0 always has data (wmin maps to idx=0).
    full_indices.push_back(0)
    for i in range(1, m_size):
        if bucket_a[i] > 0.0:
            full_indices.push_back(i)

    cdef Py_ssize_t mp = <Py_ssize_t>full_indices.size()
    cx.resize(mp)
    calpha.resize(mp)
    cbeta.resize(mp)
    cbeta_c.resize(mp)
    cgamma.resize(mp)
    cgamma_c.resize(mp)

    cdef Py_ssize_t fi
    for i in range(mp - 1):
        fi = full_indices[i]
        cx[i] = wmin + L * fi
        calpha[i] = alpha[fi]
        cbeta[i] = beta[fi]
        cbeta_c[i] = beta_c[fi]
        cgamma[i] = gamma[fi]
        cgamma_c[i] = gamma_c[fi]
    cx[mp - 1] = wmax
    calpha[mp - 1] = alpha[m_size-1]
    cbeta[mp - 1] = beta[m_size-1]
    cbeta_c[mp - 1] = beta_c[m_size-1]
    cgamma[mp - 1] = gamma[m_size-1]
    cgamma_c[mp - 1] = gamma_c[m_size-1]

    # If we have fewer compacted points than s, every non-empty grid point
    # is already a quantization point — return them directly.
    if mp <= s:
        for i in range(mp):
            result_ptr[i] = full_indices[i]
        for i in range(mp, s):
            result_ptr[i] = m - 1
        next_s_cost[0] = 0.0
        return 0.0

    # Compute vmix estimate on compacted grid (better bound, avoids negative
    # values from empty-bucket boundary effects on the full grid)
    cdef double vmix_estimate = s * vmix_c(cx.data(), s, cbeta.data(), cbeta_c.data(), cgamma.data(), cgamma_c.data(), calpha.data(), mp, tol, result_ptr, False)
    cdef double lower_bound = 0.0
    cdef double upper_bound = vmix_estimate

    # Solve on the compacted grid
    cdef ASQSubsetWeightFn* wfn = new ASQSubsetWeightFn(cx, cbeta, cgamma, calpha)
    cdef kmeans_asq_subset_t* solver = new kmeans_asq_subset_t(<size_t>mp, wfn[0])
    del wfn

    cdef unique_ptr[kmeans_result] km_result
    cdef unique_ptr[kmeans_result] km_result2
    cdef kmeans_result* res_ptr2
    km_result = solver.compute_with_binary_search(<size_t>(s - 1), lower_bound, upper_bound)
    if next_s:
        km_result2 = solver.compute_with_binary_search(<size_t>s, lower_bound, upper_bound)
        res_ptr2 = km_result2.get()
        next_s_cost[0] = res_ptr2.cost
    del solver

    # Extract path (stored reversed: last split first).
    # Map compacted indices back to full grid indices so the wrapper can
    # reconstruct Q values via  Q[i] = wmin + L * result_ptr[i].
    cdef kmeans_result* res_ptr = km_result.get()
    cdef double cost = res_ptr.cost
    cdef vector[size_t]* pv = &res_ptr.path
    cdef Py_ssize_t npath = <Py_ssize_t>pv.size()

    cdef Py_ssize_t write_n = npath
    for i in range(write_n):
        result_ptr[i] = full_indices[pv[0][npath - 1 - i]]

    return cost


cdef inline double interval_approx_c_sorted(double* w_sorted, double* weights, int s, double wmin, double wmax, Py_ssize_t m, Py_ssize_t d, Py_ssize_t* result_ptr, double tol, bint next_s, double* next_s_cost, bint use_vmix_approx) noexcept nogil:
    """
    Sorted-input specialization of interval_approx_c.

    Requires w_sorted to be non-decreasing. Because duplicates-to-a-bucket are
    contiguous, we can stream through the input once and emit the compacted
    (non-empty-bucket) prefix sums directly, avoiding:
      - the m_size-sized bucket_a/bucket_b/bucket_g arrays,
      - the full alpha/beta/gamma prefix-sum arrays,
      - the separate compaction pass over the grid.

    The running Kahan accumulator is advanced one element at a time; at each
    bucket transition we snapshot the accumulator into the compacted arrays
    (that snapshot is the prefix sum at the end of the previous bucket).
    """
    cdef double L = (wmax - wmin) / m
    cdef double inv_L = m / (wmax - wmin)
    cdef Py_ssize_t i

    # Simple per-element accumulators within the current bucket — no Kahan.
    # Kahan is applied once per bucket transition (O(m) total) when folding
    # bucket_bs/bucket_gs into the compensated prefix sums, matching the
    # original's approach of simple scatter-accumulate then Kahan prefix sum.
    cdef double alpha_sum = 0.0
    cdef double bucket_bs = 0.0, bucket_gs = 0.0
    cdef double prefix_bs = 0.0, prefix_bkc = 0.0, by, bt
    cdef double prefix_gs = 0.0, prefix_gkc = 0.0, gy, gt

    cdef Py_ssize_t reserve_n = d if d < m + 1 else m + 1
    cdef vector[Py_ssize_t] full_indices
    cdef vector[double] cx, calpha, cbeta, cbeta_c, cgamma, cgamma_c
    full_indices.reserve(reserve_n)
    cx.reserve(reserve_n)
    calpha.reserve(reserve_n)
    cbeta.reserve(reserve_n)
    cbeta_c.reserve(reserve_n)
    cgamma.reserve(reserve_n)
    cgamma_c.reserve(reserve_n)

    cdef Py_ssize_t prev_idx = -1
    cdef Py_ssize_t idx
    cdef double wi, weights_i, wwi

    for i in range(d):
        wi = w_sorted[i]
        weights_i = weights[i]

        idx = <Py_ssize_t>((wi - wmin) * inv_L) + 1
        if wmin == wi:
            idx = 0
        if idx >= m + 1:
            idx = m

        if idx != prev_idx:
            if prev_idx >= 0:
                # Close previous bucket: fold its simple sum into the
                # compensated prefix totals with one Kahan step each.
                by = bucket_bs - prefix_bkc
                bt = prefix_bs + by
                prefix_bkc = (bt - prefix_bs) - by
                prefix_bs = bt

                gy = bucket_gs - prefix_gkc
                gt = prefix_gs + gy
                prefix_gkc = (gt - prefix_gs) - gy
                prefix_gs = gt

                full_indices.push_back(prev_idx)
                cx.push_back(wmin + L * prev_idx)
                calpha.push_back(alpha_sum)
                cbeta.push_back(prefix_bs)
                cbeta_c.push_back(-prefix_bkc)
                cgamma.push_back(prefix_gs)
                cgamma_c.push_back(-prefix_gkc)

                bucket_bs = 0.0
                bucket_gs = 0.0
            prev_idx = idx

        # Accumulate into bucket sums — plain adds, no dependency chain.
        wwi = weights_i * wi
        alpha_sum += weights_i
        bucket_bs += wwi
        bucket_gs += wwi * wi

    # Close the final bucket.
    by = bucket_bs - prefix_bkc
    bt = prefix_bs + by
    prefix_bkc = (bt - prefix_bs) - by
    prefix_bs = bt

    gy = bucket_gs - prefix_gkc
    gt = prefix_gs + gy
    prefix_gkc = (gt - prefix_gs) - gy
    prefix_gs = gt

    full_indices.push_back(prev_idx)
    cx.push_back(wmax)
    calpha.push_back(alpha_sum)
    cbeta.push_back(prefix_bs)
    cbeta_c.push_back(-prefix_bkc)
    cgamma.push_back(prefix_gs)
    cgamma_c.push_back(-prefix_gkc)

    cdef Py_ssize_t mp = <Py_ssize_t>full_indices.size()

    # Fewer distinct non-empty buckets than requested quantization points: every
    # non-empty grid point is already a quantization point.
    if mp <= s:
        for i in range(mp):
            result_ptr[i] = full_indices[i]
        for i in range(mp, s):
            result_ptr[i] = m - 1
        next_s_cost[0] = 0.0
        return 0.0

    cdef double lower_bound = 0.0
    cdef double upper_bound
    cdef Py_ssize_t vmix_alloc_n
    cdef Py_ssize_t* vmix_result_tmp
    cdef double vmix_next_cost_est
    cdef double vmix_cost_est
    if use_vmix_approx:
        vmix_alloc_n = m + 10 if m + 10 > s else s
        vmix_result_tmp = <Py_ssize_t*>malloc(vmix_alloc_n * sizeof(Py_ssize_t))
        vmix_next_cost_est = 0.0
        vmix_cost_est = vmix_approx_c(w_sorted, weights, s, m, d, vmix_result_tmp, tol, True, &vmix_next_cost_est, False, True)
        free(vmix_result_tmp)
        if vmix_cost_est > vmix_next_cost_est:
            upper_bound = 2.0 * (vmix_cost_est - vmix_next_cost_est)
            lower_bound = (vmix_cost_est - vmix_next_cost_est) / 2.0
        else:
            upper_bound = vmix_cost_est
    else:
        upper_bound = s * vmix_c(cx.data(), s, cbeta.data(), cbeta_c.data(), cgamma.data(), cgamma_c.data(), calpha.data(), mp, tol, result_ptr, False)

    cdef ASQSubsetWeightFn* wfn = new ASQSubsetWeightFn(cx, cbeta, cgamma, calpha)
    cdef kmeans_asq_subset_t* solver = new kmeans_asq_subset_t(<size_t>mp, wfn[0])
    del wfn

    cdef unique_ptr[kmeans_result] km_result
    cdef unique_ptr[kmeans_result] km_result2
    cdef kmeans_result* res_ptr2
    km_result = solver.compute_with_binary_search(<size_t>(s - 1), lower_bound, upper_bound)
    if next_s:
        km_result2 = solver.compute_with_binary_search(<size_t>s, lower_bound, upper_bound)
        res_ptr2 = km_result2.get()
        next_s_cost[0] = res_ptr2.cost
    del solver

    cdef kmeans_result* res_ptr = km_result.get()
    cdef double cost = res_ptr.cost
    cdef vector[size_t]* pv = &res_ptr.path
    cdef Py_ssize_t npath = <Py_ssize_t>pv.size()

    cdef Py_ssize_t write_n = npath
    for i in range(write_n):
        result_ptr[i] = full_indices[pv[0][npath - 1 - i]]

    return cost


def vmix_approx(w_sorted, int s, int m, weights=None, double tol=1e-1, bint return_set=True, bint next_s=False, bint interpolation_search=False, bint vmix_estimate=True):
    """
    If return_set=False, the first element returned is the cost of s+1 (if next_s = True) or 0, rather than the set
    """
    cdef Py_ssize_t d = len(w_sorted)
    if s <= 2:
        raise ValueError("s must be at least 2")
    if s >= d:
        return np.unique(w_sorted), 0

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] w_arr = np.ascontiguousarray(w_sorted, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] weights_arr
    if weights is None:
        weights_arr = np.ones(d, dtype=np.float64)
    else:
        weights_arr = np.ascontiguousarray(weights, dtype=np.float64)

    cdef Py_ssize_t alloc_n = m + 10 if m + 10 > s else s
    cdef Py_ssize_t* result_ptr = <Py_ssize_t*>malloc(alloc_n * sizeof(Py_ssize_t))
    cdef double next_s_cost = 0
    cdef double cost = vmix_approx_c(&w_arr[0], &weights_arr[0], s, m, d, result_ptr, tol, next_s, &next_s_cost, interpolation_search, vmix_estimate)

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] Q = np.empty(s, dtype=np.float64)
    cdef Py_ssize_t i
    if return_set:
        Q[0] = w_arr[0]
        for i in range(1, s - 1):
            Q[i] = w_arr[result_ptr[i]]
        Q[s - 1] = w_arr[d - 1]
        free(result_ptr)

        return Q, cost
    else:
        free(result_ptr)
        return next_s_cost, cost


def interval_approx(w, int s, int m, weights=None, double tol=1e-3, bint return_set=True, bint next_s=False):
    """
    If return_set=False, the first element returned is the cost of s+1 (if next_s=True) or 0, rather than the set.
    """
    cdef Py_ssize_t d = len(w)
    if s <= 2:
        raise ValueError("s must be at least 2")
    if s >= d:
        return np.unique(w), 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] w_arr = np.ascontiguousarray(w, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] weights_arr
    if weights is None:
        weights_arr = np.ones(d, dtype=np.float64)
    else:
        weights_arr = np.ascontiguousarray(weights, dtype=np.float64)

    cdef double wmin = np.min(w_arr)
    cdef double wmax = np.max(w_arr)
    cdef double L = (wmax - wmin) / m
    # npath from solver can exceed s when binary search hits iter limit (k_found > k);
    # allocate m+10 to guarantee enough room for any path the solver may return
    cdef Py_ssize_t alloc_n = m + 10 if m + 10 > s else s
    cdef Py_ssize_t* result_ptr = <Py_ssize_t*>malloc(alloc_n * sizeof(Py_ssize_t))
    cdef double next_s_cost = 0.0
    cdef double cost = interval_approx_c(&w_arr[0], &weights_arr[0], s, wmin, wmax, m, d, result_ptr, tol, next_s, &next_s_cost)

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] Q = np.empty(s, dtype=np.float64)
    cdef Py_ssize_t i
    if return_set:
        for i in range(s - 1):
            Q[i] = wmin + L * result_ptr[i]
        Q[s - 1] = wmax
        free(result_ptr)
        return Q, cost
    else:
        free(result_ptr)
        return next_s_cost, cost


def interval_approx_sorted(w_sorted, int s, int m, weights=None, double tol=1e-3, bint return_set=True, bint next_s=False, bint use_vmix_approx=False):
    """
    Sorted-input specialization of interval_approx. w_sorted MUST be non-decreasing.

    Same return contract as interval_approx. Avoids np.min/np.max (uses endpoints)
    and dispatches to interval_approx_c_sorted, which builds prefix sums in a single
    streaming pass over w_sorted.

    use_vmix_approx: when True, uses vmix_approx to derive tighter lower/upper bounds
    for compute_with_binary_search (analogously to exact_adv with estimate_method='vmix_approx').
    """
    cdef Py_ssize_t d = len(w_sorted)
    if s <= 2:
        raise ValueError("s must be at least 2")
    if s >= d:
        return np.unique(w_sorted), 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] w_arr = np.ascontiguousarray(w_sorted, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] weights_arr
    if weights is None:
        weights_arr = np.ones(d, dtype=np.float64)
    else:
        weights_arr = np.ascontiguousarray(weights, dtype=np.float64)

    cdef double wmin = w_arr[0]
    cdef double wmax = w_arr[d - 1]
    cdef double L = (wmax - wmin) / m
    cdef Py_ssize_t alloc_n = m + 10 if m + 10 > s else s
    cdef Py_ssize_t* result_ptr = <Py_ssize_t*>malloc(alloc_n * sizeof(Py_ssize_t))
    cdef double next_s_cost = 0.0
    cdef double cost = interval_approx_c_sorted(&w_arr[0], &weights_arr[0], s, wmin, wmax, m, d, result_ptr, tol, next_s, &next_s_cost, use_vmix_approx)

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] Q = np.empty(s, dtype=np.float64)
    cdef Py_ssize_t i
    if return_set:
        for i in range(s - 1):
            Q[i] = wmin + L * result_ptr[i]
        Q[s - 1] = wmax
        free(result_ptr)
        return Q, cost
    else:
        free(result_ptr)
        return next_s_cost, cost
