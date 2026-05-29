/**
 * kmeans_wilber_impl.hpp
 *
 * Template implementation of Wilber's SMAWK-based optimal DP clustering.
 * This header is designed to be included from within kmeans.h, after
 * kmeans_result, kmeans, and search_strategy have been declared.
 * Do not include it standalone.
 *
 * ---------------------------------------------------------------------
 * WeightFn concept
 * ---------------------------------------------------------------------
 * The template parameter W must be a struct/class providing:
 *
 *   std::vector<double> W::pts
 *       The underlying sorted point array.  Required so the generic
 *       interpolation-loop fallback can perturb points and retry.
 *
 *   W(const std::vector<double> &pts)
 *       Construct from a (possibly perturbed) point array.  Used by the
 *       noise-retry fallback to build a fresh solver on perturbed data.
 *
 *   double W::operator()(size_t i, size_t j) const
 *       Cost of assigning the contiguous range [i, j) (0-indexed, j
 *       exclusive) to a single cluster.  Must be totally monotone /
 *       Monge-concave so that the SMAWK column-minima property holds.
 *
 *   double W::total_cost() const
 *       Cost of putting all n points in one cluster, i.e. w(0, n).
 *       Used to initialize the upper bound for the lambda search.
 *
 *   double W::center(size_t i, size_t j) const
 *       Cluster representative for [i, j).  Return NaN if not applicable
 *       (the result's centers vector will then be left empty).
 *
 * All methods are called directly (non-virtual) so the compiler inlines
 * them; there is zero overhead compared to hardcoding the cost.
 *
 * ---------------------------------------------------------------------
 * Adding a new weight function
 * ---------------------------------------------------------------------
 * 1. Define a struct satisfying the WeightFn concept above.
 * 2. Optionally typedef / alias the instantiation:
 *       using MyKmeans = kmeans_wilber_t<MyWeightFn>;
 * 3. Provide a Python-callable Cython wrapper (see cython/kmeans1d.pyx
 *    for the L2 example).
 */

#pragma once

#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <memory>
#include <stdexcept>

// kmeans_result, kmeans, search_strategy are assumed to be already defined
// by the including translation unit (kmeans.h).

template <typename W>
class kmeans_wilber_t : public kmeans {
public:
    /**
     * Construct solver.
     * @param n    Number of elements to partition (indices 0..n-1).
     * @param wfn  Weight functor (moved into the object).
     */
    kmeans_wilber_t(size_t n, W wfn);

    std::unique_ptr<kmeans_result> compute(size_t k, double lower_bound, double upper_bound) override;
    std::unique_ptr<kmeans_result> compute_and_report(size_t k, double lower_bound, double upper_bound) override;
    std::string name() override { return "wilber"; }
    void set_search_strategy(search_strategy strat) { search_strat = strat; }

    /**
     * Public wrapper to binary search variant.
     * Solves for exactly k clusters using explicit binary search instead of
     * the (default) linear interpolation search for the Lagrange multiplier.
     * Simpler but slower than the default algorithm.
     */
    std::unique_ptr<kmeans_result> compute_with_binary_search(size_t k, double lower_bound, double upper_bound) override {
        return compute_binary_search(k, lower_bound, upper_bound);
    }

protected:
    /**
     * Called when compute_interpolation_search detects a loop (empty lambda
     * interval).  Default: fall back to binary search, which always terminates.
     *
     * Override this in a subclass for a faster strategy when the weight
     * function supports it (e.g. kmeans_wilber uses noise perturbation of the
     * underlying points, which resolves most numerical degeneracies without
     * abandoning interpolation search entirely).
     */
    virtual std::unique_ptr<kmeans_result>
    handle_interpolation_loop(size_t k, double lambda_fail, double lower_bound, double upper_bound);

    // --- State accessible to subclasses ---
    W      wfn;
    size_t n;
    double lambda;
    std::vector<double> f;
    std::vector<size_t> bestleft;
    search_strategy search_strat;

    // Protected so subclasses can call them (e.g. noise-fallback creates a
    // fresh instance and calls compute_interpolation_search on it).
    std::unique_ptr<kmeans_result> compute_binary_search(size_t k, double lower_bound, double upper_bound);
    std::pair<std::unique_ptr<kmeans_result>, bool>
        compute_interpolation_search(size_t k, bool use_fallback, double lower_bound, double upper_bound);
    double get_actual_cost(size_t n, std::unique_ptr<kmeans_result> &res);
    std::pair<double, size_t> wilber(size_t n);

private:
    // DP helpers — fully inlined by the compiler; no virtual dispatch.
    inline double weight(size_t i, size_t j);
    inline double g(size_t i, size_t j);

    std::vector<size_t> smawk_inner(std::vector<size_t> &columns, size_t e,
                                    std::vector<size_t> &rows);
    std::vector<double> smawk(size_t i0, size_t i1, size_t j0, size_t j1,
                              std::vector<size_t> &idxes);
    std::vector<double> smawk_naive(size_t i0, size_t i1, size_t j0, size_t j1,
                                    std::vector<size_t> &idxes);
};

// ============================================================
// Implementation
// ============================================================

template <typename W>
kmeans_wilber_t<W>::kmeans_wilber_t(size_t n, W wfn)
    : wfn(std::move(wfn)), n(n), lambda(0.0),
      f(n + 1, 0.0), bestleft(n + 1, 0),
      search_strat(search_strategy::INTERPOLATION) {}

// --- DP helpers -------------------------------------------------

template <typename W>
inline double kmeans_wilber_t<W>::weight(size_t i, size_t j) {
    if (i >= j) return std::numeric_limits<double>::max();
    return wfn(i, j) + lambda;
}

template <typename W>
inline double kmeans_wilber_t<W>::g(size_t i, size_t j) {
    return f[i] + weight(i, j);
}

// --- Interpolation-loop fallback --------------------------------

/**
 * When interpolation search detects an empty lambda interval (numerical
 * degeneracy), fall back to binary search, which always terminates.
 */
template <typename W>
std::unique_ptr<kmeans_result>
kmeans_wilber_t<W>::handle_interpolation_loop(size_t k, double lambda_fail, double lower_bound, double upper_bound) {
    std::cerr << "[kmeans_wilber: interpolation loop detected, falling back to binary search]" << std::endl;
    return compute_binary_search(k, lower_bound, upper_bound);
}

// --- Result reconstruction --------------------------------------

/**
 * Trace bestleft[], accumulate true cost (no lambda), fill centers.
 * Centers are pushed in reverse order during the trace and then reversed.
 * If wfn.center() returns NaN the centers vector is left empty.
 */
template <typename W>
double kmeans_wilber_t<W>::get_actual_cost(size_t n,
                                           std::unique_ptr<kmeans_result> &res) {
    double cost = 0.0;
    size_t m = n;
    res->path.push_back(m);
    while (m != 0) {
        size_t prev = bestleft[m];
        res->path.push_back(prev);
        cost += wfn(prev, m);
        double c = wfn.center(prev, m);
        if (!std::isnan(c)) res->centers.push_back(c);
        m = prev;
    }
    std::reverse(res->centers.begin(), res->centers.end());
    res->cost = cost;
    return cost;
}

// --- SMAWK totally-monotone matrix search -----------------------

/**
 * Core recursive SMAWK subroutine (see kmeans_wilber.cpp for full docs).
 * Identical logic to the original; g() resolves at compile time via W.
 */
template <typename W>
std::vector<size_t> kmeans_wilber_t<W>::smawk_inner(std::vector<size_t> &columns,
                                                     size_t e,
                                                     std::vector<size_t> &rows) {
    size_t nc = columns.size();
    size_t result_size = (nc + e - 1) / e;
    if (rows.size() == 1)
        return std::vector<size_t>(result_size, 0);

    // reduce
    std::vector<size_t> new_rows, translate;
    if (result_size < rows.size()) {
        for (size_t i = 0; i < rows.size(); ++i) {
            auto r = rows[i];
            while (new_rows.size() &&
                   g(r, columns[e * (new_rows.size() - 1)]) <=
                       g(new_rows.back(), columns[e * (new_rows.size() - 1)])) {
                new_rows.pop_back();
                translate.pop_back();
            }
            if (e * new_rows.size() < nc) {
                new_rows.push_back(r);
                translate.push_back(i);
            }
        }
    } else {
        new_rows = rows;
        for (size_t i = 0; i < rows.size(); ++i) translate.push_back(i);
    }

    if (result_size == 1)
        return std::vector<size_t>{translate[0]};

    // recurse
    std::vector<size_t> cm_rec = smawk_inner(columns, 2 * e, new_rows);
    std::vector<size_t> column_minima;

    // combine
    column_minima.push_back(translate[cm_rec[0]]);
    for (size_t i = 1; i < cm_rec.size(); ++i) {
        size_t from       = cm_rec[i - 1];
        size_t to         = cm_rec[i];
        size_t new_column = 2 * i - 1;

        column_minima.push_back(translate[from]);
        for (size_t r = from; r <= to; ++r) {
            if (g(new_rows[r], columns[new_column * e]) <=
                g(rows[column_minima[new_column]], columns[new_column * e]))
                column_minima[new_column] = translate[r];
        }
        column_minima.push_back(translate[to]);
    }

    if (column_minima.size() < result_size) {
        size_t from       = cm_rec.back();
        size_t new_column = column_minima.size();
        column_minima.push_back(translate[from]);
        for (size_t r = from; r < new_rows.size(); ++r) {
            if (g(new_rows[r], columns[new_column * e]) <=
                g(rows[column_minima[new_column]], columns[new_column * e]))
                column_minima[new_column] = translate[r];
        }
    }

    return column_minima;
}

template <typename W>
std::vector<double> kmeans_wilber_t<W>::smawk(size_t i0, size_t i1,
                                               size_t j0, size_t j1,
                                               std::vector<size_t> &idxes) {
    std::vector<size_t> rows, cols;
    for (size_t i = i0; i <= i1; ++i) rows.push_back(i);
    for (size_t j = j0; j <= j1; ++j) cols.push_back(j);

    std::vector<size_t> cm = smawk_inner(cols, 1, rows);
    std::vector<double> res(cm.size());
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = g(rows[cm[i]], cols[i]);
        idxes.push_back(rows[cm[i]]);
        assert(res[i] != std::numeric_limits<double>::max());
    }
    return res;
}

template <typename W>
std::vector<double> kmeans_wilber_t<W>::smawk_naive(size_t i0, size_t i1,
                                                     size_t j0, size_t j1,
                                                     std::vector<size_t> &idxes) {
    std::vector<double> cm(j1 - j0 + 1, std::numeric_limits<double>::max());
    idxes.resize(j1 - j0 + 1, n + 10);
    for (size_t j = j0; j <= j1; ++j) {
        for (size_t i = i0; i <= i1; ++i) {
            if (i >= j) continue;
            double val = g(i, j);
            if (val < cm[j - j0]) { cm[j - j0] = val; idxes[j - j0] = i; }
        }
    }
    return cm;
}

// --- Core Wilber DP pass ----------------------------------------

/**
 * One pass for the current lambda.  Fills f[] and bestleft[].
 * Returns (f[n], number_of_clusters).
 */
template <typename W>
std::pair<double, size_t> kmeans_wilber_t<W>::wilber(size_t n) {
    f.resize(n + 1, 0);
    bestleft.resize(n + 1, 0);
    f[0] = 0;
    size_t c = 0, r = 0;

    while (c < n) {
        size_t p = std::min(2 * c - r + 1, n);

        {
            std::vector<size_t> bl;
            auto cm = smawk(r, c, c + 1, p, bl);
            for (size_t j = c + 1; j <= p; ++j) {
                f[j]        = cm[j - (c + 1)];
                bestleft[j] = bl[j - (c + 1)];
            }
        }

        if (c + 1 <= p - 1) {
            std::vector<size_t> bl;
            auto H = smawk(c + 1, p - 1, c + 2, p, bl);

            size_t j0 = p + 1;
            for (size_t j = p; j >= c + 2; --j)
                if (H[j - (c + 2)] < f[j]) j0 = j;

            if (j0 == p + 1) {
                c = p;
            } else {
                f[j0]        = H[j0 - (c + 2)];
                bestleft[j0] = bl[j0 - (c + 2)];
                r = c + 1;
                c = j0;
            }
        } else {
            c = p;
        }
    }

    size_t m = n, length = 0;
    while (m > 0) { m = bestleft[m]; ++length; }
    return {f[n], length};
}

// --- Lambda search strategies -----------------------------------

template <typename W>
std::unique_ptr<kmeans_result>
kmeans_wilber_t<W>::compute_binary_search(size_t k, double lower_bound, double upper_bound) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);
    double lo = lower_bound, hi = upper_bound;
    if(lo < 0.0){
        lo = 0.0;
    }
    if(hi < 0.0){
        hi = wfn.total_cost();
    }

    bool using_restricted_bounds = (lo != 0.0 || hi != wfn.total_cost());

    double val_found; size_t k_found;
    bool diag = (getenv("KMW_DIAG") != nullptr);
    if (diag) {
        std::cerr << std::setprecision(17)
                  << "[bsearch_start] k=" << k << " n=" << n
                  << " lo=" << lo << " hi=" << hi
                  << " total_cost=" << wfn.total_cost() << std::endl;
        // Check k_found at lo and hi before the loop
        double saved_lambda = lambda;
        double vlo, vhi; size_t klo, khi;
        lambda = lo; std::tie(vlo, klo) = wilber(n);
        lambda = hi; std::tie(vhi, khi) = wilber(n);
        lambda = saved_lambda;
        std::cerr << "[bsearch_start] k_found@lo=" << klo << " k_found@hi=" << khi << std::endl;
    }
    for (size_t cnt = 0; ; ++cnt) {
        lambda = lo + (hi - lo) / 2;
        std::tie(val_found, k_found) = wilber(n);
        if (diag) {
            std::cerr << std::setprecision(17)
                      << "[bsearch iter " << cnt << "] lambda=" << lambda
                      << " k_found=" << k_found << " lo=" << lo << " hi=" << hi
                      << " hi-lo=" << (hi - lo) << std::endl;
        }
        if      (k_found == k) break;
        else if (k_found  < k) hi = lambda;
        else                   lo = lambda;
        if (cnt > 60 && using_restricted_bounds) {
            std::cerr << "[kmeans_wilber binary search: >60 iters with restricted bounds, retrying with full range]" << std::endl;
            return compute_binary_search(k, 0.0, wfn.total_cost());
        }
        if (cnt > 200) {
            std::cerr << "[kmeans_wilber binary search: >200 iters]" << std::endl;
            break;
        }
    }
    get_actual_cost(n, res);
    return res;
}

/**
 * Finds lambda via linear interpolation search.
 * When a loop is detected (empty lambda interval), delegates to
 * handle_interpolation_loop() if use_fallback is true, otherwise
 * returns (empty_result, false) so the caller can decide.
 */
template <typename W>
std::pair<std::unique_ptr<kmeans_result>, bool>
kmeans_wilber_t<W>::compute_interpolation_search(size_t k, bool use_fallback, double lower_bound, double upper_bound) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);

    double lo_intercept = 0;
    size_t lo_k = n;
    double hi_intercept = upper_bound;
    size_t hi_k = 1;

    double val_found; size_t k_found;
    lambda = upper_bound;
    std::tie(val_found, hi_k) = wilber(n);
    hi_intercept = val_found - (double)hi_k * lambda;

    for (size_t cnt = 0; ; ++cnt) {
        lambda = (hi_intercept - lo_intercept) / (double)(lo_k - hi_k);
        std::tie(val_found, k_found) = wilber(n);

        if (k_found <= hi_k || k_found >= lo_k) {
            std::cerr << "[Warning: K Found Outside search range - Empty Lambda "
                         "Interval or numerical issues]" << std::endl;
            std::cerr << std::setprecision(20)
                      << "stats [k_found, k-range searched] " << k_found
                      << " - (" << hi_k << " ,  " << lo_k << " ) - lambda "
                      << lambda << std::endl;
            if (use_fallback) {
                res = handle_interpolation_loop(k, lambda, lower_bound, upper_bound);
                return {std::move(res), true};
            }
            return {std::move(res), false};
        }

        if      (k_found > k) { lo_k = k_found; lo_intercept = val_found - (double)lo_k * lambda; }
        else if (k_found < k) { hi_k = k_found; hi_intercept = val_found - (double)hi_k * lambda; }
        else                  { break; }

        if (cnt > 1000) {
            std::cout << "[Warning: More than 1000 steps - breaking]" << std::endl;
            assert(false);
        }
    }

    get_actual_cost(n, res);
    return {std::move(res), true};
}

// --- Public entry point -----------------------------------------

template <typename W>
std::unique_ptr<kmeans_result> kmeans_wilber_t<W>::compute(size_t k, double lower_bound, double upper_bound) {
    std::unique_ptr<kmeans_result> res(new kmeans_result);

    if (k >= n) {
        // Each element in its own cluster.
        double cost = 0.0;
        res->path.push_back(n);
        for (size_t i = n; i > 0; --i) {
            cost += wfn(i - 1, i);
            double c = wfn.center(i - 1, i);
            if (!std::isnan(c)) res->centers.push_back(c);
            res->path.push_back(i - 1);
        }
        // centers were pushed in reverse (n-1, n-2, ..., 0); reverse to ascending.
        std::reverse(res->centers.begin(), res->centers.end());
        res->cost = cost;
        return res;
    }
    if (k == 1) {
        res->path = {n, 0};
        res->cost = wfn.total_cost();
        double c = wfn.center(0, n);
        if (!std::isnan(c)) res->centers.push_back(c);
        return res;
    }

    if(lower_bound < 0.0){
        lower_bound = 0.0;
    }
    if(upper_bound < 0.0){
        upper_bound = wfn.total_cost();
    }

    bool succ;
    switch (search_strat) {
    case search_strategy::BINARY:
        return compute_binary_search(k, lower_bound, upper_bound);
    case search_strategy::INTERPOLATION:
        std::tie(res, succ) = compute_interpolation_search(k, true, lower_bound, upper_bound);
        return res;
    default:
        throw std::runtime_error("kmeans_wilber_t: unknown search strategy");
    }
}

template <typename W>
std::unique_ptr<kmeans_result> kmeans_wilber_t<W>::compute_and_report(size_t k, double lower_bound, double upper_bound) {
    return compute(k, lower_bound, upper_bound);
}
