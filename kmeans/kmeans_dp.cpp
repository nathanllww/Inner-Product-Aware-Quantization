#include "kmeans.h"
#include <memory>
#include <iostream>

/** Initialises shared DP state: prefix-sum structure, two DP row buffers, and a copy of the points. */
kmeans_dp::kmeans_dp(const std::vector<double> &points) : is(points), row(points.size()), row_prev(points.size()),
                                                          points(points.begin(), points.end()), n(points.size()) {}

/**
 * Recursively reconstructs cluster centers for a k-clustering of points by
 * divide-and-conquer on k:
 *
 *   1. Solve for k/2 clusters from the left  → get the last DP row.
 *   2. Solve for k-k/2 clusters from the right (on reversed/negated data)
 *      → get the last DP row.
 *   3. Find the split index i* that minimises
 *        left_cost(0..i) + right_cost(i+1..n-1).
 *   4. Recursively reconstruct centers for each half.
 *
 * The negation of reversed points in step 2 ensures that the reversed problem
 * produces the same pairwise distances and therefore the same DP costs.
 *
 * @param points   The (sorted) point subarray to cluster.
 * @param k        Number of clusters desired.
 * @param centers  Output: cluster centers are appended in sorted order.
 * @returns        Total L2 clustering cost.
 */
double kmeans_dp::report(std::vector<double> &points, size_t k, std::vector<double> &centers)
{
  //, std::vector<size_t> &path) {
  if (k == 1 || points.size() == 1) {
    if(k > points.size()){
      //std::cout << "we have a problem here " << k << " " << points.size() << std::endl;
      //assert(false);
    }
    
    interval_sum<double> is(points);// seems to be waste of time
    centers.push_back(is.mean(0, points.size() - 1));
    return is.query(0, points.size());//why are thse different
    }
    std::vector<double> reversed_points(points.rbegin(), points.rend());
    for (auto &p : reversed_points) p = -p;
    // also seems strange. We reversed the list. Pairwise distances does not change when multiplying all with -1. So it seems this is to ensure that a sorting does not reverse the reversion of the order....
    
    std::vector<double> last_row_left;
    std::vector<double> last_row_right;


    {
        std::unique_ptr<kmeans_dp> left(get_instance(points));
        std::unique_ptr<kmeans_result> res_left(left->compute(k / 2));
        last_row_left = std::move(left->row);
    }

    {
        std::unique_ptr<kmeans_dp> right(get_instance(reversed_points));
        std::unique_ptr<kmeans_result> res_right(right->compute(k - k/2));
        last_row_right = std::move(right->row);
    }
    double best = std::numeric_limits<double>::max();
    size_t best_idx = 0;
    for (size_t i = 0; i < points.size() - 1; ++i) {
        double cost_left = last_row_left[i]; // cost of clustering points[0..i] into k/2 clusters.
        double cost_right = last_row_right[points.size() - 2 - i]; // cost of clustering points[n-1 .. i+1] into k-k/2 clusters.

        double cost = cost_left + cost_right;
        if (cost < best) {
            best = cost;
            best_idx = i;
        }
    }

    {
        std::vector<double> empty;
        std::swap(empty, last_row_left);
    }
    {
        std::vector<double> empty;
        std::swap(empty, last_row_right);
    }
    {
        std::vector<double> empty;
        std::swap(empty, reversed_points);
    }
    std::vector<double> points_left(points.begin(), points.begin() + best_idx + 1);    
    std::vector<double> points_right(points.begin() + best_idx + 1, points.end());    
    size_t half = k/2;
    report(points_left, k/2, centers);
    size_t other_half = k - half;
    //path.push_back(best_idx);
    //size_t cs = path.size();
    report(points_right, other_half, centers);
    //for(size_t i=cs; i < path.size();++i){
    //path[i] += best_idx;
    //}
      
    return best;
}

/**
 * Computes both the optimal cost (via compute()) and the cluster centers
 * (via the divide-and-conquer report()).  Returns a result with both fields set.
 */
std::unique_ptr<kmeans_result> kmeans_dp::compute_and_report(size_t k) {
    std::vector<double> centers;
    std::vector<size_t> path;
    report(points, k, centers);
    std::unique_ptr<kmeans_result> res = compute(k);
    res->centers = std::move(centers);
    //res->path = std::move(path);
    //std::cout << "centers found " << res->centers.size() << std::endl;
    return res;
}
