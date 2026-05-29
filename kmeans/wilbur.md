# How Wilber's Algorithm Works

This document explains the exact 1D k-means algorithm implemented in
`kmeans_wilber.cpp`.  The solver finds the globally-optimal partition of `n`
sorted points into `k` clusters minimising total sum-of-squared deviations
from each cluster mean.

## 1. The DP Recurrence

The foundation is a standard dynamic program.  Let `f[j]` be the minimum L2
cost of clustering the first `j` points into some number of clusters, and let
`cost(i, j)` be the L2 within-cluster variance of points `[i, j-1]`:

```
cost(i, j) = sum_{l=i}^{j-1} (x_l - mean(i,j))^2
```

This can be evaluated in O(1) using two prefix-sum arrays (see
`interval_sum.cpp`): `cost(i, j) = sum_sq(i,j) - sum(i,j)^2 / (j-i)`.

The Lagrangian-relaxed recurrence adds a fixed penalty `λ` per cluster:

```
f[j] = min_{i < j}  f[i] + cost(i, j) + λ
```

Define the *transition matrix* `G[i][j] = f[i] + cost(i, j) + λ`.  Solving
the DP means finding the row minimum of each column of G.

## 2. The Totally-Monotone Property

A matrix is *totally monotone* if its column minima are non-decreasing across
rows.  G satisfies this because the L2 cost function is *concave SMAWK*
(equivalently, satisfies the "inverse quadrangle inequality"): optimal split
points never decrease as the endpoint j increases.

This means if the best predecessor for column j is row `i*`, then for all
`j' > j` the best predecessor is `≥ i*`.

## 3. SMAWK: O(n) Column Minima of a Totally Monotone Matrix

The SMAWK algorithm (Aggarwal, Klawe, Moran, Shor, Wilber 1987) exploits the
totally-monotone property to find all n column minima in O(n) time instead of
O(n²).

### 3.1 Reduce Step

Given an n×m totally-monotone matrix (n rows, m columns), if m > n we can
eliminate at least m−n columns while preserving all column minima.

The reduction scans columns left-to-right, maintaining a "survivor stack":
- Push the current column.
- While the top two columns on the stack satisfy a dominance condition
  (the second-to-top row beats the top at the top's assigned column), pop.

After reduction, at most n columns remain.

### 3.2 Recursion

With m ≤ n columns remaining, recurse on even-indexed columns (stride doubled)
with roughly half the rows.  This gives T(n) = T(n/2) + O(n) = O(n).

### 3.3 Combine Step

For each odd column (between two solved even neighbours), the optimal row is
bounded by the solutions of its two neighbours.  A linear scan in this range
fills the odd entries in O(n) total.

### 3.4 Implementation

`kmeans_wilber` uses a single recursive function `smawk_inner(columns, stride, rows)`.
The public entry point `smawk(i0, i1, j0, j1, idxes)` converts index ranges to
vectors and delegates to `smawk_inner`.  A brute-force `smawk_naive` is kept as
a reference implementation.

## 4. Wilber's Online SMAWK Algorithm

The full DP has k layers (one per cluster count), each an n-column minimisation
problem.  Naively this is O(kn) calls to SMAWK = O(kn) total, giving O(kn)
overall — but Wilber's 1988 paper shows how to compute the *unconstrained*
Lagrangian DP (any number of clusters) in O(n log n) with a single SMAWK sweep.

### The key insight

Because the DP is self-referential — `f[i]` appears in the definition of
`G[i][j]` — we cannot apply SMAWK directly.  Instead, the algorithm maintains
two pointers `c` (right frontier) and `r` (left frontier), processing columns
in batches:

1. **Extend**: Set `p = min(2c − r + 1, n)` — roughly double the solved range.
2. **Fill f[c+1..p]**: Apply SMAWK to the submatrix `G[r..c, c+1..p]`.  These
   `f` values are correct because their best predecessors all lie in `[r..c]`
   (which are already solved).
3. **Challenge**: Apply SMAWK to `G[c+1..p-1, c+2..p]` to check whether any
   newly-solved row `> c` improves a column `> c`.  Find `j0`, the first such
   improvement.
4. **Advance**: If no improvement (`j0 = p+1`), advance `c = p`.  Otherwise
   update `f[j0]` from the challenger row and set `r = c+1`, `c = j0`.

Each iteration either doubles the solved range or advances `c` by at least 1,
giving O(log n) SMAWK calls of size O(n) each → O(n log n) total.

After the sweep, `bestleft[j]` holds the optimal split point for each `j`, and
tracing back gives the number of clusters produced by the current `λ`.

## 5. Lagrangian Relaxation: Finding k Clusters

Wilber's DP (Section 4) solves the *unconstrained* problem for a fixed `λ`:
it returns *some* number of clusters that minimises `Σ cost + λ · (# clusters)`.

To get *exactly* k clusters, we search for the `λ` that makes the DP return k
clusters.  The key properties are:

- As `λ → 0`, penalty is negligible → DP produces n clusters (each point alone).
- As `λ → ∞`, penalty dominates → DP produces 1 cluster (all points together).
- The optimal cluster count is non-increasing in `λ`.

### 5.1 Binary Search

Maintain `[lo, hi]` where `lo` produces too many clusters and `hi` too few.
Set `λ = (lo + hi) / 2` and bisect.  Terminates when `k_found == k`.

### 5.2 Interpolation Search (default)

Maintain two bracket points with associated (λ, cost, k_found) triples.
The Lagrangian cost as a function of k is piecewise linear with breakpoints
at each integer k value.  Interpolate the two known lines to estimate where
they intersect at k, setting the next `λ` accordingly.  Converges in ~20–50
iterations in practice (vs ~50–100 for binary search).

### 5.3 Degenerate Case: Noise Fallback

Occasionally two successive lambda values produce the same `k_found`, leaving
an empty interval.  This happens when two clustering solutions have identical
cost.  The fallback (`compute_interpolation_search_with_noise`) adds tiny
random perturbations (scaled by `sqrt(machine_eps)`) to break ties, retrying
up to 10 times with increasing noise range.

## 6. Cost Reconstruction

Once the right `λ` is found, `get_actual_cost` traces `bestleft[]` from `n`
back to `0`, accumulating true L2 cluster costs (without lambda penalties)
and computing each cluster center as the mean of its points.  Centers are
reversed into ascending order before returning.

## 7. Complexity Summary

| Step                        | Cost           |
|-----------------------------|----------------|
| Build prefix sums           | O(n)           |
| One Wilber DP pass          | O(n log n)     |
| Lambda search iterations    | O(log n)       |
| **Total**                   | **O(n log² n)**|

Space is O(n) for `f[]`, `bestleft[]`, and the prefix-sum arrays.

## 8. Comparison with Other Solvers in This Codebase

| Class                    | Algorithm                          | Time        |
|--------------------------|------------------------------------|-------------|
| `kmeans_wilber`          | Wilber + SMAWK + interpolation     | O(n log² n) |
| `kmeans_hirschberg_larmore` | HL deque DP + interpolation     | O(n log n)  |
| `kmeans_monotone`        | Divide-and-conquer DP              | O(kn log n) |
| `kmeans_linear`          | SMAWK per DP layer                 | O(kn)       |
| `kmeans_slow`            | Naive DP (reference)               | O(kn²)      |
| `kmeans_lloyd_fast`      | Lloyd's EM (approximate)           | O(nk · iters)|
