.. _cpu-performance:

CPU performance
===============

CPU execution uses Whydra's core scientific dependencies.
Keep the algorithm, parameters, and data fixed when measuring performance.

Conditional independence tests
------------------------------

The built-in Fisher-Z test uses simplified formulas for conditioning sets
of size zero or one when the correlation submatrix passes its numerical
checks. Other cases use the general path, which checks the eigenvalues
and inverts the matrix. Near the ``alpha`` threshold, wrappers may use
the reference calculation. The logical request order and stopping
decisions stay the same. Intermediate p-values can differ at the bit level.

``CIT.batch`` accepts requests as ``(x, y, conditioning_set)`` triples
and returns p-values in request order. Request count and numerical buffer
size limit each batch. Symmetric requests share one cache entry. The
first request determines the direction of that entry. The cache preserves
LRU order and handles duplicates in request order. If a request fails,
the cache retains completed results and releases reservations for
unfinished requests. Identical concurrent scalar requests share the
same in-flight computation. Custom conditional independence tests,
unsupported numerical cases, a small cache, or passes that protect
required edges may use the scalar path.

Stable skeleton search in PC and FCI groups unconditional queries by
neighbors, and conditional queries by pairs at the same depth. A serial
pass computes distinct cache misses together and applies hits, misses,
and duplicates in the original order. A batch holds the cache lock
while it runs. Parallel workers perform numerical operations without
that shared lock. As needed, the implementation also batches eigenvalue
checks and inversion for first-order queries.

When comparing results, account for each stage's comparison with
``alpha``. A stage uses either ``> alpha`` or ``>= alpha``.
Grouping requests preserves the union of stable separating sets and
the protections for edges specified by background knowledge.

GES
---

Each pass builds immutable integer bit masks for parents, neighbors,
adjacencies, and semi-directed transitions. It uses these masks to check
cliques and paths and to merge ordered candidate lists. After a graph
change, a pass can reuse candidates for a node when all their local
dependencies are unchanged. It checks path validity against the current graph.
Insert and Delete modify the contents of arrays. References to the same
arrays alone are insufficient to reuse candidates.

GES caches local scores by integer family keys, which avoids reordering
the parent set for each lookup. Candidate enumeration can be exhaustive
or capped. GES breaks ties between equal score changes in a fixed order.
For debugging, ``verify_deltas=True`` recomputes the graph score after
each move to check the score change.

To recover a CPDAG, GES constructs a consistent DAG extension and
applies Meek rules R1-R4 using bit masks. After a move, recovery covers the
whole connected region around the changed endpoints, including both
sides of a split component. When GES reuses candidate lists during
exhaustive search, it restores their statistics in aggregate. During
capped search, it retains warnings for truncated lists.

``ges_search`` defaults to ``bic_backend="auto"``. For the built-in
Gaussian BIC, it selects ``gram`` when there are at least five
variables and 200 observations. ``gram`` batches calculations on
centered sufficient statistics and uses the reference scorer to
resolve close candidates. Smaller problems, BDeu, and custom scorers
use ``reference``. Set ``bic_backend="reference"`` to choose the
scalar path explicitly. To run Gram batches in parallel, set
``n_jobs`` and select ``parallel_backend="threads"`` or
``"processes"``. Automatic Gram execution remains serial.

Set ``bic_backend="covariance"`` to use an alternative Gaussian BIC
path with a bounded cache of Cholesky factors. For ill-conditioned
inputs, precision loss, large shifts, or too few rows, this path
falls back to the reference score computed with ``lstsq``.
Numerical differences can change the order of nearly tied moves.
GES recomputes the final score of the selected graph with the reference
formula. BDeu does not support ``covariance``. Read the selected backend
from ``graph.metadata["bic_backend"]``.

Set ``max_operator_set=None`` for exhaustive enumeration. Check
``approximate`` and ``truncated_candidate_pools`` in the metadata to
see whether it truncated any candidate lists. The counters
``ges_forward_rebuilt_targets``, ``ges_forward_reused_targets``,
``ges_forward_score_reuses``, and ``ges_backward_score_reuses`` describe
reuse of candidates and scores. ``LocalScoreCache`` counts hits and
misses only for lookups that reach the cache.

Dispatch and execution diagnostics
----------------------------------

``n_jobs`` specifies the requested number of workers. A stage can stay
serial if its work items are too small to benefit from parallelism.
Dispatch considers the candidate count, conditioning-set size, expected
cache misses, and process startup costs. It starts pools as needed.
FCI v2 estimates the candidate count before speculatively testing
multiple edges in Possible-D-SEP. Its internal threshold for the
built-in Fisher-Z test is 8192 relative work units. Smaller workloads
run serially. For stages that measure elapsed time, ``auto`` may reuse
an existing pool or select a smaller one.

Read ``graph.metadata["execution"]["passes"]`` for the phase, depth,
requested worker count, work estimate, and selected execution mode.
The fields available depend on the algorithm and stage.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``effective_n_jobs`` in graph metadata
     - The normalized worker setting for the run.
   * - ``estimated_kernel_work`` in a stage record
     - The relative cost estimate used to select an execution mode.
   * - ``execution["passes"]``
     - Dispatch decisions for individual stages and depths.
   * - ``ci_stats``, when the component reports statistics
     - Counts for queries, cache activity, batches, or speculative computations.

CPU kernels may offer ``auto``, ``threads``, and ``processes`` for
individual stages. Pass options to the stage or function that declares
them. For example, ``ges_search`` has ``parallel_backend`` for score
batches. When calling ``pdsep_v2`` directly, ``parallel_min_tests`` sets
a threshold in relative work units. A value of zero keeps dispatch
enabled for that direct call.

Serial and parallel execution of the same algorithm preserve search
limits and separating-set rules. Cached FCI and FCI v2 have different
aliases and use different search kernels. Keep the algorithm name fixed
when measuring the effect of ``n_jobs``.

FCI v2 can evaluate several edges using a graph snapshot. It then
applies only results whose contexts are still valid, in order.
The PDS statistics ``selected_ci_requests``, ``live_ci_requests``,
``speculative_ci_requests``, and ``total_ci_requests`` count requests,
including cache hits. ``invalidated_speculative_edges`` identifies edges
whose contexts required discarding their speculative results.

``live_evaluated_edges`` records the edges checked serially.
``estimated_kernel_work`` is expressed in relative work units.
Worker counts on a stage record describe the dispatch decision.
Measure elapsed time with a timer or use the actual duration field
when the stage returns one.

FCI v2 orientation groups edges by their marks in a fixed order. It
reuses adjacency indexes during the R0-R10 pass and reads the current
marks immediately before applying each rule. Applying background
knowledge refreshes the indexes.

Measure a run with ``n_jobs=1``, then repeat it with the desired worker
count. Compare the graphs and timings on the same data with the same
node names, CI test, ``alpha``, background knowledge, and search limits.
On Windows, when running a script with processes, place the call under
``if __name__ == "__main__"``.

.. code-block:: python

   from time import perf_counter

   import numpy as np
   from whydra import create_standalone_algorithm

   def main():
       rng = np.random.default_rng(42)
       data = rng.normal(size=(1200, 12))
       data[:, 1] += data[:, 0]
       data[:, 2] += data[:, 0]

       results = []
       for jobs in (1, 4):
           algorithm = create_standalone_algorithm(
               "pc_stable_parallel", indep_test="fisherz", n_jobs=jobs
           )
           started = perf_counter()
           graph = algorithm.run(data)
           print(jobs, round(perf_counter() - started, 3))
           print(graph.metadata.get("execution", {}).get("passes", []))
           results.append(graph.graph.copy())

       assert np.array_equal(*results)

   if __name__ == "__main__":
       main()

The example shows which execution mode the algorithm selected.
To measure speedup, use a sample representative of your task and run it
several times on the same computer. Record the time for the run that
creates the worker pool separately from the times for runs that reuse it.

Caches for test results and scores last for one run. Memory for the dense
endpoint-mark matrix grows quadratically with the number of vertices
:math:`p` (:math:`O(p^2)`).

The core dependencies support CPU execution. When the native Fisher-Z
extension is available, supported stages can use it for batches.
``WHYDRA_BUILD_NATIVE=0`` disables the native extension build.
``WHYDRA_PROFILE=1`` enables the built-in instrumentation.
For line-by-line profiling, install ``line_profiler`` and run with
``kernprof``. Use the same profiling and build settings when comparing
ordinary timings.

CI query accounting in RFCI, CFCI, and FCI+
-------------------------------------------

RFCI, CFCI, and FCI+ use query-aware CI adapters to batch operations for
the built-in Fisher-Z test. They preserve the stage's external cache,
the CIT cache, the counters, and the test order. If a request fails,
the adapters retain completed results that precede it in the test order.
The serial adapter handles up to 1024 requests
at a time. Custom tests, overridden graph methods, shared-memory RFCI
caches that require eviction, and requests already in progress may
use the scalar path.

CFCI groups candidates for conservative triple classification. FCI+
groups augmentation checks by pair, preserving order and the maximum
conditioning-set size. RFCI batches checks against explicit lists of
conditioning sets. When D-SEP search stops early, it respects the
order in which the first separating set appears.

The PAG orientation rules for CFCI and FCI+, and the R1-R3 and R5-R10
passes for RFCI, use ordered edge snapshots and adjacency indexes.
They verify applicability against the current marks. The shared rules
rebuild the indexes after applying background knowledge. RFCI can
remove an edge using its own rules when checking a discriminating path.

Speculative Possible-D-SEP computations in CFCI
-----------------------------------------------

CFCI refinement computes reachability on the preliminarily oriented
skeleton and then processes eligible pairs. With the built-in Fisher-Z
test, it can evaluate some requests for a pair in advance. The ordered
adapter starts with a short block of up to 32 candidates and grows it
as needed up to ``256 * effective_n_jobs(n_jobs)`` with the standard
settings. With a custom conditional independence test, it requests
results individually.

The search applies results in order and stops at the first separating
set. An error in a speculative request becomes a search error only if
the search reaches that request. All paths respect the
conditioning-set size limit.

Use ``pdsep_ci_tests``, ``pdsep_pairs_scanned``, and
``pdsep_pairs_with_diffset``, along with the worker request counters in
``ci_stats``, for diagnostics. ``pdsep_worker_speculative_tests`` counts
speculative worker requests whose results the search did not use. For
parallel cached CFCI, inspect the selected mode and the Possible-D-SEP
computation statistics.
