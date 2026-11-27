.. _api-behavior:

API and statistical behavior
============================

This guide covers graph interfaces, run parameters, and statistical
limits in the algorithms.

Graph representation and adapters
---------------------------------

In ``graph.graph``, the entry ``m[i, j]`` stores the endpoint mark at
the **column node**. The values ``0``, ``-1``, ``1``, and ``2`` denote
no mark, a tail, an arrowhead, and a circle, respectively. With A and B
in that order, the edge :math:`A \to B` has the following matrix.

.. code-block:: text

   [[ 0, 1],
    [-1, 0]]

Both ``get_endpoint(A, B)`` and ``get_endpoint_at(A, B)`` return the mark
at A. The ``graph.graph_kind`` attribute stores the graph type as
``dag``, ``cpdag``, or ``pag``. A PAG retains its type even if it has no circles.

Use ``to_causallearn`` and ``from_causallearn`` from
``whydra.evaluation.graph_utils`` to exchange graphs with causal-learn.
These adapters transpose the matrix to account for the different
endpoint conventions. Install ``.[comparison]`` to use them.
For NetworkX, use ``convert_general_graph_to_nx`` and ``from_networkx``
from the same module. Preserve the ``endpoint_u`` and ``endpoint_v``
attributes if you plan to convert the graph back to Whydra.

.. list-table::
   :header-rows: 1

   * - Operation
     - Interchange convention
   * - ``graph``
     - The adapter transposes the matrix when converting to causal-learn.
   * - ``get_endpoint``
     - Whydra reads the mark at the first argument. Causal-learn reads
       the mark at the second.
   * - ``get_nodes``, ``get_graph_edges``, ``get_edge``, ``add_edge``
     - Whydra uses its own node and edge classes.
   * - Causal-learn graph methods
     - Convert the graph with ``to_causallearn`` before calling methods
       that Whydra does not provide.
   * - Conversion to NetworkX
     - The ``endpoint_u`` and ``endpoint_v`` attributes preserve all nine
       combinations of tails, arrowheads, and circles.

The wrappers check required graph properties, including the absence of
directed cycles, arrowheads at ancestors, and arrowheads at nodes
incident to undirected edges. The existence of a compatible MAG and
the realizability of a PAG require separate checks.

``whydra.evaluation.metrics.calculate_metrics`` matches nodes by name.
Pass ``node_mapping`` when the graphs use different names. Specify the
type for reference PAGs and CPDAGs. A binary adjacency matrix for a
reference DAG must encode an acyclic graph. Set ``pag_proxy=True`` to
compute approximate metrics when comparing a PAG with a CPDAG. The
function records
``approximate=True`` and ``metric_mode="pag_to_cpdag_proxy"``.
Use a reference PAG or MAG for experiments with latent variables.
If a denominator is zero, the metric reports zero for compatibility.
The ``*_f1_defined`` flags and support counts indicate whether the
score can be interpreted.

Parameters and data
-------------------

Create an algorithm with
``whydra.create_standalone_algorithm(name, **options)`` and call
``algorithm.run(data)``. The function
``whydra.available_standalone_algorithms()`` returns the names of the
available classes. For a one-off run on an array or file,
use ``whydra.run_standalone_algorithm``. It passes algorithm options
to the constructor.

:doc:`Algorithm names and aliases <../api/modules>`.

For the PC, FCI, and RAI wrappers, ``run()`` arguments override
constructor settings, which in turn override the defaults. The method
validates the arguments before starting the search. Passing
``depth`` raises an error on wrappers that do not support it. GFCI
has its own ``depth`` parameter. The wrappers support the combination
``uc_rule=0`` and ``uc_priority=2``. The CPU default is ``n_jobs=1``.
Negative values follow joblib conventions. Zero is invalid.
An unstable search requires a single worker. The progress indicator
defaults to off (``show_progress=False``).

Use ``max_cond_set_size`` with PC, the FCI family, and RAI.
``None`` removes the limit, zero permits unconditional tests only, and a
nonnegative integer sets the maximum number of conditioning variables.
Passing ``None`` to ``run()`` also clears a limit set in the constructor.
The wrappers reject booleans, strings, and non-integer numbers.

The limit applies to every conditional independence testing stage,
including Possible-D-SEP, conservative triple classification, FCI+
augmentation, tests during RFCI orientation, and RAI refinement.
A skipped test produces neither a p-value nor a separating set.
If the limit prevents a conservative triple classification or an RFCI
path test from completing, the orientation remains unresolved.
``depth_limited=True`` indicates that a limit is set, even if the search
never needed a larger conditioning set. Read the values from
``graph.metadata["max_cond_set_size"]`` and
``graph.metadata["effective_max_cond_set_size"]``.

GFCI uses ``depth`` as a separate limit on its conditional independence
search. With ``depth=-1``, only the global ``max_cond_set_size`` limit
applies, if set. When both limits
are finite, the smaller one applies. GES parameters control operator
enumeration in the scoring stage. In ``StandaloneFCIStable``,
``m_max`` limits only Possible-D-SEP, and ``use_pds_path=True`` further
narrows the candidates at that stage. The default is
``use_pds_path=False``. Low-level functions do not attach metadata
to the endpoint matrix itself. Where supported, they record the limit
in ``CausalGraph.G.metadata``.

.. code-block:: python

   import numpy as np
   from whydra import create_standalone_algorithm

   rng = np.random.default_rng(42)
   x = rng.normal(size=400)
   data = np.column_stack(
       (x, x + rng.normal(size=400), x + rng.normal(size=400))
   )
   pc = create_standalone_algorithm("pc_stable", max_cond_set_size=1)
   capped = pc.run(data)
   full = pc.run(data, max_cond_set_size=None)
   assert capped.metadata["depth_limited"]
   assert full.metadata["effective_max_cond_set_size"] is None

Each row in a tabular array represents an observation, and each column
represents a variable. A ``DataFrame`` preserves its column names.
For arrays, pass ``node_names``. Names must be unique. ``fisherz``
requires finite numerical values and nonzero variance.
``strict_degenerate=True`` rejects constant columns during graph
preparation. The test requires positive degrees of freedom. A singular
or ill-conditioned submatrix raises an error if the ratio of its smallest
eigenvalue to its largest eigenvalue is at most ``1e-12``. By default, a
constant column first triggers a warning. Data preparation rejects
missing and infinite values. ``chisq`` treats values as category labels.
If a column has many distinct values and you intend to treat them as
categories, pass ``data_type="categorical"``. Labels can be strings or
fractional numbers.

For an ambiguous CSV layout, specify ``sep`` or ``delimiter``, along with
``header``. The loader accepts numeric tables separated by commas,
semicolons, whitespace, or tabs. Loading ``.pkl`` or ``.pickle`` requires
``trusted_pickle=True``. Unpickling can execute code. Use CSV or NPY
for external data.

GFCI chooses between ``local_score_BIC`` and ``local_score_BDeu``.
``auto`` selects the score based on the conditional independence test.
An unrecognized name raises an error. BDeu assigns consecutive indices
to categories. The equivalent sample size is 1. The number of parent
configurations is the product of the number of categories for each
parent variable. This includes configurations absent from the sample.
The function raises an error if computing the prior parameter underflows.

Set ``max_operator_set=None`` to enumerate all GES candidates. The
algorithm warns if it truncates candidates. Read ``approximate`` and
``truncated_candidate_pools`` in the metadata to see whether truncation
occurred in the stages the search visited.

Statistical behavior and algorithm limits
-----------------------------------------

``fci_stable`` computes Possible-D-SEP per edge and removes an edge as
soon as it finds a separating set. Cached FCI variants use kernels that
compute Possible-D-SEP per node. When comparing results, record the exact
algorithm name, ``alpha``, the CI test, and the search limits. Both
implementations assign circle endpoints to the skeleton and apply R0
before Possible-D-SEP. FCI v2 applies R1 and R2 together. Its parallel
search checks the context before applying a result. R5 orients existing
uncovered paths with circle endpoints and preserves the skeleton.

Preliminary orientation uses known separating sets. The graph can
distinguish an unknown separating set from a known empty set.
PC-stable uses the union of variables in the separating sets it finds
when orienting colliders. FCI stores the separating sets separately
from the collider orientation summaries. The FCI+ adapter supports a
flat separating-set representation. CFCI treats a triple as ambiguous
when conditional independence test results are unavailable and retains
the Possible-D-SEP test results.

GFCI applies a modified R0 rule (C'/F') to each triple, following
Ogarrio, Spirtes, and Ramsey (2016). It stores the pattern from the
scoring stage separately and uses conditional independence tests to find
missing separating sets. The results guide collider orientation, and
the algorithm retains the separating sets it finds. When comparing against
an external implementation, record the score formula and search parameters.
Whydra's Gaussian score and Tetrad's SEM-BIC use different formulas.

RAI is experimental. The default is ``certify_skeleton=False``.
With ``True``, RAI refines the skeleton with a stable adjacency search
and rebuilds the CPDAG. This requires additional computation and can
change the statistical result.

RAI follows the recursive search scheme and checks whether results
computed in parallel are still applicable. RAI raises
``NotImplementedError`` if you pass unsupported background knowledge.

`RAI paper <https://jmlr.org/papers/v10/yehezkel09a.html>`_ ·
`GFCI paper <https://proceedings.mlr.press/v52/ogarrio16.html>`_

Background knowledge
--------------------

``require(a, b)`` sets the direction if the edge exists when the constraint
is applied. ``require_adjacency(a, b)`` protects the edge during the
statistical search. Specify both rules if both requirements apply.
``add_tiers({a: 0, b: 1})`` disallows directions from a later tier to an
earlier tier. Pass ``allow_within=False`` to disallow directions within
a tier as well.

A direction constraint marked ``PRE_SEARCH`` applies before the search.
For a PAG, a prohibition set for ``PRE_SEARCH`` must also apply at
the relevant orientation stage. Whydra rolls back an invalid orientation
and records the conflict in the report, even at a stage with no active
constraints. ``StandaloneRAIStable`` does not support background
knowledge.

External datasets and package contents
--------------------------------------

``BenchmarkCaseRepository(cases_root=...)`` accepts a path to a directory
of external datasets. By default, it looks for
``whydra_benchmarks/cases`` under the current working directory. See
:ref:`benchmark-cases` for the directory layout and the loader.

PNG export requires ``.[viz]`` and the Graphviz
executable. ``.[comparison]`` includes the causal-learn adapter.

:doc:`Third-party code and licenses <../CODE_PROVENANCE>`.

:doc:`CPU caches, memory, and profiling <cpu-performance>`.
