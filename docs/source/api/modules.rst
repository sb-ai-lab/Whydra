API Reference
=============

Working with Data
-----------------

:py:class:`whydra.BenchmarkCaseRepository` loads tables and metadata from
a local benchmark directory. Its ``get_case_ground_truth()`` method
returns the path to the ground-truth graph file. The Whydra package does
not ship benchmark datasets. See :ref:`benchmark-cases` for the directory
layout.

.. toctree::
   :maxdepth: 1

   generated/whydra.data.BenchmarkCaseRepository

Algorithms
----------

.. toctree::
   :maxdepth: 1

   generated/whydra.algorithms.wrappers.run_standalone_algorithm

Algorithm Factory
~~~~~~~~~~~~~~~~~

.. py:module:: whydra

.. py:function:: create_standalone_algorithm(name, **kwargs)

   Creates a configured algorithm instance. Pass a class name or an alias
   from the table as ``name``. The factory forwards ``kwargs`` to the
   constructor and raises ``ValueError`` for unrecognized names. Call
   ``run(data)`` on the instance to obtain a graph.

.. py:function:: available_standalone_algorithms()

   Returns the canonical class names accepted by the factory. Install
   optional dependencies for the algorithms that require them.

.. list-table:: Algorithm Aliases
   :header-rows: 1
   :widths: 35 45 20

   * - Factory Alias
     - Class
     - Output
   * - ``pc_stable``
     - ``StandalonePCStable``
     - CPDAG
   * - ``pc_stable_parallel``
     - ``ParallelPCStable``
     - CPDAG
   * - ``fci_stable``
     - ``StandaloneFCIStable``
     - PAG
   * - ``fci_stable_cached``
     - ``StandaloneFCIStableCached``
     - PAG
   * - ``fci_stable_parallel_cached``
     - ``StandaloneFCIStableParallelCached``
     - PAG
   * - ``cfci_stable``
     - ``StandaloneCFCIStable``
     - PAG
   * - ``cfci_stable_parallel_cached``
     - ``StandaloneCFCIStableParallelCached``
     - PAG
   * - ``rfci_stable``
     - ``StandaloneRFCIStable``
     - PAG
   * - ``gfci_stable``
     - ``StandaloneGFCIStable``
     - PAG
   * - ``fci_plus_stable``
     - ``StandaloneFCIPlusStable``
     - PAG
   * - ``fci_plus_stable_parallel_cached``
     - ``StandaloneFCIPlusStableParallelCached``
     - PAG
   * - ``rai_stable``
     - ``StandaloneRAIStable``
     - CPDAG
   * - ``dag_gnn``
     - ``DagGNN``
     - Directed graph
   * - ``pcmci``
     - ``StandalonePCMCI``
     - Time-series graph
   * - ``pcmci_plus``
     - ``StandalonePCMCIPlus``
     - Time-series graph
   * - ``j_pcmci_plus``
     - ``StandaloneJPCMCIPlus``
     - Time-series graph

Search Results
~~~~~~~~~~~~~~

The tabular and time-series wrappers return a ``GeneralGraph``.
``get_nodes()`` returns the nodes, ``get_graph_edges()`` returns the edges,
and ``get_endpoint_at(node, other)`` returns the mark at ``node``.
The statistical wrappers set ``graph_kind`` to ``"cpdag"`` or ``"pag"``
for the corresponding outputs. The ``metadata`` dictionary can include
execution details, background knowledge, and statistical tests.

``GeneralGraph.graph[i, j]`` stores the mark at node ``j`` on its
edge with node ``i``. The native codes are ``0`` for no mark, ``-1``
for a tail, ``1`` for an arrowhead, and ``2`` for a circle. For
:math:`i \to j`, ``graph[i, j]`` is ``1`` and ``graph[j, i]`` is
``-1``. Use the appropriate encoding when exchanging matrices.

For time-series search, ``graph_mode`` controls the output graph. In
``summary`` mode, each node represents a variable. In ``lagged`` mode,
lags appear as separate nodes. The full tensors are available on
the model as ``lag_graph_``, ``p_matrix_``, and ``val_matrix_``.

Background Knowledge
~~~~~~~~~~~~~~~~~~~~

.. py:class:: BackgroundKnowledge

   Describes allowed, forbidden, and required links. Refer to a node by
   name or index. Pass the object as ``background_knowledge`` to an
   algorithm that supports it.

   - ``forbid(source, target)`` forbids the given direction.
   - ``require(source, target)`` requires the given direction.
   - ``require_adjacency(source, target)`` requires adjacency without specifying a direction.
   - ``add_tiers(tiers, allow_within=True)`` orders groups of nodes.

   The algorithm and rules determine when a rule takes effect. RAI does
   not support background knowledge. For example, with PC you can use
   ``BackgroundKnowledge().forbid("B", "A")``.

GES
~~~

.. py:function:: whydra.algorithms.score_based.ges_search(data, score_func, node_names, *, max_operator_set=3, max_parents=None, min_delta=0.0, verbose=False, verify_deltas=False, stats=None, bic_backend="auto", n_jobs=1, parallel_backend="auto")

   Returns the pair ``(CPDAG graph, total score)``. Use
   ``"local_score_BIC"`` for a Gaussian model or ``"local_score_BDeu"`` for
   a discrete model. ``max_operator_set`` caps the size of the sets used by
   insert and delete operators. ``None`` removes the cap.

   :doc:`CPU execution settings <../guides/cpu-performance>`.
