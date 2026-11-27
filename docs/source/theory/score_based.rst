.. _score-based:

============================
Score-based causal discovery
============================

Score-based methods use a score to compare causal structures. The search
chooses the best-scoring candidate it considers. Depending on the method,
candidates are individual graphs or equivalence classes of graphs.

Common criteria include AIC, BIC, MDL, and regularized log-likelihood.
The definition of the score determines whether the search minimizes or
maximizes it.

The search can use enumeration, greedy steps, or continuous optimization.

.. note::

   Theoretical guarantees depend on the model assumptions, the score, and
   the limits of the search. Methods can return different types of graphs.

Continuous score-based methods
------------------------------

Continuous methods represent edges with real-valued weights and impose
an acyclicity constraint during optimization. Neural methods combine this
approach with learning a model of the data. The architecture and loss function
determine which nonlinear dependencies the model can represent.

.. _tab-nn-score-based:

.. list-table:: Examples of continuous search methods from the literature
   :header-rows: 1
   :widths: 28 72

   * - Publication
     - Description
   * - `NOTEARS (Zheng et al., 2018) <https://proceedings.neurips.cc/paper/2018/hash/e347c51419ffb23ca3fd5050202f9c3d-Abstract.html>`_
     - Formulates DAG search as a continuous optimization problem with a
       differentiable acyclicity constraint.
   * - `NO-BEARS (Lee et al., 2019) <https://arxiv.org/abs/1911.00081>`_
     - Uses a spectral approximation to the acyclicity constraint and
       polynomial regression for nonlinear dependencies.

Example of a classical algorithm
--------------------------------

.. _alg-ges:

Greedy Equivalence Search (GES)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GES searches greedily over Markov equivalence classes of DAGs. In Whydra,
the search starts with an empty graph. The first phase inserts edges while
it can find valid insertions that improve the score. The second phase
deletes edges while a deletion improves the score.
`Chickering (2002) <https://jmlr.org/papers/v3/chickering02b.html>`_

The function ``whydra.algorithms.score_based.ges_search`` returns
``(graph, score)``. The ``graph`` is a CPDAG stored in a ``GeneralGraph``,
and ``score`` is the final score. Use ``local_score_BIC`` for continuous
data and ``local_score_BDeu`` for discrete data.

.. note::

   By default, ``max_operator_set=3`` limits the size of the auxiliary sets
   in insertion and deletion operators. Set ``max_operator_set=None`` to
   remove this limit. The cost of enumeration depends on graph density.

.. _alg-dag-gnn:

DAG-GNN
-------

DAG-GNN uses a variational autoencoder with a graph neural network to model
nonlinear dependencies. The training loss includes an acyclicity constraint.
`Yu et al. (2019) <https://proceedings.mlr.press/v97/yu19a.html>`_

Whydra supports continuous and discrete data and runs DAG-GNN on CPU or
a CUDA GPU. After training, ``graph_threshold`` removes edges with small
absolute weights. If a later step requires a DAG, check the resulting graph
for directed cycles.

Constraint-based and neural score-based methods
-----------------------------------------------

.. _tab-constraint-vs-nn:

.. list-table:: PC and FCI compared with DAG-GNN
   :header-rows: 1
   :widths: 27 36 37

   * - Criterion
     - PC and FCI
     - DAG-GNN
   * - Method
     - Conditional independence tests
     - Fits a generative model and optimizes edge weights.
   * - Complexity
     - Depends on skeleton density, conditioning set size, and test cost.
     - Depends on the architecture, number of variables, and number of
       optimization steps.
   * - Nonlinear dependencies
     - Depend on the chosen conditional independence test.
     - Depend on the architecture and data model.
   * - Noise
     - Accuracy depends on the properties of the test and the sample size.
     - Accuracy depends on the noise model, loss function, and regularization.
   * - Output
     - PC returns a CPDAG. FCI returns a PAG.
     - Returns a directed graph after thresholding the edge weights.
   * - Hidden common causes
     - FCI allows them under its assumptions. PC assumes causal sufficiency.
     - Whydra constructs the graph over the supplied variables.
   * - Edge interpretation
     - Endpoint marks show which aspects of edge orientation are identifiable.
     - Edges show the relationships selected by the trained model.
   * - Increasing the number of variables
     - The number of conditioning sets can grow rapidly.
     - Training uses a dense weight matrix with one row and one column per
       variable.
