.. _intro:

Introduction
============

Causal discovery estimates causal structure from data. In a causal DAG
(directed acyclic graph), nodes represent variables and directed edges
represent direct causal relationships.

Observational data often identify only a class of graphs that imply the same
conditional independencies. PC-stable and GES return a CPDAG. A directed edge
has the same orientation in every DAG in the class. An undirected edge can
point in different directions in different DAGs. FCI allows for hidden common
causes and selection bias. It returns a PAG, which represents a Markov
equivalence class of maximal ancestral graphs (MAGs) over the observed
variables. A circle at one end of a PAG edge marks an endpoint whose type
varies across graphs in the class.

Causal discovery has applications in medicine, biology, economics, and the
social sciences. Two classes of algorithms are common.

- **Constraint-based methods**, which use conditional independence tests.
- **Score-based methods**, which evaluate models using a chosen score.

Classical methods for independent and identically distributed observations do
not account for temporal dependence or delays between cause and effect. Time
series methods use the order of the observations and the lags between them.

Algorithms in Whydra
--------------------

.. list-table:: Algorithms in Whydra
   :header-rows: 1
   :widths: 18 35 47

   * - Algorithm
     - Full name
     - Description
   * - :ref:`PC-stable <alg-pc-stable>`
     - Peter-Clark, stable variant
     - Fixes the adjacency sets at each depth of the skeleton search. Returns a CPDAG.
   * - :ref:`FCI-stable <alg-fci-stable>`
     - Fast Causal Inference, stable variant
     - Allows for hidden common causes and selection bias. Returns a PAG.
   * - :ref:`GFCI <alg-gfci>`
     - Greedy Fast Causal Inference
     - Starts from a GES graph and refines it with conditional independence
       tests. Returns a PAG.
   * - :ref:`RAI <alg-rai>`
     - Recursive Autonomy Identification
     - Experimental recursive search algorithm. Returns a CPDAG.
   * - :ref:`DAG-GNN <alg-dag-gnn>`
     - DAG Structure Learning with Graph Neural Networks
     - Uses a variational autoencoder with a graph neural network to estimate
       directed relationships.
   * - :ref:`GES <alg-ges>`
     - Greedy Equivalence Search
     - Searches greedily over equivalence classes. Returns a CPDAG and its score.
   * - `PCMCI <https://pmc.ncbi.nlm.nih.gov/articles/PMC6881151/>`_
     - PC + Momentary Conditional Independence
     - Searches for lagged relationships in a single time series.
   * - `PCMCI+ <https://proceedings.mlr.press/v124/runge20a.html>`_
     - PCMCI with contemporaneous relationships
     - Searches for lagged relationships and relationships among variables at
       the same time step.
   * - `J-PCMCI+ <https://proceedings.mlr.press/v216/gunther23a.html>`_
     - Joint PCMCI+
     - Searches jointly over time series from several environments.

See the :doc:`../tutorial` for examples with time series.
