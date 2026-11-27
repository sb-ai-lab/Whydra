Quickstart
==========

Example 1. Run on Synthetic Data
--------------------------------

PC-stable takes a table of observations. In this example, ``X`` and
``Y`` independently affect ``Z``. The DataFrame supplies variable names,
and the Fisher-Z test (``fisherz``) is suitable for continuous data.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from whydra import run_standalone_algorithm

   rng = np.random.default_rng(42)
   x, y, noise = rng.normal(size=(3, 1000))
   data = pd.DataFrame({"X": x, "Y": y, "Z": 2 * x + y + noise})

   graph = run_standalone_algorithm(
       "pc_stable", data, alpha=0.01, indep_test="fisherz", n_jobs=1
   )
   print(graph.graph_kind)
   print(sorted(map(str, graph.get_graph_edges())))

For this sample, the algorithm returns a CPDAG with edges ``X --> Z`` and
``Y --> Z``. In real applications, whether an edge can be oriented
unambiguously depends on the sample, the assumptions of the method, and
the conditional independence test.

Example 2. Run on Benchmark Data
--------------------------------

Set up a local case directory as described in :ref:`benchmark-cases`.
This example uses the discrete ``bnlearn/alarm`` case. Supply your
own directory path and replace the case name as needed.

.. code-block:: python

   from whydra import BenchmarkCaseRepository, run_standalone_algorithm

   cases = BenchmarkCaseRepository(cases_root="cases")
   print(cases.list_available_cases())

   data = cases.get_case_data(case_name="alarm", benchmark_name="bnlearn")
   graph = run_standalone_algorithm(
       "pc_stable", data, alpha=0.05, indep_test="chisq", n_jobs=1
   )
   print([str(edge) for edge in graph.get_graph_edges()])

Use ``chisq`` for categorical data. If a case contains multiple tables,
pass ``dataset_key`` to select one explicitly.
