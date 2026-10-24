
===============================================
Кейс 1: Быстрый запуск на синтетических данных
===============================================

.. code-block:: python

   import numpy as np
   from whydra import run_standalone_algorithm

   rng = np.random.default_rng(42)
   data = rng.standard_normal((1000, 10))

   g1 = run_standalone_algorithm("pc_stable", data, alpha=0.05, indep_test="fisherz", n_jobs=4)
   print(f"Quickstart_ok=True graph_type:\n{g1}")


==================================
Кейс 2: Запуск на benchmark-данных
==================================

.. code-block:: python

   from whydra import BenchmarkCaseRepository, run_standalone_algorithm

   loader = BenchmarkCaseRepository()
   available_cases = list(loader.list_available_cases())
   print(f"available_cases={available_cases}")

   data2 = loader.get_case_data(case_name="alarm", benchmark_name="bnlearn")
   print("data2 = ", data2)

   g2 = run_standalone_algorithm("pc_stable", data2, alpha=0.05, indep_test="fisherz", n_jobs=4)
   print(f"Benchmark_ok=True graph_type:\n{g2}")