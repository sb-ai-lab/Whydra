whydra.data.BenchmarkCaseRepository
===================================

.. py:class:: whydra.BenchmarkCaseRepository(cases_root=None)

   Reads cases from a directory you prepare separately. The default is
   ``whydra_benchmarks/cases`` relative to the working directory. The
   Whydra package does not ship benchmark datasets.

   Create ``<root>/<family>/<case>/input`` and
   ``<root>/<family>/<case>/ground.truth`` for each case. You can store
   metadata in ``case.json`` alongside those directories.

   .. py:method:: list_available_cases(benchmark_name=None)

      Returns sorted case identifiers. When the benchmark family is omitted,
      identifiers are relative to the root, for example ``bnlearn/child``.
      When you specify a family, identifiers are relative to its directory.

   .. py:method:: get_case_datasets(case_name=None, benchmark_name=None)

      Returns a dictionary of arrays loaded for the case. If several cases
      share a short name, pass the full identifier or specify the benchmark
      family.

   .. py:method:: get_case_data(case_name, benchmark_name=None, dataset_key=None)

      Returns one array. When ``dataset_key`` is omitted, the method takes
      the first dataset from the dictionary. Pass the key explicitly for
      reproducible runs.

   .. py:method:: get_case_ground_truth(case_name, benchmark_name=None)

      Returns the path to the ground-truth graph file as a string. Read
      the graph and align its node order separately.

   .. py:method:: get_case_metadata(case_name, benchmark_name=None)

      Returns the benchmark family, data type, ground truth, and node order
      when available. You can refine these values in ``case.json``.
      If Whydra cannot infer the data type, set ``data_type`` to
      ``"continuous"`` or ``"categorical"``.

.. code-block:: python

   from whydra import BenchmarkCaseRepository

   repo = BenchmarkCaseRepository("benchmarks/cases")
   for case in repo.list_available_cases():
       print(case, list(repo.get_case_datasets(case)))
