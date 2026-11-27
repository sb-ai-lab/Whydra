whydra.algorithms.wrappers.run\_standalone\_algorithm
=====================================================

.. py:function:: whydra.run_standalone_algorithm(name, data_or_path, header=None, *, sep=None, delimiter=None, trusted_pickle=False, **kwargs)

   Creates an algorithm by name, prepares the input, runs the search,
   and returns the resulting graph. To access model attributes after a run, call
   :py:func:`whydra.create_standalone_algorithm` and then ``run`` on the model.

   :param name: A class name or factory alias.
   :param data_or_path: A NumPy array, DataFrame, or file path. J-PCMCI+ also
                        accepts a list, tuple, dictionary, or 3D array of environments.
   :param header: For a text file, ``True`` treats the first row as a header,
                  ``False`` treats it as data, and ``None`` detects the header automatically.
   :param sep: An explicit column separator for text tables.
   :param delimiter: Alias for ``sep``. If both are supplied, they must match.
   :param trusted_pickle: Allows reading trusted ``.pkl`` and ``.pickle`` files.
                           Deserialization can execute code.
   :param kwargs: Keyword arguments for the algorithm constructor.
   :returns: A ``GeneralGraph``.

Input Files
-----------

Whydra supports ``.npy`` files and text tables. When the loader reads the
first row as a header, it uses the column names as node names. For
``.pkl`` and ``.pickle`` files, set ``trusted_pickle=True`` and store
the data as a NumPy array.

.. code-block:: python

   from whydra import run_standalone_algorithm

   graph = run_standalone_algorithm(
       "pc_stable", "observations.csv", header=True, sep=",", alpha=0.01
   )
   print([str(edge) for edge in graph.get_graph_edges()])
