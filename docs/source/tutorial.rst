Causal Discovery with Whydra
============================

Whydra estimates causal structure from observations. In a table, rows
represent observations and columns represent variables. For time series,
order the rows chronologically.

Contents
--------

- :ref:`min-requirements`
- :ref:`installation`
- :ref:`optional-deps`
- :ref:`supported-algorithms`
- :ref:`benchmark-cases`

.. _min-requirements:

Minimum Requirements
--------------------

- Python 3.10 or later.
- Install with ``pip`` or ``uv``.
- You can isolate dependencies in a ``venv`` or ``conda`` environment.

.. _installation:

Installation
------------

Install from PyPI
~~~~~~~~~~~~~~~~~

The published Whydra package is available on `PyPI <https://pypi.org/project/whydra/>`_.
Run the command from any directory.

.. code-block:: console

   python -m pip install whydra

.. _optional-deps:

Optional Dependencies
---------------------

Graph Visualization
~~~~~~~~~~~~~~~~~~~

The ``.[viz]`` extra installs ``pydot``. Install Graphviz to visualize
graphs, then run ``dot -V`` in a terminal to check the installation. On
Debian and Ubuntu, you can install the ``graphviz`` package. On Windows,
add the Graphviz ``bin`` directory to ``PATH``.

.. code-block:: console

   python -m pip install ".[viz]"

`Graphviz installers <https://graphviz.org/download/>`_

Neural Network Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~

DAG-GNN requires PyTorch. The ``.[torch]`` extra installs it.

:doc:`Build and verify CUDA support for PC, FCI, and RAI <cuda>`.

.. code-block:: console

   python -m pip install ".[torch]"

.. _supported-algorithms:

Supported Algorithms
--------------------

- PC-stable and its parallel variant return a CPDAG.
- FCI-stable, CFCI, RFCI, GFCI, and FCI+ return a PAG.
- The experimental RAI-stable algorithm returns a CPDAG.
- DAG-GNN uses a neural network to estimate directed edges.
- PCMCI, PCMCI+, and J-PCMCI+ analyze time series.
- Call ``whydra.algorithms.score_based.ges_search`` to run GES.

:py:func:`whydra.available_standalone_algorithms` lists the class names.
See :doc:`api/modules` for algorithm aliases and result types.

On CPU, ``n_jobs=1`` selects sequential execution. Parallel support
depends on the algorithm and stage. RAI can run conditional independence
tests in parallel. It applies the results in the same order as the
sequential implementation. See :doc:`guides/cpu-performance` for details.

The examples below use rows for observations and columns for variables.
To inspect model attributes after a run, create the model with
:py:func:`whydra.create_standalone_algorithm`.

FCI with Hidden Common Causes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FCI returns a PAG. This graph represents uncertainty about edge
endpoint marks when some variables are unobserved. In this example, the
hidden common cause ``latent`` influences ``A`` and ``B``. The analysis
uses only the observed variables.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from whydra import create_standalone_algorithm

   rng = np.random.default_rng(17)
   latent, noise_a, noise_b, noise_c = rng.normal(size=(4, 1200))
   a = latent + noise_a
   b = latent + noise_b
   data = pd.DataFrame({"A": a, "B": b, "C": b + noise_c})

   model = create_standalone_algorithm("fci_stable", alpha=0.01)
   graph = model.run(data)
   print(graph.graph_kind)  # pag
   print([str(edge) for edge in graph.get_graph_edges()])

A circle at an edge endpoint means its mark is undetermined.

:doc:`PAG marks and assumptions of conditional independence search <theory/constraint_based>`.

Discrete Data and Background Knowledge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``indep_test="chisq"`` with categorical variables.
A background knowledge rule in this example forbids :math:`B \to A`.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from whydra import BackgroundKnowledge, StandalonePCStable

   rng = np.random.default_rng(19)
   a = rng.integers(0, 2, size=800)
   b = a ^ (rng.random(800) < 0.15)
   data = pd.DataFrame({"A": a, "B": b.astype(int)})

   knowledge = BackgroundKnowledge().forbid("B", "A")
   graph = StandalonePCStable(indep_test="chisq").run(
       data, background_knowledge=knowledge
   )
   print([str(edge) for edge in graph.get_graph_edges()])

Use node names in the rules. Save the rules with the run settings so
you can reproduce the result.

Causal Discovery in a Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Order time series observations chronologically. ``tau_max`` sets the
maximum lag in time steps between rows. In this example, ``x[t-1]``
affects ``y[t]``.
The ``lagged`` mode represents lags as separate nodes.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from whydra import create_standalone_algorithm

   rng = np.random.default_rng(23)
   x, noise = rng.normal(size=(2, 500))
   y = noise.copy()
   y[1:] += 0.9 * x[:-1]
   series = pd.DataFrame({"x": x, "y": y})

   model = create_standalone_algorithm("pcmci", tau_max=1, graph_mode="lagged")
   graph = model.run(series)
   print([str(edge) for edge in graph.get_graph_edges()])
   print(model.lag_graph_.shape, model.p_matrix_.shape)

``lag_graph_``, ``p_matrix_``, and ``val_matrix_`` store results by
variable pair and lag. Set ``graph_mode="summary"`` to collapse lags into
a graph over the variables. By default, PCMCI tests lags of at least one
step. PCMCI+ also tests contemporaneous links (``tau_min=0``).

Joint Search across Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

J-PCMCI+ analyzes multiple time series jointly. The tables must contain
the same set of columns. Whydra aligns columns by name and computes
lags within each environment.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from whydra import create_standalone_algorithm

   rng = np.random.default_rng(31)

   def series_in_environment(shift):
       x = rng.normal(loc=shift, size=400)
       y = rng.normal(size=400)
       y[1:] += 0.8 * x[:-1]
       return pd.DataFrame({"x": x, "y": y})

   series_by_environment = {
       "experiment_a": series_in_environment(-1),
       "experiment_b": series_in_environment(1),
   }
   model = create_standalone_algorithm("j_pcmci_plus", tau_max=1)
   graph = model.run(series_by_environment)

   print([str(edge) for edge in graph.get_graph_edges()])
   print(model.lag_graph_[0, 1, 1])  # Link from x[t-1] to y[t]

By default, J-PCMCI+ accounts for differences between environments
with ``dataset_dummy=True``. The model retains results by lag and
context. The returned graph shows links among the input variables.

GES Search
~~~~~~~~~~

GES compares objective function values. ``local_score_BIC`` is
available for continuous data and ``local_score_BDeu`` for discrete data.

.. code-block:: python

   import numpy as np

   from whydra.algorithms.score_based import ges_search

   rng = np.random.default_rng(29)
   x, noise = rng.normal(size=(2, 500))
   data = np.column_stack((x, 0.8 * x + noise))
   graph, score = ges_search(data, "local_score_BIC", ["x", "y"])

   print([str(edge) for edge in graph.get_graph_edges()])
   print(score)

Use the same data and objective function to compare graph scores.

:doc:`GES assumptions and search limits <theory/score_based>`.

.. _benchmark-cases:

Load Your Own Data
------------------

``BenchmarkCaseRepository`` reads external datasets. To create a local
example in bnlearn format, create ``cases/bnlearn/demo/input/data.txt``.

.. code-block:: text

   X1 X2
   0 1
   1 0

Next to ``input``, create a ``ground.truth`` directory. It can be empty
if you only want to load the table. Run the example from the directory
containing ``cases``.

.. code-block:: python

   from whydra.data import BenchmarkCaseRepository

   cases = BenchmarkCaseRepository(cases_root="cases")
   data = cases.get_case_data("bnlearn/demo")
   print(data.tolist())  # [[0, 1], [1, 0]]

See :doc:`api/generated/whydra.data.BenchmarkCaseRepository` for the
directory layout and loader methods. To run directly from a
CSV file with a header, use :py:func:`whydra.run_standalone_algorithm`.
