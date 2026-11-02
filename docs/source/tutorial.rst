
============================
Whydra Causal graph analysis
============================

Библиотека для построения причинных графов и оценки uplift-эффекта в прикладных задачах.

Содержание
----------

* :ref:`min-requirements`
* :ref:`installation`
* :ref:`optional-deps`
* :ref:`supported-algorithms`
* :ref:`benchmark-cases`
* :ref:`dev-env`
* :ref:`local-docs`

.. _min-requirements:

Минимальные требования
----------------------

* Python >= 3.8
* pip >= 23
* Рекомендуется использовать виртуальное окружение (``venv`` или ``conda``)

.. _installation:

Установка
---------

Установка из репозитория (основная ветка)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Предварительно обновляем ``pip``, ``setuptools``, ``wheel``:

.. code-block:: bash

    python3 -m pip install --upgrade pip setuptools wheel
    pip install setuptools
    pip install git+https://gitlab.appliedai.tech/causal_graph/modules/pipeline.git

Установка конкретной ветки или тега
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install git+https://gitlab.appliedai.tech/causal_graph/modules/pipeline.git@add_pip_install

.. _optional-deps:

Опциональные зависимости
------------------------

Graphviz (для визуализации графов)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Перед запуском убедитесь, что у вас установлен Graphviz в системе:

* **Linux:** ``sudo apt-get install graphviz``
* **Windows:** Скачайте инсталлятор с `сайта Graphviz <https://graphviz.org/download/>`_, установите и **добавьте путь к bin в PATH**.

PyTorch ( GPU-версия )
~~~~~~~~~~~~~~~~~~~~~~

Если вы используете алгоритмы или части пайплайна с поддержкой GPU:

.. code-block:: bash

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Если GPU не требуется, установите CPU-вариант PyTorch согласно официальной инструкции.

.. _supported-algorithms:

Поддерживаемые алгоритмы
------------------------

* pc_stable
* fci_stable 
* rai_stable 
* dag_gnn 
* nobears_stable 
* notears_stable 

Список неполон: PR добавил RFCI, GFCI, CFCI, FCI+ и кэшированные
и параллельные варианты FCI. Полная таблица алгоритмов с их именами —
в ``README.md``, раздел «Алгоритмы».

Для ``pc_stable``, ``fci_stable``, ``rai_stable`` параллелизм задаётся через ``n_jobs``. При ``n_jobs=1`` используется последовательный режим. У ``rai_stable`` результат от ``n_jobs`` не зависит: параллелятся только CI-тесты, граф при любом ``n_jobs`` совпадает с последовательным режимом. Для последовательного запуска используйте ``rai_stable(..., n_jobs=1)`` или ``StandaloneRAIStable(n_jobs=1)``; без явного параметра функция использует все ядра, а обёртка — четыре процесса.

.. _benchmark-cases:

Поддерживаемые benchmark-кейсы
------------------------------

* **bnlearn:** alarm, andes, asia, barley, cancer, child, earthquake, hailfinder, hepar2, insurance, sachs, survey, water, win95pts
* **feedback:** Network1_amp, Network2_amp, Network3_amp, Network4_amp, Network5_amp, Network5_cont, Network5_cont_p3n7, Network5_cont_p7n3, Network6_amp, Network6_cont, Network7_amp, Network7_cont, Network8_amp_amp, Network8_amp_cont, Network8_cont_amp, Network9_amp_amp, Network9_amp_cont, Network9_cont_amp
* **lucas:** lucas0, lucas1, lucas2
* **real:** htru2, pairs, sachs
* **tetrad:** tetrad_discrete_1 ... tetrad_discrete_10, tetrad_linear_1 ... tetrad_linear_10

.. _dev-env:

Окружение для разработки
------------------------

.. code-block:: bash

    conda create -n sber_env python=3.8.3 pip
    conda activate sber_env
    pip install -r requirements.txt

.. _local-docs:

Сборка локальной документации
-----------------------------

.. code-block:: bash

    sphinx-build -b html docs/source docs/build/html

Стартовая точка входа в документацию: ``docs/build/html/index.html``
