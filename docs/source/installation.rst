Installation
============

PhenoMap was developed on Linux workstations with NVIDIA A100 80GB GPUs. A CUDA-capable GPU
with at least 24 GB memory is recommended for histology feature extraction,
large graph training and whole-slide inference.

Clone repository
----------------

.. code-block:: bash

   git clone https://github.com/shuailinxue/PhenoMap.git
   cd PhenoMap

Create environment
------------------

.. code-block:: bash

   conda create --name phenomap python=3.11
   conda activate phenomap

Install dependencies
--------------------

.. code-block:: bash

   pip install -r requirements.txt

Documentation dependencies
--------------------------

To build the Sphinx documentation locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html

The generated HTML pages will be written to ``docs/build/html``.

Runtime notes
-------------

The development environment uses the ``cell2st`` conda environment. On the
original compute system, proxy variables are configured automatically by the
environment activation hook. Hugging Face tokens should be provided through
environment variables rather than committed to YAML configuration files.
