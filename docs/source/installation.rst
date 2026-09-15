Installation
============

Linux and a CUDA-capable GPU with at least 24 GB memory are recommended for
histology feature extraction, large graph training and whole-slide inference.

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

To regenerate the optional reference-transfer labels in Tutorial 02, install
the annotation extras:

.. code-block:: bash

   pip install -r requirements-annotation.txt

Documentation dependencies
--------------------------

To build the Sphinx documentation locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html

The generated HTML pages will be written to ``docs/build/html``.
