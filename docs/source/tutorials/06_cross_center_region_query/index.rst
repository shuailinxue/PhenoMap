Cross-center phenotype region query
===================================

This function transfers phenotype-region queries across centers, cohorts or
annotation domains. It combines source-domain phenotype supervision with
optional target-domain ROI masks to improve cross-center localization.

.. figure:: ../../_static/figures/fig5_query.jpg
   :width: 90%
   :align: center
   :alt: Cross-center phenotype region query

   Example cross-domain phenotype-region query on annotated ROI data.

Input
-----

* Source H&E image, cell mask, cell expression and Step2 phenotype scores.
* Target H&E image folder.
* Optional target ROI masks.
* Cross-center training or inference config.

Train or adapt
--------------

.. code-block:: bash

   python step3/train_lightning.py \
     --mode train \
     --config step3/config/config_tiger_roi.yaml

Query target slides
-------------------

.. code-block:: bash

   python step3/train_lightning.py \
     --mode inference \
     --config step3/config/config_tiger_roi.yaml \
     --ckpt_path /path/to/last.ckpt

Main outputs
------------

* Cross-center phenotype score maps.
* ROI-level ranking metrics.
* ROC and precision-recall curves against target annotations.
* Ablation comparisons for prompt, text similarity and scGPT alignment.

Relevant files
--------------

* ``step3/config/config_tiger_roi.yaml``
* ``step3/config/config_her2_roi.yaml``
* ``step3/config_ablation/``
* ``step3/evaluate_tiger_roi_ablation.py``
